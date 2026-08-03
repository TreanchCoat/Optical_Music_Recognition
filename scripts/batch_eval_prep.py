r"""
batch_eval_prep.py — Run the OMR pipeline over a piece-organised image
folder and lay out the results for `evaluation/eval_xml.py batch`.

Expected input layout
---------------------
The root contains one subfolder per piece, and each subfolder contains
that piece's page images::

    <INPUT_ROOT>/
        01 Okribuli makruli/
            page1.png
            page2.png
            ...
        02 Something Else/
            page1.png
            ...

Output layout
-------------
This script writes, for each piece::

    <OUTPUT_ROOT>/
        01 Okribuli makruli/
            page1.xml          ← score.xml produced by the pipeline
            page2.xml
            ...

…which is exactly the layout `eval_xml.py batch --pred-root <OUTPUT_ROOT>`
expects.  Filenames are preserved from the input images (just the
extension swapped to .xml), so per-page traceability is intact.

Behaviour
---------
* Resumable.  If `<OUTPUT_ROOT>/<piece>/<page>.xml` already exists, that
  page is skipped.  Re-run safely after a crash or partial run.
* Per-page intermediate artifacts (rectified.png, detections.json,
  labeled_crops/, binarized/, …) are written into a `_tmp_<page>/`
  folder under the piece directory while the pipeline runs, then
  cleaned up.  Pass `--keep-artifacts` to keep them for inspection.
* Failures on one page don't kill the batch — the error is logged and
  the script moves to the next image.

Default paths
-------------
The script's defaults point at the Khalkhuri / Abesadzeebi subset of
the user's corpus (the case being tested first).  Override with CLI
args for any other folder.

Usage
-----
    # Use defaults
    python scripts/batch_eval_prep.py

    # Custom paths
    python scripts/batch_eval_prep.py \
        --input  "E:\Thesis\pdf\...\some other piece folder" \
        --output "E:\Thesis\eval_out\some_subset"

    # Use a different model (e.g. the new Georgian-tuned weights)
    python scripts/batch_eval_prep.py \
        --model  "S:\omr\models\georgian_v2.pt"

    # Keep all intermediate artifacts for diagnostic inspection
    python scripts/batch_eval_prep.py --keep-artifacts
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

# Bootstrap — make src/ and scripts/ importable regardless of where this
# script was launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_PROJECT_ROOT / 'scripts'))
import env_loader  # noqa: E402  — loads .env + finishes path setup


# ─────────────────────────────────────────────────────────────────────────────
# Defaults — point at the first subset to evaluate
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT_ROOT = Path(
    r'E:\Thesis\pdf\output\PDF Files-20260310T194030Z-1-001'
    r'\PDF Files\Khalkhuri\Abesadzeebi'
)
DEFAULT_OUTPUT_ROOT = Path(r'S:\omr\eval_out\Abesadzeebi')

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline import (lazy — ultralytics + torch take a moment to load)
# ─────────────────────────────────────────────────────────────────────────────

def _load_run_pipeline():
    """Import run_pipeline lazily so --help is instant."""
    from pipeline import run_pipeline  # noqa: E402
    return run_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _process_page(run_pipeline,
                  image_path: Path,
                  pred_piece_dir: Path,
                  model_path: str,
                  keep_artifacts: bool,
                  ) -> str:
    """
    Run the pipeline on one image; move score.xml to ``pred_piece_dir/<stem>.xml``.

    Returns a short status string ('OK', 'SKIP', 'NO_XML', or 'FAIL: ...').
    """
    page_name = image_path.stem
    target_xml = pred_piece_dir / f'{page_name}.xml'
    if target_xml.is_file():
        return 'SKIP'

    tmp_dir = pred_piece_dir / f'_tmp_{page_name}'
    try:
        run_pipeline(
            image_path=str(image_path),
            output_dir=str(tmp_dir),
            model_path=model_path,
            save_binarized_crops=False,   # batch eval: skip diagnostic artifacts
            save_labeled_crops=False,     # (keeps _tmp tiny, avoids filling disk)
        )
    except Exception as e:
        # Leave tmp dir for inspection so the user can see what happened.
        return f'FAIL: {type(e).__name__}: {e}'

    produced_xml = tmp_dir / 'score.xml'
    if not produced_xml.is_file():
        return 'NO_XML'

    # Move score.xml up to <piece>/<page>.xml
    shutil.move(str(produced_xml), str(target_xml))

    # Clean up intermediates unless --keep-artifacts
    if not keep_artifacts:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        # Rename _tmp_<page>/ → <page>_artifacts/ so the user can find them
        artifacts_dir = pred_piece_dir / f'{page_name}_artifacts'
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir, ignore_errors=True)
        tmp_dir.rename(artifacts_dir)

    return 'OK'


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog='batch_eval_prep',
        description='Run pipeline over a piece-organised image folder; '
                    'arrange XMLs for eval_xml batch mode.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--input',  default=str(DEFAULT_INPUT_ROOT),
                    help=f'Root folder with one subdir per piece.  '
                         f'Default: {DEFAULT_INPUT_ROOT}')
    ap.add_argument('--output', default=str(DEFAULT_OUTPUT_ROOT),
                    help=f'Where to write pred XMLs (mirrors the piece '
                         f'subfolder structure).  Default: {DEFAULT_OUTPUT_ROOT}')
    ap.add_argument('--model',  default=str(env_loader.MODEL_PATH),
                    help=f'YOLO weights .pt path.  Default (from .env): '
                         f'{env_loader.MODEL_PATH}')
    ap.add_argument('--keep-artifacts', action='store_true',
                    help='Keep per-page intermediate outputs (rectified.png, '
                         'detections.json, labeled_crops/, binarized/, '
                         'etc.) under <page>_artifacts/.  Useful for '
                         'diagnostics; default is to delete them.')
    ap.add_argument('--dry-run', action='store_true',
                    help="Don't run the pipeline; just list what would be "
                         "processed and the output paths.")
    ap.add_argument('--limit', type=int, default=0,
                    help='Process only the first N piece folders (0 = all). '
                         'Useful for a quick smoke run on a few pieces.')
    ap.add_argument('--offset', type=int, default=0,
                    help='Skip the first N piece folders.  Combine with --limit '
                         'to split work across machines: machine A runs '
                         '--limit K, machine B runs --offset K.  Both machines '
                         'must see the same input tree so the ordering matches.')
    ap.add_argument('--num-shards', type=int, default=1,
                    help='Split pieces into N interleaved shards for parallel '
                         'workers.  Worker i runs --shard i --num-shards N and '
                         'takes every Nth piece, so scattered/remaining work '
                         'spreads evenly (combined with SKIP, ideal for '
                         'rebalancing the slow tail of a run).')
    ap.add_argument('--shard', type=int, default=0,
                    help='Which interleaved shard this worker handles '
                         '(0..num_shards-1).')
    args = ap.parse_args(argv)

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    model_path = str(Path(args.model).resolve())

    if not input_root.is_dir():
        print(f'ERROR: input folder not found: {input_root}', file=sys.stderr)
        return 2
    if not Path(model_path).is_file():
        print(f'ERROR: model not found: {model_path}', file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)

    # Discover "piece" folders: any directory (at any depth) that directly
    # contains at least one image file.  The corpus nests pieces several
    # levels deep, e.g.
    #   PDF Files/<category>/<album>/<collection>/<piece>/img_1.png
    # so a single-level scan of input_root is not enough.
    def _find_piece_dirs(root: Path) -> List[Path]:
        found = []
        candidates = [root, *(p for p in root.rglob('*') if p.is_dir())]
        for d in candidates:
            if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS
                   for p in d.iterdir()):
                found.append(d)
        return sorted(found)

    piece_dirs = _find_piece_dirs(input_root)
    if args.num_shards and args.num_shards > 1:
        piece_dirs = piece_dirs[args.shard::args.num_shards]
        print(f'--shard {args.shard}/{args.num_shards}: processing '
              f'{len(piece_dirs)} interleaved piece folder(s).')
    if args.offset and args.offset > 0:
        skipped = min(args.offset, len(piece_dirs))
        piece_dirs = piece_dirs[args.offset:]
        print(f'--offset {args.offset}: skipping the first {skipped} '
              f'piece folder(s); {len(piece_dirs)} remain.')
    if args.limit and args.limit > 0:
        piece_dirs = piece_dirs[:args.limit]
        print(f'--limit {args.limit}: processing only the first '
              f'{len(piece_dirs)} piece folder(s).')
    if not piece_dirs:
        print(f'ERROR: no folders containing images found under {input_root}',
              file=sys.stderr)
        return 1

    # Per-piece image counts up front so the user knows scope.  Output folders
    # are named by the leaf folder (so they match the same-named GT XML).
    # Many pieces share a leaf name (e.g. "01 ..."), so disambiguate collisions
    # to avoid predictions overwriting each other.
    total_pages = 0
    plan: List[tuple] = []
    used_names: dict = {}
    for piece in piece_dirs:
        images = sorted(p for p in piece.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not images:
            continue
        out_name = piece.name
        if out_name in used_names:
            used_names[out_name] += 1
            disambiguated = f'{out_name}__{used_names[out_name]}'
            print(f'  [warn] duplicate piece name "{out_name}" -> '
                  f'"{disambiguated}" (GT pairing will not match for this copy)',
                  file=sys.stderr)
            out_name = disambiguated
        else:
            used_names[out_name] = 1
        plan.append((piece, images, out_name))
        total_pages += len(images)

    print(f'Input : {input_root}')
    print(f'Output: {output_root}')
    print(f'Model : {model_path}')
    print(f'Pieces: {len(piece_dirs)}    Total pages: {total_pages}')
    if args.dry_run:
        for piece, images, out_name in plan:
            print(f'  {out_name}: {len(images)} image(s)')
            for img in images:
                print(f'    {img.name} -> {output_root / out_name / (img.stem + ".xml")}')
        return 0

    print(f'Loading pipeline (this takes a few seconds — ultralytics + torch)...')
    run_pipeline = _load_run_pipeline()
    print(f'Ready.  Starting batch.\n')

    # Stats
    n_ok = n_skip = n_no_xml = n_fail = 0
    t_start = time.time()

    for piece_idx, (piece, images, out_name) in enumerate(plan, 1):
        piece_name = out_name
        pred_piece_dir = output_root / piece_name
        pred_piece_dir.mkdir(parents=True, exist_ok=True)

        print(f'[{piece_idx}/{len(plan)}] {piece_name}  ({len(images)} page(s))')

        for j, img in enumerate(images, 1):
            t0 = time.time()
            try:
                status = _process_page(
                    run_pipeline=run_pipeline,
                    image_path=img,
                    pred_piece_dir=pred_piece_dir,
                    model_path=model_path,
                    keep_artifacts=args.keep_artifacts,
                )
            except KeyboardInterrupt:
                print('\nInterrupted by user.')
                _print_summary(n_ok, n_skip, n_no_xml, n_fail, t_start)
                return 130
            except Exception as e:
                status = f'FAIL: {type(e).__name__}: {e}'
                traceback.print_exc(limit=2)

            dt = time.time() - t0
            print(f'    [{j:>3d}/{len(images):<3d}] {img.name:40s}  '
                  f'{status:30s}  {dt:6.1f}s')

            if status == 'OK':
                n_ok += 1
            elif status == 'SKIP':
                n_skip += 1
            elif status == 'NO_XML':
                n_no_xml += 1
            else:
                n_fail += 1

    _print_summary(n_ok, n_skip, n_no_xml, n_fail, t_start)
    print()
    print(f'Next: run the evaluation —')
    print(f'  python evaluation\\eval_xml.py batch \\')
    print(f'      --pred-root  "{output_root}" \\')
    print(f'      --gt-root    "<path to matching Finale XMLs>" \\')
    print(f'      --out        "{output_root.parent}\\eval_results.csv" \\')
    print(f'      --verbose')
    return 0


def _print_summary(n_ok: int, n_skip: int, n_no_xml: int,
                   n_fail: int, t_start: float) -> None:
    total = n_ok + n_skip + n_no_xml + n_fail
    elapsed = time.time() - t_start
    print()
    print('─' * 70)
    print(f'Done in {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'  OK     : {n_ok}')
    print(f'  SKIP   : {n_skip}  (already present)')
    print(f'  NO_XML : {n_no_xml}  (pipeline ran but produced no score.xml)')
    print(f'  FAIL   : {n_fail}')
    print(f'  TOTAL  : {total}')


if __name__ == '__main__':
    sys.exit(main())
