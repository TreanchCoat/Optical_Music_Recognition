"""
pseudo_label.py — Generate YOLOv8-format training labels from an existing
YOLO model.

Pipeline position
-----------------
1. `scripts/preprocess_for_training.py` produces cleaned staff crops in
   ``PREP_DIR/images/`` plus a ``manifest.csv``.
2. **This script** runs ``deepscores_crops_v1.pt`` (or any model you pass)
   over those crops and writes one ``<stem>.txt`` per image to
   ``PREP_DIR/labels/`` in YOLO format::

       <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

3. ``training/yolo_train.py`` fine-tunes a fresh model from the resulting
   images+labels pair.

Side artefacts written to ``PREP_DIR/``:

* ``classes.txt``                 — one class name per line (index = class_id).
* ``data.yaml``                   — Ultralytics dataset config with train/val
                                    split pointing at ``images/`` and ``labels/``.
* ``pseudo_label_manifest.csv``   — per-image: stem, n_detections, time_s.
* ``pseudo_label_errors.log``     — images that failed inference.

Usage
-----
    python pseudo_label.py PREP_DIR [options]

Examples
--------
    # Pseudo-label the handwritten preprocessing output
    python pseudo_label.py S:\\omr\\training_data\\handwritten

    # Custom model + lower confidence threshold + 95/5 split
    python pseudo_label.py S:\\omr\\training_data\\printed \\
        --model S:\\omr\\models\\deepscores_crops_v1.pt \\
        --conf 0.20 --val-fraction 0.05

    # Drop the same `staff` / `staffLine` classes the pipeline filters
    # at inference time (default behaviour — disable with --keep-staff).
    python pseudo_label.py S:\\omr\\training_data\\handwritten --keep-staff

    # Resume — skip images that already have a .txt in labels/.
    python pseudo_label.py S:\\omr\\training_data\\handwritten --resume

Notes
-----
* The model name list is read from the loaded ``.pt`` (Ultralytics ``model.names``).
  Inheriting from ``deepscores_crops_v1.pt`` keeps the existing class
  set so the trained model is drop-in compatible with the rest of the
  pipeline.
* The default confidence threshold (0.25) matches ``symbol_detector``.
  Lower it for higher recall (more permissive auto-labels); raise it
  for cleaner labels at the cost of missed symbols.
* The val split is randomised by stem hash so the same image always
  lands in the same split, even if the script is re-run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

# Bootstrap: this script lives at <project>/training/, so the project root
# is its parent directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
import env_loader  # noqa: E402

import cv2  # noqa: E402


# Classes that ``symbol_detector`` filters at inference time.  Keep them
# out of the training labels too so the new model doesn't learn to
# emit them.  Imported from src/ so this stays in sync with the inference
# pipeline (single source of truth).
from symbol_detector import IGNORED_YOLO_CLASSES as IGNORED_CLASSES_DEFAULT  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_val(stem: str, val_fraction: float) -> bool:
    """
    Deterministic train/val split by stem hash.

    Same stem → same bucket across runs.  Uses md5 of the stem mod 1000
    so the granularity is 0.1 %.
    """
    h = int(hashlib.md5(stem.encode('utf-8')).hexdigest()[:8], 16) % 1000
    return h < val_fraction * 1000


def _load_existing_labels(labels_dir: Path) -> set:
    """Return stems whose .txt already exists (for --resume)."""
    if not labels_dir.is_dir():
        return set()
    return {p.stem for p in labels_dir.glob('*.txt')}


def _xyxy_to_yolo(x1, y1, x2, y2, W, H):
    """Convert pixel-space (x1,y1,x2,y2) to normalized (cx,cy,w,h)."""
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w  = (x2 - x1)       / W
    h  = (y2 - y1)       / H
    return cx, cy, w, h


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog='pseudo_label',
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('prep_dir',
                    help='Output directory of preprocess_for_training.py. '
                         'Must contain an images/ subfolder.')
    ap.add_argument('--model', default=str(env_loader.MODEL_PATH),
                    help=f'YOLO .pt weights to use for pseudo-labeling. '
                         f'Defaults to env_loader.MODEL_PATH ({env_loader.MODEL_PATH}).')
    ap.add_argument('--conf', type=float, default=0.25,
                    help='Confidence threshold (default 0.25 — matches '
                         'symbol_detector).')
    ap.add_argument('--imgsz', type=int, default=1280,
                    help='Inference size (default 1280 — matches the '
                         'pipeline; thin symbols need the resolution).')
    ap.add_argument('--device', default=None,
                    help="Ultralytics device string (e.g. '0', 'cpu'). "
                         'Default: auto-detect.')
    ap.add_argument('--val-fraction', type=float, default=0.10,
                    help='Fraction of images put in the val split '
                         '(default 0.10).  Deterministic by stem hash.')
    ap.add_argument('--keep-staff', action='store_true',
                    help='Keep `staff` / `staffLine` classes in the '
                         'labels (default: drop them, matching '
                         'symbol_detector behaviour).')
    ap.add_argument('--resume', action='store_true',
                    help='Skip images whose .txt already exists in '
                         'labels/.')
    ap.add_argument('--limit', type=int, default=0,
                    help='Stop after this many images (0 = no limit).')
    args = ap.parse_args(argv)

    prep_dir = Path(args.prep_dir).resolve()
    images_dir = prep_dir / 'images'
    labels_dir = prep_dir / 'labels'

    if not images_dir.is_dir():
        print(f'images/ subfolder not found under {prep_dir}', file=sys.stderr)
        print('Did you run preprocess_for_training.py first?', file=sys.stderr)
        return 2

    labels_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        print(f'Model not found: {model_path}', file=sys.stderr)
        return 2

    # Lazy import — keeps the bootstrap cheap.
    from ultralytics import YOLO

    print(f'Loading model: {model_path}')
    model = YOLO(str(model_path))
    class_names = model.names  # dict[int, str]
    if not isinstance(class_names, dict):
        # Some Ultralytics versions return a list — normalise.
        class_names = {i: n for i, n in enumerate(class_names)}

    n_classes = len(class_names)
    print(f'Model exposes {n_classes} classes.')

    ignored = set() if args.keep_staff else set(IGNORED_CLASSES_DEFAULT)
    if ignored:
        print(f'Filtering classes: {sorted(ignored)}')

    # Write classes.txt (index-aligned to model.names).
    classes_txt = prep_dir / 'classes.txt'
    with classes_txt.open('w', encoding='utf-8') as f:
        for i in range(n_classes):
            f.write(class_names[i] + '\n')
    print(f'Wrote {classes_txt}')

    # Discover input images.
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
    )
    print(f'Found {len(image_paths)} image(s) in {images_dir}.')

    if args.resume:
        done = _load_existing_labels(labels_dir)
        if done:
            image_paths = [p for p in image_paths if p.stem not in done]
            print(f'Resume mode: {len(done)} labels already exist; '
                  f'{len(image_paths)} remaining.')

    if args.limit:
        image_paths = image_paths[:args.limit]

    if not image_paths:
        print('Nothing to do.')
        # Even with nothing to do, regenerate data.yaml below so partial
        # runs stay consistent.

    manifest_path = prep_dir / 'pseudo_label_manifest.csv'
    errors_path   = prep_dir / 'pseudo_label_errors.log'

    n_done = 0
    n_failed = 0
    total_dets = 0
    t_start = time.perf_counter()

    write_manifest_header = (
        not manifest_path.exists() or manifest_path.stat().st_size == 0
    )
    with manifest_path.open('a', newline='', encoding='utf-8') as mf, \
         errors_path.open('a', encoding='utf-8') as ef:
        writer = csv.writer(mf)
        if write_manifest_header:
            writer.writerow(['stem', 'n_detections', 'n_kept',
                             'split', 'time_s'])

        for idx, img_path in enumerate(image_paths, start=1):
            stem = img_path.stem
            t0 = time.perf_counter()
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise RuntimeError(f'Cannot read image: {img_path}')
                H, W = img.shape[:2]

                # Ensure 3-channel — YOLO expects BGR.
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                kwargs = dict(conf=args.conf, imgsz=args.imgsz, verbose=False)
                if args.device is not None:
                    kwargs['device'] = args.device
                results = model(img, **kwargs)[0]

                n_raw = 0
                kept_lines: List[str] = []
                if results.boxes is not None and len(results.boxes) > 0:
                    xyxy    = results.boxes.xyxy.cpu().numpy()
                    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
                    n_raw = len(cls_ids)
                    for (x1, y1, x2, y2), cls_id in zip(xyxy, cls_ids):
                        name = class_names.get(int(cls_id), str(int(cls_id)))
                        if name in ignored:
                            continue
                        cx, cy, w, h = _xyxy_to_yolo(x1, y1, x2, y2, W, H)
                        # Clamp into [0, 1] just in case YOLO returned a
                        # slightly out-of-frame box.
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        w  = max(0.0, min(1.0, w))
                        h  = max(0.0, min(1.0, h))
                        if w <= 0 or h <= 0:
                            continue
                        kept_lines.append(
                            f'{int(cls_id)} {cx:.6f} {cy:.6f} '
                            f'{w:.6f} {h:.6f}'
                        )

                label_path = labels_dir / f'{stem}.txt'
                # Always write the file, even if empty — Ultralytics treats
                # the absence of a label file as "skip" rather than "negative".
                with label_path.open('w', encoding='utf-8') as lf:
                    lf.write('\n'.join(kept_lines))
                    if kept_lines:
                        lf.write('\n')

                split = 'val' if _is_val(stem, args.val_fraction) else 'train'
                dt = time.perf_counter() - t0
                writer.writerow([stem, n_raw, len(kept_lines), split,
                                 f'{dt:.3f}'])
                mf.flush()
                n_done += 1
                total_dets += len(kept_lines)

                if idx % 25 == 0 or idx == len(image_paths):
                    elapsed = time.perf_counter() - t_start
                    rate = n_done / max(elapsed, 1e-9)
                    eta = (len(image_paths) - idx) / max(rate, 1e-9)
                    print(f'  [{idx}/{len(image_paths)}] '
                          f'{stem}  kept={len(kept_lines)}/{n_raw}  '
                          f'rate={rate:.1f} img/s  '
                          f'eta={eta/60:.1f}m', flush=True)

            except Exception as e:  # noqa: BLE001
                n_failed += 1
                ef.write(f'{stem}\t{type(e).__name__}: {e}\n')
                ef.write(traceback.format_exc() + '\n')
                ef.flush()
                print(f'  [{idx}/{len(image_paths)}] FAILED {stem}: '
                      f'{e}', file=sys.stderr, flush=True)

    elapsed = time.perf_counter() - t_start
    print()
    print(f'Pseudo-labeled {n_done} image(s) in {elapsed/60:.1f} min; '
          f'{n_failed} failures, {total_dets} total detections kept.')

    # ─── Write data.yaml for the trainer ────────────────────────────
    # Train/val are filtered by per-stem hash inside the trainer.  The
    # simplest robust setup is to point Ultralytics at the same images/
    # dir for both splits and rely on `train.txt` / `val.txt` listing
    # files for each split.  Generate those here so the split is stable.
    train_list = prep_dir / 'train.txt'
    val_list   = prep_dir / 'val.txt'
    all_imgs = sorted(images_dir.iterdir())
    n_train = n_val = 0
    with train_list.open('w', encoding='utf-8') as ftr, \
         val_list.open('w',   encoding='utf-8') as fvl:
        for p in all_imgs:
            if p.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            # Only include images that have a (possibly empty) label.
            if not (labels_dir / f'{p.stem}.txt').exists():
                continue
            target = fvl if _is_val(p.stem, args.val_fraction) else ftr
            target.write(str(p.resolve()) + '\n')
            if target is fvl:
                n_val += 1
            else:
                n_train += 1
    print(f'Split: train={n_train}  val={n_val}  '
          f'(val_fraction={args.val_fraction:.2%}, stem-hash deterministic).')

    data_yaml = prep_dir / 'data.yaml'
    with data_yaml.open('w', encoding='utf-8') as f:
        f.write('# Auto-generated by training/pseudo_label.py\n')
        f.write(f'# Source preprocessing dir: {prep_dir}\n')
        f.write(f'# Source model:             {model_path}\n')
        f.write('path: ' + str(prep_dir).replace('\\', '/') + '\n')
        f.write('train: ' + str(train_list).replace('\\', '/') + '\n')
        f.write('val: '   + str(val_list  ).replace('\\', '/') + '\n')
        f.write('names:\n')
        for i in range(n_classes):
            # YAML-escape the name minimally — class names are alphanumeric
            # in DeepScores so this is safe.
            f.write(f'  {i}: {class_names[i]}\n')
    print(f'Wrote {data_yaml}')

    print()
    print('Next step:')
    print(f'  python training/yolo_train.py --data {data_yaml}')

    return 0 if n_failed == 0 else 3


if __name__ == '__main__':
    sys.exit(main())
