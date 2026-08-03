"""
build_data_yaml.py — Build a YOLOv8 dataset config from a folder of
already-cropped + already-labeled staff images.

Pipeline position
-----------------
Use this INSTEAD of ``pseudo_label.py`` when the dataset is already
labeled (no inference needed).

Expects a flat folder with paired files::

    DATASET_DIR/
        img_1_staff01.png    img_1_staff01.txt
        img_1_staff02.png    img_1_staff02.txt
        ...

Each ``.txt`` is YOLO-format (one detection per line:
``class_id  cx_norm  cy_norm  w_norm  h_norm``).

Writes the following next to the dataset::

    DATASET_DIR/train.txt        absolute image paths, train split
    DATASET_DIR/val.txt          absolute image paths, val split
    DATASET_DIR/classes.txt      class names indexed by id (one per line)
    DATASET_DIR/data.yaml        Ultralytics dataset config

Usage
-----
    python build_data_yaml.py DATASET_DIR [options]

Example
-------
    # Build config for the 15 100-image labeled corpus, defaulting class
    # names to the ones baked into deepscores_crops_v1.pt.
    python build_data_yaml.py S:\\training_images_crops

    # Custom split, custom name source, explicit output dir
    python build_data_yaml.py S:\\training_images_crops \\
        --val-fraction 0.05 \\
        --names-from S:\\omr\\models\\deepscores_crops_v1.pt \\
        --output S:\\training_images_crops

Notes
-----
* The split is deterministic by image stem hash (same as
  ``pseudo_label.py``) — re-running won't shuffle.
* Images without a corresponding .txt are skipped with a warning;
  pass ``--allow-empty-labels`` to include them as "negative samples"
  (Ultralytics treats a missing .txt as background — usually fine).
* Class names default to ``model.names`` from ``MODEL_PATH`` (i.e.
  ``deepscores_crops_v1.pt``).  Override with ``--names-from`` to point
  at a different .pt, or with ``--names-file`` to read a plain-text
  list (one name per line).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Bootstrap so env_loader / src/ are reachable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
import env_loader  # noqa: E402


def _is_val(stem: str, val_fraction: float) -> bool:
    """Deterministic train/val split by stem hash — same as pseudo_label.py."""
    h = int(hashlib.md5(stem.encode('utf-8')).hexdigest()[:8], 16) % 1000
    return h < val_fraction * 1000


def _read_names_from_pt(pt_path: Path) -> Dict[int, str]:
    """Return ``{class_id: name}`` from a YOLO .pt file via Ultralytics."""
    from ultralytics import YOLO
    print(f'Loading class names from {pt_path} …')
    model = YOLO(str(pt_path))
    names = model.names
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return names


def _read_names_from_file(txt_path: Path) -> Dict[int, str]:
    """Plain-text fallback — one class name per line, index = line number."""
    with txt_path.open('r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return {i: name for i, name in enumerate(lines)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog='build_data_yaml',
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('dataset_dir',
                    help='Folder containing paired .png/.txt files.')
    ap.add_argument('--output', default=None,
                    help='Where to write data.yaml / train.txt / val.txt / '
                         'classes.txt.  Defaults to DATASET_DIR.')
    ap.add_argument('--val-fraction', type=float, default=0.10,
                    help='Fraction of images put in the val split '
                         '(default 0.10).  Deterministic by stem hash.')
    ap.add_argument('--names-from', default=str(env_loader.MODEL_PATH),
                    help=f'YOLO .pt to extract class names from. '
                         f'Default: {env_loader.MODEL_PATH}.')
    ap.add_argument('--names-file', default=None,
                    help='Plain-text class list (one per line) — overrides '
                         '--names-from when provided.')
    ap.add_argument('--allow-empty-labels', action='store_true',
                    help='Include images that have no matching .txt '
                         '(Ultralytics treats them as background).')
    ap.add_argument('--ext', default='png,jpg,jpeg',
                    help='Comma-separated image extensions to look for.')
    args = ap.parse_args(argv)

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(f'Dataset dir not found: {dataset_dir}', file=sys.stderr)
        return 2

    output_dir = Path(args.output).resolve() if args.output else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover paired files ────────────────────────────────────────
    exts = {e.strip().lower().lstrip('.') for e in args.ext.split(',') if e.strip()}
    print(f'Scanning {dataset_dir} for *.{{{",".join(sorted(exts))}}} …')
    images: List[Path] = []
    for p in dataset_dir.iterdir():
        if p.is_file() and p.suffix.lower().lstrip('.') in exts:
            images.append(p)
    images.sort()
    print(f'Found {len(images)} image candidate(s).')

    paired: List[Path] = []
    n_missing = 0
    for img in images:
        label = img.with_suffix('.txt')
        if label.is_file():
            paired.append(img)
        elif args.allow_empty_labels:
            paired.append(img)
        else:
            n_missing += 1
    if n_missing:
        print(f'  skipped {n_missing} image(s) with no matching .txt '
              f'(use --allow-empty-labels to include them).')
    if not paired:
        print('No usable (image, label) pairs found.', file=sys.stderr)
        return 1
    print(f'Usable pairs: {len(paired)}.')

    # ── Resolve class names ──────────────────────────────────────────
    if args.names_file:
        names_path = Path(args.names_file).resolve()
        if not names_path.is_file():
            print(f'--names-file not found: {names_path}', file=sys.stderr)
            return 2
        class_names = _read_names_from_file(names_path)
        print(f'Loaded {len(class_names)} class names from {names_path}.')
    else:
        pt_path = Path(args.names_from).resolve()
        if not pt_path.is_file():
            print(f'--names-from .pt not found: {pt_path}', file=sys.stderr)
            return 2
        class_names = _read_names_from_pt(pt_path)
        print(f'Loaded {len(class_names)} class names from {pt_path}.')

    n_classes = max(class_names.keys()) + 1 if class_names else 0
    if n_classes == 0:
        print('No class names found.', file=sys.stderr)
        return 1

    # ── Write classes.txt ────────────────────────────────────────────
    classes_path = output_dir / 'classes.txt'
    with classes_path.open('w', encoding='utf-8') as f:
        for i in range(n_classes):
            f.write(class_names.get(i, f'class_{i}') + '\n')
    print(f'Wrote {classes_path}')

    # ── Write train.txt / val.txt ────────────────────────────────────
    train_path = output_dir / 'train.txt'
    val_path   = output_dir / 'val.txt'
    n_train = n_val = 0
    with train_path.open('w', encoding='utf-8') as ftr, \
         val_path.open('w',   encoding='utf-8') as fvl:
        for p in paired:
            target = fvl if _is_val(p.stem, args.val_fraction) else ftr
            target.write(str(p.resolve()) + '\n')
            if target is fvl:
                n_val += 1
            else:
                n_train += 1
    print(f'Split: train={n_train}  val={n_val}  '
          f'(val_fraction={args.val_fraction:.2%}, stem-hash deterministic).')

    # ── Write data.yaml ──────────────────────────────────────────────
    data_yaml = output_dir / 'data.yaml'
    with data_yaml.open('w', encoding='utf-8') as f:
        f.write('# Auto-generated by training/build_data_yaml.py\n')
        f.write(f'# Source dataset: {dataset_dir}\n')
        f.write('path: ' + str(dataset_dir).replace('\\', '/') + '\n')
        f.write('train: ' + str(train_path).replace('\\', '/') + '\n')
        f.write('val: '   + str(val_path  ).replace('\\', '/') + '\n')
        f.write('names:\n')
        for i in range(n_classes):
            f.write(f'  {i}: {class_names.get(i, f"class_{i}")}\n')
    print(f'Wrote {data_yaml}')

    print()
    print('Next step:')
    print(f'  python training\\yolo_train.py --data {data_yaml}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
