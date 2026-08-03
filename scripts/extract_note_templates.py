"""
extract_note_templates.py — Build note_alignment templates from a labeled corpus.

What it does
------------
Scans a folder of paired (image, YOLO-format-label) files (e.g.
``S:\\training_images_crops``), finds the cleanest example for each class
listed in ``--classes``, and writes one RGBA template PNG per class into
``<DATA_DIR>/note_templates/``.

"Cleanest" heuristic:
  - Crop the labelled box, pad slightly.
  - Otsu-threshold to a binary mask.
  - Score = mask-fill ratio (foreground area / total area) inside the box,
    favouring values in [0.10, 0.35] — too sparse means the symbol got
    cropped off, too dense means the box swallowed a stem or accidental.
  - Keep the K best per class; export the MEDIAN-thresholded composite as
    the template, so a single bad crop can't poison the result.

Output
------
For each requested class::

    <DATA_DIR>/note_templates/<class_name>.png    (RGBA, transparent bg)

These are consumed by ``src/note_alignment.py``.

Usage
-----
    python scripts/extract_note_templates.py S:\\training_images_crops
    python scripts/extract_note_templates.py S:\\training_images_crops \\
        --classes noteheadBlack,noteheadHalf,noteheadWhole \\
        --per-class 50 \\
        --names-from S:\\omr\\models\\deepscores_crops_v1.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Bootstrap so env_loader / src/ are reachable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
import env_loader  # noqa: E402

import cv2
import numpy as np


# Default classes to extract — keep in sync with note_alignment.ALIGN_CONFIG.
DEFAULT_CLASSES = [
    'noteheadBlack', 'noteheadHalf', 'noteheadWhole', 'noteheadDoubleWhole',
    'restWhole', 'restHalf', 'restQuarter',
    'rest8th', 'rest16th', 'rest32nd', 'rest64th',
]


def _read_names_from_pt(pt_path: Path) -> Dict[int, str]:
    """Return ``{class_id: name}`` from a YOLO .pt via Ultralytics."""
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    names = model.names
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return names


def _read_names_from_file(p: Path) -> Dict[int, str]:
    with p.open('r', encoding='utf-8') as f:
        return {i: ln.strip() for i, ln in enumerate(f) if ln.strip()}


def _iter_label_lines(label_path: Path):
    """Yield ``(class_id, cx, cy, w, h)`` floats, normalized [0, 1]."""
    with label_path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
            yield cls_id, cx, cy, w, h


def _crop_box(img: np.ndarray, cx: float, cy: float, w: float, h: float,
              pad_frac: float = 0.10) -> Optional[np.ndarray]:
    """Crop a YOLO-normalized box from ``img`` with fractional padding."""
    H, W = img.shape[:2]
    bw = w * W
    bh = h * H
    pad = pad_frac * max(bw, bh)
    x1 = int(round(cx * W - bw / 2 - pad))
    y1 = int(round(cy * H - bh / 2 - pad))
    x2 = int(round(cx * W + bw / 2 + pad))
    y2 = int(round(cy * H + bh / 2 + pad))
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    return img[y1:y2, x1:x2]


def _binarize(crop: np.ndarray) -> np.ndarray:
    """Otsu → uint8 mask (foreground = 255)."""
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    _t, mask = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def _score_crop(mask: np.ndarray) -> float:
    """
    Higher = better template candidate.

    Penalises masks that are too sparse (cropped off) or too dense
    (probably caught a neighbouring stem / accidental).  Sweet spot is
    around 0.18 fill.
    """
    if mask.size == 0:
        return -1.0
    fill = float(mask.mean()) / 255.0
    if fill < 0.05 or fill > 0.6:
        return -1.0
    # Triangular preference centred on 0.18.
    return 1.0 - abs(fill - 0.18) / 0.18


def _compose_template(crops: List[np.ndarray],
                      target_size: int = 80) -> Optional[np.ndarray]:
    """
    Resample each crop to ``target_size`` height and take the MEDIAN
    binary mask across crops.  Returns RGBA (HxWx4) with the median glyph
    drawn black on a transparent background.

    Two-pass: first compute each crop's normalised width, take the
    median, then resize all crops to (median_width, target_size).  This
    avoids the previous failure mode where the *first* crop's aspect
    ratio became the canvas for every other crop — a single wide outlier
    (e.g. a notehead whose YOLO box included a stem) would then stretch
    every template too wide.  Median is robust to outliers in both
    directions.

    Median (rather than mean) consensus on the masks themselves means a
    single mis-binarised crop can't add a ghost pixel; a pixel only
    survives if at least half the inputs agree.
    """
    # ── Pass 1: per-crop normalised width to figure out the canonical
    # aspect ratio for this class.
    new_widths: List[int] = []
    for crop in crops:
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            continue
        new_widths.append(max(4, int(round(w * (target_size / h)))))

    if not new_widths:
        return None
    canonical_w = int(np.median(new_widths))

    # ── Pass 2: resize each crop to (canonical_w, target_size).  When a
    # crop's natural width differs from canonical_w, center-crop (wider)
    # or pad-with-white (narrower).  Pad colour matches the binariser's
    # background expectation so the alpha mask comes out clean.
    norm_masks: List[np.ndarray] = []
    for crop in crops:
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            continue
        new_h = target_size
        new_w = max(4, int(round(w * (target_size / h))))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if new_w != canonical_w:
            if new_w > canonical_w:
                excess = new_w - canonical_w
                left = excess // 2
                resized = resized[:, left:left + canonical_w]
            else:
                pad = canonical_w - new_w
                left = pad // 2
                right = pad - left
                if resized.ndim == 3:
                    resized = cv2.copyMakeBorder(
                        resized, 0, 0, left, right, cv2.BORDER_CONSTANT,
                        value=(255, 255, 255))
                else:
                    resized = cv2.copyMakeBorder(
                        resized, 0, 0, left, right, cv2.BORDER_CONSTANT,
                        value=255)
        gray = (cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                if resized.ndim == 3 else resized)
        mask = _binarize(gray)
        norm_masks.append(mask)

    if not norm_masks:
        return None

    stack = np.stack(norm_masks, axis=0).astype(np.uint8)
    consensus = (stack > 0).sum(axis=0) >= (len(norm_masks) / 2.0)
    consensus = consensus.astype(np.uint8) * 255

    h, w = consensus.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 0           # black ink
    rgba[..., 3] = consensus    # alpha = mask
    return rgba


def extract(dataset_dir: Path,
            output_dir: Path,
            class_names: Dict[int, str],
            wanted: List[str],
            per_class: int,
            target_size: int,
            limit_images: Optional[int] = None) -> Dict[str, int]:
    """Return ``{class_name: n_examples_used}`` for the templates written."""
    # Reverse-map class name → id so we can recognise which lines we care about.
    name_to_id = {n: i for i, n in class_names.items()}
    wanted_ids = {name_to_id[n]: n for n in wanted if n in name_to_id}
    missing = [n for n in wanted if n not in name_to_id]
    if missing:
        print(f'  warning: these requested classes are NOT in the model — '
              f'skipped: {", ".join(missing)}')
    if not wanted_ids:
        print('No requested classes match the model — nothing to do.')
        return {}

    # Walk the dataset.  Collect (score, crop) per class.
    print(f'Scanning {dataset_dir} for paired images + labels …')
    best: Dict[str, List[Tuple[float, np.ndarray]]] = {n: [] for n in wanted_ids.values()}

    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    n_imgs = 0
    n_label_scans = 0
    n_skipped_no_match = 0
    # Early-exit when every wanted class has enough candidates already.
    # Without this, on a huge corpus we'd uselessly walk hundreds of
    # thousands of files after each bucket is saturated.
    saturated_at = per_class * 4

    def _unsaturated_ids():
        """class_ids still hungry for more candidates."""
        return {cid for cid, name in wanted_ids.items()
                if len(best[name]) < saturated_at}

    for img_path in dataset_dir.iterdir():
        if limit_images is not None and n_imgs >= limit_images:
            print(f'  --limit-images={limit_images} reached; stopping.')
            break
        hungry = _unsaturated_ids()
        if not hungry:
            print(f'  All classes saturated after {n_imgs} images; stopping.')
            break
        if img_path.suffix.lower() not in image_exts:
            continue
        label_path = img_path.with_suffix('.txt')
        if not label_path.is_file():
            continue
        # Label-first prescan: read the .txt (fast) and check whether
        # ANY annotation matches a still-hungry class.  Decoding the
        # image is by far the expensive step — skipping it when the
        # label has nothing we want speeds rare-class collection by
        # roughly 10x on a heterogeneous corpus.
        n_label_scans += 1
        wanted_lines = [t for t in _iter_label_lines(label_path)
                        if t[0] in hungry]
        if not wanted_lines:
            n_skipped_no_match += 1
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        n_imgs += 1
        for cls_id, cx, cy, w, h in wanted_lines:
            crop = _crop_box(img, cx, cy, w, h)
            if crop is None:
                continue
            mask = _binarize(crop)
            score = _score_crop(mask)
            if score <= 0:
                continue
            cls_name = wanted_ids[cls_id]
            bucket = best[cls_name]
            bucket.append((score, crop))
            # Keep only top `per_class * 2` to bound memory; we'll trim later.
            if len(bucket) > per_class * 4:
                bucket.sort(key=lambda t: -t[0])
                del bucket[per_class * 2:]

    print(f'Scanned {n_label_scans} label file(s); '
          f'decoded {n_imgs} image(s); '
          f'skipped {n_skipped_no_match} (no wanted class).')

    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, int] = {}
    for cls_name, bucket in best.items():
        if not bucket:
            print(f'  {cls_name}: no usable examples — skipped.')
            continue
        bucket.sort(key=lambda t: -t[0])
        chosen = [c for _s, c in bucket[:per_class]]
        rgba = _compose_template(chosen, target_size=target_size)
        if rgba is None:
            print(f'  {cls_name}: compositing failed — skipped.')
            continue
        out_path = output_dir / f'{cls_name}.png'
        # cv2 expects BGRA on write
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(out_path), bgra)
        written[cls_name] = len(chosen)
        print(f'  {cls_name}: composed from {len(chosen)} example(s) → {out_path}')

    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog='extract_note_templates',
        description='Build note_alignment templates from a labeled corpus.',
    )
    ap.add_argument('dataset_dir',
                    help='Folder containing paired .png/.txt YOLO files.')
    ap.add_argument('--output', default=None,
                    help='Where to write <class>.png templates.  Defaults to '
                         f'{env_loader.DATA_DIR / "note_templates"}.')
    ap.add_argument('--names-from', default=str(env_loader.MODEL_PATH),
                    help='YOLO .pt to pull class names from.  '
                         f'Default: {env_loader.MODEL_PATH}.')
    ap.add_argument('--names-file', default=None,
                    help='Plain-text class list (one per line); '
                         'overrides --names-from when provided.')
    ap.add_argument('--classes', default=','.join(DEFAULT_CLASSES),
                    help='Comma-separated YOLO class names to export.')
    ap.add_argument('--per-class', type=int, default=40,
                    help='Top-N best examples to composite per class.')
    ap.add_argument('--target-size', type=int, default=80,
                    help='Output template height in pixels.')
    ap.add_argument('--limit-images', type=int, default=None,
                    help='Stop after scanning N images.  Useful on huge '
                         'corpora when "good enough" templates from a '
                         'subset are fine.  Default: scan all.')
    args = ap.parse_args(argv)

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        print(f'Dataset dir not found: {dataset_dir}', file=sys.stderr)
        return 2

    output_dir = (Path(args.output).resolve() if args.output
                  else Path(env_loader.DATA_DIR) / 'note_templates')

    if args.names_file:
        names = _read_names_from_file(Path(args.names_file).resolve())
        print(f'Loaded {len(names)} class names from {args.names_file}.')
    else:
        pt_path = Path(args.names_from).resolve()
        if not pt_path.is_file():
            print(f'--names-from .pt not found: {pt_path}', file=sys.stderr)
            return 2
        names = _read_names_from_pt(pt_path)
        print(f'Loaded {len(names)} class names from {pt_path}.')

    wanted = [c.strip() for c in args.classes.split(',') if c.strip()]
    print(f'Extracting templates for: {", ".join(wanted)}')

    written = extract(dataset_dir, output_dir, names, wanted,
                      per_class=args.per_class,
                      target_size=args.target_size,
                      limit_images=args.limit_images)
    if not written:
        print('No templates written.')
        return 1

    print()
    print(f'Wrote {len(written)} template(s) to {output_dir}')
    print('Next step:')
    print('  Set apply_alignment_refinement=True in detect_page or call '
          'note_alignment.refine_page directly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
