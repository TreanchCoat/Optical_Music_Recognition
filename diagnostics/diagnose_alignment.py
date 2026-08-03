"""
diagnose_alignment.py — A/B compare detections.json with vs without note alignment.

Workflow
--------
Run the pipeline twice on the same input — once with alignment off, once
with alignment on — then diff the resulting detection centres::

    python scripts/pipeline.py page1.png out_off/ --no-align
    python scripts/pipeline.py page1.png out_on/

    python diagnostics/diagnose_alignment.py \\
        out_off/detections.json out_on/detections.json \\
        --image out_on/rectified.png \\
        --overlay out_on/alignment_diff.png

What it prints
--------------
For every YOLO class that appears in both files, a one-line summary of
how far the alignment moved that class's centre:

    class             n_off  n_on  matched  mean_dx  mean_dy  med_shift  max_shift
    noteheadBlack     124    124   124        0.3     -0.8       1.1        4.4
    restQuarter         8      8     8        0.0      0.2       0.5        1.3
    ...

A "matched" detection is one whose centre in `on` is within
``--match-radius`` pixels (default 12) of the closest same-class centre
in `off`.  Unmatched-on-either-side counts are reported as a sanity
check (alignment should not invent or drop detections — only move them).

What it draws
-------------
If `--overlay` is given (and `--image` provides a backdrop), a PNG is
written with one arrow per matched detection:

    blue dot      = original YOLO centre (off)
    red dot       = aligned centre (on)
    green arrow   = displacement (off → on)

Magnitudes are clamped for visibility — see ``ARROW_SCALE``.

Why it matters
--------------
Alignment is a polish step; the right outcome is small shifts (a few px)
that land symbol centres on dark pixels in the cleaned image.  Large
shifts usually mean a template mismatch — either the wrong class
template, or a template that's too small/large for the YOLO box.  The
``max_shift`` column is the one to watch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Arrow length multiplier for the overlay — small shifts (1-2 px) are
# otherwise invisible against a busy score image.
ARROW_SCALE = 3.0

# Above this displacement (in px) a shift is "large" — gets a thicker
# arrow + counts toward the LARGE_SHIFT bucket in the summary.
LARGE_SHIFT_THRESHOLD = 6.0


# ─────────────────────────────────────────────────────────────────────────────
# JSON loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_detections(path: Path) -> List[dict]:
    """
    Flatten a detections.json into a single list of detection dicts.

    The on-disk schema is grouped by staff::

        [ { "part_id": ..., "detections": [ {...}, ... ] }, ... ]

    Per ``src/symbol_detector.py``'s output schema, every coordinate in
    a detection (cx, cy, x1, y1, x2, y2) is PAGE-RELATIVE — they already
    reference the full rectified image, no crop offsets needed.
    """
    with path.open('r', encoding='utf-8') as f:
        staves = json.load(f)
    flat: List[dict] = []
    for staff in staves:
        for d in staff.get('detections', []):
            flat.append({
                'id': d.get('id'),
                'class': d.get('class', '<unknown>'),
                'conf': d.get('conf', 0.0),
                'full_cx': float(d.get('cx', 0)),
                'full_cy': float(d.get('cy', 0)),
                'staff_part': staff.get('part_id'),
                'staff_idx': staff.get('staff_in_part'),
            })
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Matching: nearest same-class detection within a radius
# ─────────────────────────────────────────────────────────────────────────────

def _match_by_id_then_nearest(
    off: List[dict],
    on: List[dict],
    radius: float,
) -> List[Tuple[dict, dict]]:
    """
    Pair each ``off`` detection with the closest same-class ``on``
    detection.

    Strategy:
      1.  If both detections share a stable ``id`` (det_NNNN), pair them
          directly.  This is the common path: ``id_map`` is computed in
          ``pipeline.py`` and emitted into both files identically, so
          alignment-only differences keep the same ids.
      2.  Otherwise, fall back to nearest-neighbour within ``radius``.

    Returns a list of (off_det, on_det) tuples for matched pairs.
    """
    # First pass: id-based matching.
    on_by_id = {d['id']: d for d in on if d.get('id')}
    paired: List[Tuple[dict, dict]] = []
    matched_on_ids = set()
    remaining_off: List[dict] = []

    for od in off:
        oid = od.get('id')
        if oid and oid in on_by_id:
            paired.append((od, on_by_id[oid]))
            matched_on_ids.add(oid)
        else:
            remaining_off.append(od)

    # Second pass: nearest-neighbour for unmatched off-side detections.
    remaining_on = [d for d in on if d.get('id') not in matched_on_ids]
    used_on = set()
    for od in remaining_off:
        best_idx = -1
        best_dist = radius
        for i, nd in enumerate(remaining_on):
            if i in used_on:
                continue
            if nd['class'] != od['class']:
                continue
            dx = nd['full_cx'] - od['full_cx']
            dy = nd['full_cy'] - od['full_cy']
            d = math.hypot(dx, dy)
            if d <= best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0:
            paired.append((od, remaining_on[best_idx]))
            used_on.add(best_idx)

    return paired


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def _summarise(off: List[dict],
               on: List[dict],
               pairs: List[Tuple[dict, dict]]) -> Dict[str, dict]:
    """
    Compute per-class statistics::

        {
          'noteheadBlack': {
              'n_off': 124, 'n_on': 124, 'matched': 124,
              'mean_dx': ..., 'mean_dy': ...,
              'med_shift': ..., 'max_shift': ..., 'n_large': 3,
          },
          ...
        }
    """
    by_class_off: Dict[str, int] = defaultdict(int)
    by_class_on: Dict[str, int] = defaultdict(int)
    for d in off:
        by_class_off[d['class']] += 1
    for d in on:
        by_class_on[d['class']] += 1

    shifts: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for od, nd in pairs:
        if od['class'] != nd['class']:
            continue  # shouldn't happen with our matcher but be defensive
        dx = nd['full_cx'] - od['full_cx']
        dy = nd['full_cy'] - od['full_cy']
        shifts[od['class']].append((dx, dy))

    summary: Dict[str, dict] = {}
    for cls in sorted(set(by_class_off) | set(by_class_on)):
        sh = shifts.get(cls, [])
        if sh:
            dxs = np.array([d[0] for d in sh])
            dys = np.array([d[1] for d in sh])
            mags = np.hypot(dxs, dys)
            summary[cls] = {
                'n_off':    by_class_off.get(cls, 0),
                'n_on':     by_class_on.get(cls, 0),
                'matched':  len(sh),
                'mean_dx':  float(dxs.mean()),
                'mean_dy':  float(dys.mean()),
                'med_shift': float(np.median(mags)),
                'max_shift': float(mags.max()),
                'n_large':   int((mags >= LARGE_SHIFT_THRESHOLD).sum()),
            }
        else:
            summary[cls] = {
                'n_off':    by_class_off.get(cls, 0),
                'n_on':     by_class_on.get(cls, 0),
                'matched':  0,
                'mean_dx':  0.0, 'mean_dy': 0.0,
                'med_shift': 0.0, 'max_shift': 0.0, 'n_large': 0,
            }
    return summary


def _print_table(summary: Dict[str, dict]) -> None:
    """Pretty-print the summary as a fixed-width table."""
    header = (
        f'{"class":<22} {"n_off":>6} {"n_on":>6} {"matched":>8} '
        f'{"mean_dx":>9} {"mean_dy":>9} {"med_shift":>10} '
        f'{"max_shift":>10} {"n_large":>8}'
    )
    print(header)
    print('-' * len(header))
    for cls, s in sorted(summary.items()):
        print(
            f'{cls:<22} '
            f'{s["n_off"]:>6} {s["n_on"]:>6} {s["matched"]:>8} '
            f'{s["mean_dx"]:>9.2f} {s["mean_dy"]:>9.2f} '
            f'{s["med_shift"]:>10.2f} {s["max_shift"]:>10.2f} '
            f'{s["n_large"]:>8}'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _draw_overlay(image_path: Path,
                  pairs: List[Tuple[dict, dict]],
                  out_path: Path,
                  classes: Optional[set] = None) -> None:
    """
    Write a PNG with one arrow per matched detection (off → on).

    Class filter (``classes``) is useful for spotting whether one
    specific class (e.g. ``noteheadHalf``) is being mishandled.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f'Cannot read backdrop image: {image_path}')

    n_drawn = 0
    n_large = 0
    for od, nd in pairs:
        if classes and od['class'] not in classes:
            continue
        ox, oy = int(round(od['full_cx'])), int(round(od['full_cy']))
        nx, ny = int(round(nd['full_cx'])), int(round(nd['full_cy']))

        # Amplify so 1-2 px shifts are visible.
        dx = (nx - ox) * ARROW_SCALE
        dy = (ny - oy) * ARROW_SCALE
        arrow_end = (int(round(ox + dx)), int(round(oy + dy)))

        mag = math.hypot(nx - ox, ny - oy)
        thickness = 2 if mag >= LARGE_SHIFT_THRESHOLD else 1
        if mag >= LARGE_SHIFT_THRESHOLD:
            n_large += 1

        # Green arrow: off → on (scaled).  Use BGR.
        cv2.arrowedLine(img, (ox, oy), arrow_end,
                        (0, 200, 0), thickness, tipLength=0.3)
        # Blue dot: off centre.
        cv2.circle(img, (ox, oy), 2, (255, 100, 0), -1)
        # Red dot: on centre.
        cv2.circle(img, (nx, ny), 2, (0, 0, 255), -1)
        n_drawn += 1

    # Legend top-left.
    legend = [
        ('off centre',           (255, 100, 0)),
        ('on centre',            (0, 0, 255)),
        (f'off->on (x{ARROW_SCALE:g})', (0, 200, 0)),
    ]
    y = 24
    for text, color in legend:
        cv2.circle(img, (16, y - 4), 4, color, -1)
        cv2.putText(img, text, (28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22

    cv2.imwrite(str(out_path), img)
    print(f'Overlay -> {out_path}  ({n_drawn} arrows, {n_large} large)')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog='diagnose_alignment',
        description='Compare detections.json with vs without note alignment.',
    )
    ap.add_argument('off_json',
                    help='detections.json produced with --no-align')
    ap.add_argument('on_json',
                    help='detections.json produced with alignment enabled')
    ap.add_argument('--image', default=None,
                    help='Backdrop PNG for the overlay (usually '
                         'rectified.png from the "on" run).')
    ap.add_argument('--overlay', default=None,
                    help='Where to save the overlay PNG.  Skipped when '
                         'omitted.')
    ap.add_argument('--match-radius', type=float, default=12.0,
                    help='Max distance (px) to count two detections as '
                         'the same when ids differ.  Default 12.')
    ap.add_argument('--classes', default=None,
                    help='Comma-separated class names to filter the '
                         'overlay (the summary always shows all classes).')
    args = ap.parse_args(argv)

    off_path = Path(args.off_json).resolve()
    on_path = Path(args.on_json).resolve()
    if not off_path.is_file():
        print(f'ERROR: missing {off_path}', file=sys.stderr)
        return 2
    if not on_path.is_file():
        print(f'ERROR: missing {on_path}', file=sys.stderr)
        return 2

    off = _load_detections(off_path)
    on = _load_detections(on_path)
    print(f'off: {len(off):>5} detections from {off_path.name}')
    print(f'on:  {len(on):>5} detections from {on_path.name}')
    print()

    pairs = _match_by_id_then_nearest(off, on, radius=args.match_radius)
    print(f'Matched {len(pairs)} pairs '
          f'(match-radius={args.match_radius:g} px)')
    print()

    summary = _summarise(off, on, pairs)
    _print_table(summary)

    if args.overlay:
        if not args.image:
            print('  --overlay requires --image; skipping overlay.',
                  file=sys.stderr)
        else:
            cls_filter = (set(c.strip() for c in args.classes.split(','))
                          if args.classes else None)
            _draw_overlay(Path(args.image).resolve(), pairs,
                          Path(args.overlay).resolve(),
                          classes=cls_filter)

    return 0


if __name__ == '__main__':
    sys.exit(main())
