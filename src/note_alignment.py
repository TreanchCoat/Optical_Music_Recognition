"""
note_alignment.py — Headless distance-transform alignment for noteheads + rests.

Why
---
YOLO bounding boxes are typically a few pixels off. For most classes,
``bbox_refiner.py``'s connected-component approach is sufficient — it works
directly on the staff-line-removed binary image. But for hollow noteheads on
busy backgrounds, and for rests that overlap accidentals/stems, a template-
based refinement is more robust: we slide a canonical glyph mask around the
YOLO box and minimize the mean distance from mask pixels to dark pixels in
the score.

This module is the pipeline-friendly extract of ``omr_note_alignment_tool_v6.py``
(the Tk GUI experiment).  All Tk / messagebox / canvas code is removed.  What
remains is the two pure-numpy functions ``align_note_distance_transform`` and
``align_note_with_variants``, plus a few wrappers tuned for OMR use.

Pipeline position
-----------------
    image
      → preprocessing.preprocess_image()    → ProcessedScore
      → symbol_detector.detect_page()       → PageDetections
      → note_alignment.refine_page(*)       ◀── this module (OPT-IN)
      → score_to_xml.build_score_xml()      → MusicXML

The refinement is OPT-IN.  It only runs when callers pass
``apply_alignment_refinement=True`` (to ``detect_page``) or call
``refine_page`` directly.  When no template is found for a detection's
class, the detection is left untouched.

Public API
----------
    refine_detection(det, cleaned_gray, template_rgba, ...) -> bool
    refine_page(page_detections, processed_score,
                templates_dir=None, classes=None, ...) -> int
    load_template(class_name, templates_dir) -> PIL.Image | None

Templates
---------
Stored at ``<DATA_DIR>/note_templates/<class_name>.png``, one RGBA file per
YOLO class.  Each template is:
  - Rendered against transparent background
  - Roughly sized for the typical scale at which YOLO sees the symbol
    (50-120 px tall is fine — the variant search re-scales anyway)

Use ``scripts/extract_note_templates.py`` to bootstrap templates from a
labeled corpus (e.g. ``S:\\training_images_crops``).
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Per-class alignment config
# ─────────────────────────────────────────────────────────────────────────────
#
# Only classes listed here are refined.  Other classes are skipped silently
# even when ``apply_alignment_refinement=True``.
#
# ``rotations``     list of degrees to try.  Noteheads use [0, 180] so a head
#                   whose stem-up template matches a stem-down detection (or
#                   vice versa) still aligns; rests only use [0].
# ``scale_*``       fractional scale range to search around the YOLO size.
#                   Kept tight (0.85 → 1.15) because YOLO already gave us a
#                   reasonable box — we're polishing, not searching from scratch.
# ``max_shift``     px; how far the center is allowed to move from the YOLO box.
# ``threshold``     dark-pixel threshold for the distance transform.
# ``keep_size``     True → refined box keeps the YOLO width/height, only cx/cy
#                   move.  This is right for noteheads (pitch reads cy only).
#                   False → refined box adopts the template's actual extent.
# ``loss_thresh``   maximum mean distance-transform loss to accept.  Detection
#                   is left untouched when the best variant scores higher.
#                   Tuned per-class against the actual loss distribution
#                   observed on the May 22 test page (see
#                   diagnostics/diagnose_alignment.py).  Headroom of ~3x over
#                   the observed max keeps us robust to genuine but harder
#                   examples while still rejecting clearly-wrong matches.

ALIGN_CONFIG: Dict[str, dict] = {
    # ── Noteheads ──
    # Observed losses on the May 22 test set (134/41/3/0 dets respectively):
    #   noteheadBlack       median 0.00, max 0.00  → tight threshold OK
    #   noteheadHalf        median 0.16, max 0.30
    #   noteheadWhole       median 0.16, max 0.17
    #   noteheadDoubleWhole no test data — use noteheadHalf-style headroom
    #
    # inv_weight matters most for HOLLOW noteheads (Half, Whole, DoubleWhole)
    # because the inverse mask covers the interior hole — without enforcing
    # "interior must be light", the algorithm can't tell a filled disc from
    # a hollow one.  Filled noteheadBlack also benefits but less critically.
    # Loss_thresh raised modestly to account for inv_loss contribution.
    'noteheadBlack':       {'rotations': [0, 180], 'scale_min': 0.85, 'scale_max': 1.15, 'scale_step': 0.05, 'max_shift': 6,  'threshold': 150, 'keep_size': True,  'inv_weight': 1.0, 'loss_thresh': 0.8},
    'noteheadHalf':        {'rotations': [0, 180], 'scale_min': 0.85, 'scale_max': 1.15, 'scale_step': 0.05, 'max_shift': 6,  'threshold': 150, 'keep_size': True,  'inv_weight': 1.5, 'loss_thresh': 1.35},
    'noteheadWhole':       {'rotations': [0],      'scale_min': 0.85, 'scale_max': 1.15, 'scale_step': 0.05, 'max_shift': 6,  'threshold': 150, 'keep_size': True,  'inv_weight': 1.5, 'loss_thresh': 1.30},
    'noteheadDoubleWhole': {'rotations': [0],      'scale_min': 0.85, 'scale_max': 1.15, 'scale_step': 0.05, 'max_shift': 6,  'threshold': 150, 'keep_size': True,  'inv_weight': 1.5, 'loss_thresh': 1.0},

    # ── Rests ──
    # Slightly larger search radius and looser scale because the YOLO box for
    # rests sometimes clips the bottom curl.  Observed losses:
    #   restQuarter   median 0.12, max 0.13
    #   rest8th       median 0.16, max 0.23
    # Other rest classes had no test data — pick conservative thresholds
    # consistent with shape complexity (whole/half rectangles tighter than
    # 16th/32nd/64th which have many fine flag strokes).
    #
    # inv_weight kept moderate for rests: they have complex shapes with no
    # natural "hole" requirement, so the inverse term mostly just guards
    # against placement inside a beam (which is a real risk for the
    # rest-flag classes that look like small disconnected blobs).
    'restWhole':           {'rotations': [0], 'scale_min': 0.85, 'scale_max': 1.20, 'scale_step': 0.05, 'max_shift': 8, 'threshold': 150, 'keep_size': False, 'inv_weight': 1.0, 'loss_thresh': 1.2},
    'restHalf':            {'rotations': [0], 'scale_min': 0.85, 'scale_max': 1.20, 'scale_step': 0.05, 'max_shift': 8, 'threshold': 150, 'keep_size': False, 'inv_weight': 1.0, 'loss_thresh': 1.2},
    'restQuarter':         {'rotations': [0], 'scale_min': 0.85, 'scale_max': 1.20, 'scale_step': 0.05, 'max_shift': 8, 'threshold': 150, 'keep_size': False, 'inv_weight': 0.8, 'loss_thresh': 0.9},
    'rest8th':             {'rotations': [0], 'scale_min': 0.85, 'scale_max': 1.20, 'scale_step': 0.05, 'max_shift': 8, 'threshold': 150, 'keep_size': False, 'inv_weight': 0.8, 'loss_thresh': 1.0},
    'rest16th':            {'rotations': [0], 'scale_min': 0.85, 'scale_max': 1.20, 'scale_step': 0.05, 'max_shift': 8, 'threshold': 150, 'keep_size': False, 'inv_weight': 0.8, 'loss_thresh': 1.2},
    'rest32nd':            {'rotations': [0], 'scale_min': 0.85, 'scale_max': 1.20, 'scale_step': 0.05, 'max_shift': 8, 'threshold': 150, 'keep_size': False, 'inv_weight': 0.8, 'loss_thresh': 1.5},
    'rest64th':            {'rotations': [0], 'scale_min': 0.85, 'scale_max': 1.20, 'scale_step': 0.05, 'max_shift': 8, 'threshold': 150, 'keep_size': False, 'inv_weight': 0.8, 'loss_thresh': 1.5},
}

# Classes that template alignment is actually APPLIED to by default.
# Center refinement only affects the end metric for pitched symbols
# (noteheads), so rests and ultra-rare classes are excluded — refining a
# rest's centre does not change its class/duration, and the rare classes
# have too few samples to calibrate reliably.  Their configs remain in
# ALIGN_CONFIG above for reference / experimentation.  Callers can still
# override the active set per call via ``refine_page(classes=...)``.
DEFAULT_ALIGN_CLASSES = {'noteheadBlack', 'noteheadHalf', 'noteheadWhole'}


def _default_templates_dir() -> Path:
    """Resolve the default templates directory via env_loader."""
    try:
        from env_loader import DATA_DIR
        return Path(DATA_DIR) / 'note_templates'
    except Exception:
        # Fall back to a project-relative guess; only used when env_loader
        # isn't on sys.path (e.g. unit-tests that import this directly).
        return Path(__file__).resolve().parent.parent / 'data' / 'note_templates'


# ─────────────────────────────────────────────────────────────────────────────
# Template loading
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _load_template_cached(class_name: str, templates_dir_str: str) -> Optional['np.ndarray']:
    """
    Cached template loader.  Returns an HxWx4 uint8 RGBA array, or None.

    Reads PNG via cv2 (BGRA → RGBA) so we have no hard PIL dependency at
    pipeline runtime.  PIL is only needed by the GUI tool / template
    extractor.
    """
    p = Path(templates_dir_str) / f'{class_name}.png'
    if not p.is_file():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # Normalize to RGBA HxWx4
    if img.ndim == 2:
        rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    else:
        rgba = img  # already 4-channel
    # cv2 returns BGRA; convert to RGBA so external callers see the natural
    # channel order.  Internally we only ever look at alpha (rgba[..., 3])
    # and grayscale, so the swap is cosmetic but keeps the API honest.
    rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
    return rgba


def load_template(class_name: str,
                  templates_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """
    Return the RGBA template for ``class_name``, or None if missing.

    The result is cached, so calling this in a tight loop is cheap.
    """
    td = templates_dir or _default_templates_dir()
    return _load_template_cached(class_name, str(Path(td)))


def template_alpha_to_mask(template_rgba: np.ndarray,
                           gray_max: int = 245) -> np.ndarray:
    """
    Convert an RGBA template to a binary uint8 mask.

    Note pixels are pixels that are both reasonably opaque (alpha > 20)
    and reasonably dark (gray < gray_max).  This matches the GUI tool's
    behaviour so templates exported from there work without conversion.
    """
    alpha = template_rgba[:, :, 3]
    rgb = template_rgba[:, :, :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return ((alpha > 20) & (gray < gray_max)).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Core alignment — direct port of the GUI tool's two pure functions
# ─────────────────────────────────────────────────────────────────────────────

def _crop_safe(img: np.ndarray, cx: float, cy: float,
               w: int, h: int) -> Tuple[np.ndarray, int, int]:
    """
    Centered crop with bounds clamping.

    Returns (crop, x0, y0) where (x0, y0) is the top-left in image coords.
    """
    H, W = img.shape[:2]
    x1 = max(0, int(round(cx - w / 2)))
    y1 = max(0, int(round(cy - h / 2)))
    x2 = min(W, int(round(cx + w / 2)))
    y2 = min(H, int(round(cy + h / 2)))
    return img[y1:y2, x1:x2], x1, y1


def align_note_distance_transform(
    bg_gray: np.ndarray,
    note_mask: np.ndarray,
    init_x: float,
    init_y: float,
    max_shift: int = 40,
    threshold: int = 150,
    search_factor: float = 2.0,
    inv_weight: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Slide ``note_mask`` over ``bg_gray`` and pick the position whose
    template best matches the local figure-ground structure.

    Two-sided Chamfer matching::

        ink_loss = mean distance from ink-mask pixels  → nearest DARK  pixel
        inv_loss = mean distance from inverse-mask pix → nearest LIGHT pixel
        total    = ink_loss + inv_weight * inv_loss

    The ``inv_weight`` term is the fix for one-sided Chamfer's blind spot:
    without it, a candidate position lying entirely inside a thick black
    region (beam, barline, ink blob) has ink_loss=0 and wins — every ink
    pixel sits on a dark pixel, the surroundings are ignored.  With the
    inverse term, the "should-be-light" pixels of the template have to
    actually be light, so a candidate inside a beam pays a heavy inv_loss
    penalty (its surroundings ARE the beam, i.e. dark) and is correctly
    rejected.

    Pass ``inv_weight=0.0`` to recover the previous one-sided behaviour.

    Returns ``(x, y, loss)`` — refined center in ``bg_gray`` coords, and
    the combined loss at that position (lower = better).
    """
    nh, nw = note_mask.shape
    crop_w = int(max(nw * search_factor, nw + 2 * max_shift))
    crop_h = int(max(nh * search_factor, nh + 2 * max_shift))

    crop, x0, y0 = _crop_safe(bg_gray, init_x, init_y, crop_w, crop_h)
    if crop.size == 0:
        return init_x, init_y, float('inf')

    # Ink-side distance transform: dist to nearest DARK pixel.
    dark = (crop < threshold).astype(np.uint8)
    distance_input = np.where(dark > 0, 0, 1).astype(np.uint8)
    dist_dark = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)

    # Inverse-side distance transform: dist to nearest LIGHT pixel.  Only
    # computed when the inverse term will actually be used — saves the
    # cost when the caller wants legacy one-sided matching.
    if inv_weight > 0.0:
        light_input = np.where(dark > 0, 1, 0).astype(np.uint8)
        dist_light = cv2.distanceTransform(light_input, cv2.DIST_L2, 3)
    else:
        dist_light = None

    note_ys, note_xs = np.where(note_mask > 0)
    if len(note_xs) == 0:
        return init_x, init_y, float('inf')

    # Inverse mask = "should-be-light" pixels.  This is everything inside
    # the template's bounding box that ISN'T ink.  Mostly the surrounding
    # corners of the template, plus (importantly) the interior hole of
    # hollow glyphs like noteheadHalf — which is exactly what makes a
    # hollow note different from a filled one.
    if inv_weight > 0.0:
        inv_ys, inv_xs = np.where(note_mask == 0)
        n_inv = len(inv_xs)
    else:
        inv_xs = inv_ys = None
        n_inv = 0

    init_local_x = init_x - x0
    init_local_y = init_y - y0

    best_loss = float('inf')
    best_local_x = init_local_x
    best_local_y = init_local_y

    # Coarse-to-fine search — saves ~16× over per-pixel evaluation.
    steps = [4, 2, 1]
    center_x = init_local_x
    center_y = init_local_y
    radius = max_shift

    for step in steps:
        local_best_loss = float('inf')
        local_best_x = center_x
        local_best_y = center_y

        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                cx = int(round(center_x + dx))
                cy = int(round(center_y + dy))

                # Ink pixel positions in the bg crop.
                x_positions = cx - nw // 2 + note_xs
                y_positions = cy - nh // 2 + note_ys

                valid = (
                    (x_positions >= 0) & (x_positions < dist_dark.shape[1]) &
                    (y_positions >= 0) & (y_positions < dist_dark.shape[0])
                )
                if valid.sum() < 0.5 * len(note_xs):
                    continue

                ink_loss = float(np.mean(dist_dark[y_positions[valid],
                                                  x_positions[valid]]))

                # Inverse / surround term — penalises positions where the
                # template's "should-be-light" pixels actually sit on dark
                # bg (the beam / barline / smudge scenario).
                if inv_weight > 0.0 and n_inv > 0:
                    ix = cx - nw // 2 + inv_xs
                    iy = cy - nh // 2 + inv_ys
                    iv = (
                        (ix >= 0) & (ix < dist_light.shape[1]) &
                        (iy >= 0) & (iy < dist_light.shape[0])
                    )
                    if iv.sum() < 0.5 * n_inv:
                        continue
                    inv_loss = float(np.mean(dist_light[iy[iv], ix[iv]]))
                else:
                    inv_loss = 0.0

                # Mild jump penalty — discourages latching onto a nearby
                # staff line when the YOLO box was already close.
                jump_penalty = 0.002 * math.sqrt(
                    (cx - init_local_x) ** 2 + (cy - init_local_y) ** 2)
                loss = ink_loss + inv_weight * inv_loss + jump_penalty

                if loss < local_best_loss:
                    local_best_loss = loss
                    local_best_x = cx
                    local_best_y = cy

        center_x = local_best_x
        center_y = local_best_y
        radius = max(2, step * 3)

        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best_local_x = local_best_x
            best_local_y = local_best_y

    return x0 + best_local_x, y0 + best_local_y, best_loss


def align_note_with_variants(
    bg_gray: np.ndarray,
    base_template_rgba: np.ndarray,
    init_x: float,
    init_y: float,
    init_w: int,
    init_h: int,
    *,
    rotations: Iterable[int] = (0, 180),
    scale_min: float = 0.85,
    scale_max: float = 1.15,
    scale_step: float = 0.05,
    max_shift: int = 40,
    threshold: int = 150,
    search_factor: float = 2.0,
    inv_weight: float = 1.0,
) -> Dict:
    """
    Search over (rotation × scale × position) for the best alignment.

    The template is first resampled to ``init_w × init_h`` (the YOLO box
    size), then the variant search multiplies that base size by each scale
    in ``[scale_min, scale_max]`` (inclusive, ``scale_step`` increment).

    Returns a dict::

        {
          "x": refined_cx, "y": refined_cy, "loss": best_loss,
          "scale": s, "rotation": deg,
          "mask": <binary mask used for the winning variant>,
          "mask_w": int, "mask_h": int,
        }
    """
    # Resize template so 1.0× ≈ YOLO box size.  Done with cv2 so we don't
    # require PIL at runtime.
    base = cv2.resize(base_template_rgba, (max(1, int(init_w)), max(1, int(init_h))),
                      interpolation=cv2.INTER_AREA)

    scales: List[float] = []
    s = float(scale_min)
    while s <= scale_max + 1e-9:
        scales.append(round(s, 4))
        s += scale_step
    if not scales:
        scales = [1.0]

    best = {
        'loss': float('inf'),
        'x': float(init_x),
        'y': float(init_y),
        'scale': 1.0,
        'rotation': 0,
        'mask': None,
        'mask_w': init_w,
        'mask_h': init_h,
    }

    for rotation in rotations:
        if rotation == 0:
            rotated = base
        else:
            # cv2.rotate handles 90/180/270 cleanly without resizing.
            if rotation == 180:
                rotated = cv2.rotate(base, cv2.ROTATE_180)
            elif rotation == 90:
                rotated = cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 270:
                rotated = cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                # Arbitrary angle — fall back to warpAffine.  Not used in the
                # default config, but keeps the API general.
                bh, bw = base.shape[:2]
                M = cv2.getRotationMatrix2D((bw / 2, bh / 2), rotation, 1.0)
                rotated = cv2.warpAffine(base, M, (bw, bh),
                                         flags=cv2.INTER_LINEAR,
                                         borderValue=(0, 0, 0, 0))

        bh, bw = rotated.shape[:2]
        for scale in scales:
            nw = max(1, int(round(bw * scale)))
            nh = max(1, int(round(bh * scale)))
            if scale == 1.0 and (nw, nh) == (bw, bh):
                candidate = rotated
            else:
                candidate = cv2.resize(rotated, (nw, nh),
                                       interpolation=cv2.INTER_AREA)
            mask = template_alpha_to_mask(candidate)

            x, y, loss = align_note_distance_transform(
                bg_gray=bg_gray,
                note_mask=mask,
                init_x=init_x,
                init_y=init_y,
                max_shift=max_shift,
                threshold=threshold,
                search_factor=search_factor,
                inv_weight=inv_weight,
            )
            # Mild "stay near 1.0×" penalty — prevents picking an
            # unrealistically large variant just because it overlaps more
            # dark pixels.
            loss += 0.0001 * abs(scale - 1.0)

            if loss < best['loss']:
                best.update({
                    'loss': loss, 'x': x, 'y': y,
                    'scale': scale, 'rotation': rotation,
                    'mask': mask, 'mask_w': nw, 'mask_h': nh,
                })

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-facing wrappers
# ─────────────────────────────────────────────────────────────────────────────

def refine_detection(detection,
                     cleaned_gray: np.ndarray,
                     template_rgba: np.ndarray,
                     crop_y1: int,
                     crop_x1: int = 0,
                     class_name: Optional[str] = None,
                     config_override: Optional[dict] = None,
                     loss_threshold: Optional[float] = None,
                     loss_log: Optional[list] = None,
                     ) -> bool:
    """
    Refine a single Detection in place using template alignment.

    Parameters
    ----------
    detection        Detection from symbol_detector.py
    cleaned_gray     The staff-line-removed binary/grayscale image
                     (``pstaff.cleaned``).  Must be 2D uint8.
    template_rgba    HxWx4 RGBA template for this class.
    crop_y1, crop_x1 Offsets from the staff crop into the full rectified
                     image — used to update ``full_cx`` / ``full_cy``.
    class_name       Override for the per-class config lookup.
    config_override  Custom alignment config (same schema as ALIGN_CONFIG).
    loss_threshold   Maximum loss to accept.  If None (default), the
                     per-class ``loss_thresh`` from ALIGN_CONFIG is used.
                     Passing an explicit value forces it for every class
                     (useful for diagnostics / sweeps).

    Returns True if the detection was modified, False otherwise.
    """
    cls = class_name or detection.class_name
    cfg = config_override or ALIGN_CONFIG.get(cls)
    if cfg is None:
        return False
    if cleaned_gray.ndim != 2:
        cleaned_gray = cv2.cvtColor(cleaned_gray, cv2.COLOR_BGR2GRAY)

    init_w = max(1, detection.x2 - detection.x1)
    init_h = max(1, detection.y2 - detection.y1)

    result = align_note_with_variants(
        bg_gray=cleaned_gray,
        base_template_rgba=template_rgba,
        init_x=detection.cx,
        init_y=detection.cy,
        init_w=init_w,
        init_h=init_h,
        rotations=cfg.get('rotations', [0]),
        scale_min=cfg.get('scale_min', 0.85),
        scale_max=cfg.get('scale_max', 1.15),
        scale_step=cfg.get('scale_step', 0.05),
        inv_weight=cfg.get('inv_weight', 1.0),
        max_shift=cfg.get('max_shift', 6),
        threshold=cfg.get('threshold', 150),
        search_factor=2.0,
    )

    # Use per-class loss_thresh by default; the caller can still override
    # for diagnostics.
    effective_threshold = (loss_threshold if loss_threshold is not None
                           else cfg.get('loss_thresh', 1.0))
    accepted = result['loss'] <= effective_threshold
    if loss_log is not None:
        loss_log.append((cls, float(result['loss']),
                         float(effective_threshold), int(accepted)))
    if not accepted:
        return False

    new_cx = int(round(result['x']))
    new_cy = int(round(result['y']))

    if cfg.get('keep_size', True):
        half_w = init_w // 2
        half_h = init_h // 2
        new_x1 = new_cx - half_w
        new_y1 = new_cy - half_h
        new_x2 = new_cx + (init_w - half_w)
        new_y2 = new_cy + (init_h - half_h)
    else:
        mw = result.get('mask_w', init_w)
        mh = result.get('mask_h', init_h)
        half_w = mw // 2
        half_h = mh // 2
        new_x1 = new_cx - half_w
        new_y1 = new_cy - half_h
        new_x2 = new_cx + (mw - half_w)
        new_y2 = new_cy + (mh - half_h)

    detection.cx = new_cx
    detection.cy = new_cy
    detection.x1 = int(new_x1)
    detection.y1 = int(new_y1)
    detection.x2 = int(new_x2)
    detection.y2 = int(new_y2)
    detection.full_cx = detection.cx + crop_x1
    detection.full_cy = detection.cy + crop_y1
    return True


def refine_page(page_detections,
                processed_score,
                templates_dir: Optional[Path] = None,
                classes: Optional[Iterable[str]] = None,
                loss_threshold: Optional[float] = None,
                reject_unmatched: bool = False,
                loss_log_path: Optional[str] = None,
                verbose: bool = True) -> int:
    """
    Refine every notehead / rest detection on the page in place using
    template alignment.

    Parameters
    ----------
    page_detections   PageDetections from symbol_detector.detect_page
    processed_score   ProcessedScore from preprocessing.preprocess_image
                      — provides the cleaned staff crops.
    templates_dir     Where to look for ``<class_name>.png`` templates.
                      Defaults to ``DATA_DIR/note_templates``.
    classes           Restrict to these YOLO classes.  Default: every class
                      present in ALIGN_CONFIG that also has a template
                      available on disk.
    loss_threshold    Override the per-class ``loss_thresh`` for every
                      detection.  None (default) means use ALIGN_CONFIG.
    reject_unmatched  If True, DROP any templated-class detection whose best
                      alignment loss exceeds the (per-class) threshold — i.e.
                      use template matching as a false-positive filter, which
                      pairs with a low YOLO confidence threshold ("detect
                      everything, then discard what doesn't match a glyph").
                      Only templated classes (noteheads / rests with a
                      template on disk) can be rejected; every other class is
                      always kept.  Default False = legacy behaviour, where a
                      failed match simply leaves the detection unchanged.
    verbose           Print a summary line.

    Returns the number of detections modified (not counting rejections).
    """
    td = Path(templates_dir) if templates_dir else _default_templates_dir()
    if not td.is_dir():
        if verbose:
            print(f'[note_alignment] templates dir not found: {td} — skipping')
        return 0

    allowed = set(classes) if classes else set(DEFAULT_ALIGN_CLASSES)

    # Pre-warm template cache; skip classes with no template available.
    available: Dict[str, np.ndarray] = {}
    for cls in allowed:
        tpl = load_template(cls, td)
        if tpl is not None:
            available[cls] = tpl
    if not available:
        if verbose:
            print(f'[note_alignment] no templates found in {td} — skipping')
        return 0

    # Optional per-detection loss logging for threshold tuning.  Enable via
    # the loss_log_path argument or the ALIGN_LOSS_LOG environment variable.
    # Records (class, loss, threshold, accepted) are appended to a CSV, so a
    # multi-page / multi-piece run accumulates one distribution file.
    log_path = loss_log_path or os.environ.get('ALIGN_LOSS_LOG')
    loss_records: Optional[list] = [] if log_path else None

    n_modified = 0
    n_attempted = 0
    n_rejected = 0
    by_class: Dict[str, int] = {}

    for part_staves, sdet_list in zip(processed_score.parts,
                                       page_detections.parts):
        for pstaff, sdet in zip(part_staves, sdet_list):
            cleaned = pstaff.cleaned
            crop_y1 = int(pstaff.crop_y1)
            crop_x1 = int(getattr(pstaff, 'crop_x1', 0))
            kept = []
            for det in sdet.detections:
                tpl = available.get(det.class_name)
                if tpl is None:
                    # Non-templated class — never refined and never rejected.
                    kept.append(det)
                    continue
                n_attempted += 1
                ok = refine_detection(det, cleaned, tpl,
                                      crop_y1=crop_y1, crop_x1=crop_x1,
                                      loss_threshold=loss_threshold,
                                      loss_log=loss_records)
                if ok:
                    n_modified += 1
                    by_class[det.class_name] = by_class.get(det.class_name, 0) + 1
                    kept.append(det)
                else:
                    # Failed the per-class loss threshold.  Either drop it
                    # (template matching as a false-positive filter) or keep
                    # it unchanged (legacy behaviour).
                    if reject_unmatched:
                        n_rejected += 1
                    else:
                        kept.append(det)
            if reject_unmatched:
                sdet.detections = kept

    # Append collected per-detection losses to the CSV (header once).
    if log_path and loss_records:
        new_file = not os.path.exists(log_path)
        try:
            with open(log_path, 'a', encoding='utf-8') as fh:
                if new_file:
                    fh.write('class,loss,threshold,accepted\n')
                for cls_, loss_, thr_, acc_ in loss_records:
                    fh.write(f'{cls_},{loss_:.4f},{thr_:.4f},{acc_}\n')
            if verbose:
                print(f'[note_alignment] logged {len(loss_records)} losses '
                      f'-> {log_path}')
        except OSError as e:
            if verbose:
                print(f'[note_alignment] could not write loss log: {e}')

    if verbose:
        if n_attempted == 0:
            print('[note_alignment] no detections matched templated classes')
        else:
            pct = 100.0 * n_modified / max(1, n_attempted)
            summary = ', '.join(f'{k}={v}' for k, v in sorted(by_class.items()))
            reject_note = (f'  rejected {n_rejected}' if reject_unmatched else '')
            print(f'[note_alignment] refined {n_modified}/{n_attempted} '
                  f'detections ({pct:.1f}%)  [{summary}]{reject_note}')
    return n_modified
