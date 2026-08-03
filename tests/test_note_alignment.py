"""
Unit tests for src/note_alignment.py.

Synthetic-data tests — no YOLO, no real templates required.  Each test
constructs a tiny known-good scene (a black square on a white background
plus a matching RGBA template) and checks that the alignment functions
behave correctly.

Run directly::

    python tests/test_note_alignment.py

or under pytest::

    pytest tests/test_note_alignment.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Bootstrap: src/ on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

import note_alignment as na  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_white_bg(w: int, h: int) -> np.ndarray:
    """White-paper background, grayscale uint8 (255 everywhere)."""
    return np.full((h, w), 255, dtype=np.uint8)


def _draw_filled_disc(img: np.ndarray, cx: int, cy: int,
                      radius: int = 8, color: int = 0) -> None:
    """Draw a filled black-by-default disc on a grayscale image in place."""
    cv2.circle(img, (cx, cy), radius, color, thickness=-1)


def _make_disc_template(diameter: int = 16) -> np.ndarray:
    """RGBA template: filled black disc on a transparent background."""
    side = diameter + 4  # small margin so the mask isn't at the edge
    tpl = np.zeros((side, side, 4), dtype=np.uint8)
    # Draw an opaque black disc — colour tuple is (B, G, R, A).  On a
    # 4-channel image cv2.circle expects all four components; passing a
    # scalar would write only into channel 0.
    cv2.circle(tpl, (side // 2, side // 2), diameter // 2,
               color=(0, 0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    return tpl


# ─────────────────────────────────────────────────────────────────────────────
# template_alpha_to_mask
# ─────────────────────────────────────────────────────────────────────────────

def test_template_alpha_to_mask_basic():
    """An RGBA template with a solid alpha disc on transparent bg should
    produce a mask matching the alpha disc."""
    tpl = _make_disc_template(16)
    mask = na.template_alpha_to_mask(tpl)
    assert mask.dtype == np.uint8
    assert mask.shape == tpl.shape[:2]
    # Some pixels should be 1 (the disc), some 0 (the transparent corners)
    assert mask.max() == 1
    assert mask.min() == 0
    # Disc-pixel count should be in the right ballpark.  Lower bound is
    # the geometric area; upper bound includes antialiased edge fringe
    # (~35% wider than geometric for a 16-px-diameter disc rasterised
    # with LINE_AA).
    geometric = np.pi * 8 ** 2          # ~201
    assert mask.sum() >= 0.85 * geometric, (
        f'mask sum {mask.sum()} suspiciously small (< 85% of geometric '
        f'{geometric:.0f})')
    assert mask.sum() <= 1.40 * geometric, (
        f'mask sum {mask.sum()} suspiciously large (> 140% of geometric '
        f'{geometric:.0f})')


def test_template_alpha_to_mask_rejects_pale_ink():
    """A pale gray RGBA template (RGB above gray_max) shouldn't produce a mask."""
    side = 16
    tpl = np.full((side, side, 4), 250, dtype=np.uint8)  # very light gray
    tpl[..., 3] = 255                                     # fully opaque
    mask = na.template_alpha_to_mask(tpl, gray_max=245)
    assert mask.sum() == 0, (
        'pale-gray template should not register as ink')


# ─────────────────────────────────────────────────────────────────────────────
# _crop_safe
# ─────────────────────────────────────────────────────────────────────────────

def test_crop_safe_returns_offset():
    """Crop in the middle: x0/y0 should reflect the top-left of the crop."""
    img = np.zeros((100, 200), dtype=np.uint8)
    crop, x0, y0 = na._crop_safe(img, cx=100, cy=50, w=40, h=20)
    assert crop.shape == (20, 40)
    assert x0 == 80
    assert y0 == 40


def test_crop_safe_clamps_to_bounds():
    """Crop near a corner: bounds clamp without padding, x0/y0 stay valid."""
    img = np.zeros((100, 200), dtype=np.uint8)
    crop, x0, y0 = na._crop_safe(img, cx=5, cy=5, w=40, h=20)
    # x0/y0 clamp to 0; crop is the truncated region
    assert x0 == 0 and y0 == 0
    assert crop.shape[0] <= 20 and crop.shape[1] <= 40
    assert crop.shape[0] > 0 and crop.shape[1] > 0


# ─────────────────────────────────────────────────────────────────────────────
# align_note_distance_transform
# ─────────────────────────────────────────────────────────────────────────────

def test_align_dt_finds_offset_disc():
    """Place a black disc on white paper; init the search a few px off the
    true centre.  The alignment should move the centre toward the disc."""
    bg = _make_white_bg(200, 200)
    true_cx, true_cy = 100, 100
    _draw_filled_disc(bg, true_cx, true_cy, radius=8)

    # Build a matching disc mask
    mask = np.zeros((20, 20), dtype=np.uint8)
    cv2.circle(mask, (10, 10), 8, 1, thickness=-1)

    # Start the search 5 px off
    init_cx, init_cy = true_cx + 5, true_cy + 4
    refined_x, refined_y, loss = na.align_note_distance_transform(
        bg_gray=bg, note_mask=mask,
        init_x=init_cx, init_y=init_cy,
        max_shift=10, threshold=150, search_factor=2.0,
    )

    # Refined should be much closer to the true centre than the init was
    init_err = np.hypot(init_cx - true_cx, init_cy - true_cy)
    final_err = np.hypot(refined_x - true_cx, refined_y - true_cy)
    assert final_err < init_err, (
        f'alignment did not improve: init_err={init_err:.2f}, '
        f'final_err={final_err:.2f}')
    # And loss should be small when on-target
    assert loss < 1.5, f'loss too high at refined position: {loss}'


def test_align_dt_zero_mask_returns_init():
    """An empty mask has nothing to score — function should return init,
    not crash."""
    bg = _make_white_bg(50, 50)
    mask = np.zeros((10, 10), dtype=np.uint8)
    rx, ry, loss = na.align_note_distance_transform(
        bg_gray=bg, note_mask=mask, init_x=25, init_y=25,
        max_shift=5, threshold=150,
    )
    assert (rx, ry) == (25, 25)
    assert loss == float('inf')


def test_align_dt_rejects_black_box_with_inverse_term():
    """
    The core figure-ground fix: a candidate position fully inside a large
    black rectangle should NOT score well under inv_weight > 0, even
    though every ink pixel is on a dark pixel.  The surrounding "should-
    be-light" template pixels are also dark there, which the inverse
    term penalises.

    One-sided (inv_weight=0) Chamfer matching fails this test — it
    happily picks the interior of the black box because ink_loss=0.
    """
    # 200x200 white paper.  Big black rectangle in the top-left (the
    # "thick beam" lookalike) and a real disc target in the bottom-right.
    bg = _make_white_bg(200, 200)
    cv2.rectangle(bg, (10, 10), (90, 90), 0, thickness=-1)        # black box
    _draw_filled_disc(bg, 150, 150, radius=8)                      # the real disc

    # Disc mask, sized similar to the real disc.
    mask = np.zeros((20, 20), dtype=np.uint8)
    cv2.circle(mask, (10, 10), 8, 1, thickness=-1)

    # Start the search BIASED toward the black box so the algorithm has
    # to actively reject it in favour of the actual disc.
    init_x, init_y = 60, 60   # well inside the black box

    # One-sided matching: lands inside the box (ink_loss=0 there beats
    # the real disc, which the search can't even reach in one shot).
    rx0, ry0, l0 = na.align_note_distance_transform(
        bg_gray=bg, note_mask=mask, init_x=init_x, init_y=init_y,
        max_shift=15, threshold=150, inv_weight=0.0,
    )
    # The one-sided algorithm settles somewhere inside the black box
    # — the box spans x∈[10,90], y∈[10,90].  Loss should be effectively 0.
    assert l0 < 0.5, (
        f'one-sided ink loss should be near-zero inside the black box, '
        f'got {l0:.3f}')

    # Two-sided matching: the inverse term should penalise the black-box
    # interior heavily — its inv_loss is large there (the surrounding
    # template pixels are also dark, far from any light pixel).
    rx1, ry1, l1 = na.align_note_distance_transform(
        bg_gray=bg, note_mask=mask, init_x=init_x, init_y=init_y,
        max_shift=15, threshold=150, inv_weight=1.5,
    )
    assert l1 > 2.0, (
        f'two-sided loss inside black box should be large, got {l1:.3f}')
    assert l1 > l0 + 1.0, (
        f'inverse term did not penalise the black-box position '
        f'(one-sided={l0:.3f}, two-sided={l1:.3f})')


# ─────────────────────────────────────────────────────────────────────────────
# align_note_with_variants
# ─────────────────────────────────────────────────────────────────────────────

def test_align_variants_recovers_position():
    """End-to-end on synthetic data: the disc-template variant search
    should land on the disc and report a small loss."""
    bg = _make_white_bg(200, 200)
    _draw_filled_disc(bg, 100, 100, radius=8)
    tpl = _make_disc_template(16)

    result = na.align_note_with_variants(
        bg_gray=bg, base_template_rgba=tpl,
        init_x=105, init_y=104, init_w=16, init_h=16,
        rotations=(0,),
        scale_min=0.9, scale_max=1.1, scale_step=0.1,
        max_shift=10, threshold=150,
    )

    assert result['mask'] is not None
    err = np.hypot(result['x'] - 100, result['y'] - 100)
    assert err < 3.0, f'variant search did not converge: err={err:.2f}'
    assert result['loss'] < 1.5, f'unexpectedly high loss: {result["loss"]}'


# ─────────────────────────────────────────────────────────────────────────────
# refine_detection
# ─────────────────────────────────────────────────────────────────────────────

class _FakeDetection:
    """Stand-in for symbol_detector.Detection — only the attributes
    refine_detection touches."""
    __slots__ = ('class_name', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2',
                 'full_cx', 'full_cy')

    def __init__(self):
        self.class_name = 'noteheadBlack'
        # Init the centre 5 px off; refine_detection should pull it back
        self.cx = 105
        self.cy = 104
        self.x1 = 97
        self.y1 = 96
        self.x2 = 113
        self.y2 = 112
        self.full_cx = 105
        self.full_cy = 104


def test_refine_detection_mutates_centre():
    """refine_detection should update cx/cy/x1..y2 and full_cx/full_cy."""
    bg = _make_white_bg(200, 200)
    _draw_filled_disc(bg, 100, 100, radius=8)
    tpl = _make_disc_template(16)

    det = _FakeDetection()
    crop_y1, crop_x1 = 50, 30  # the "staff crop" sits at (30, 50) in the page

    ok = na.refine_detection(
        det, bg, tpl,
        crop_y1=crop_y1, crop_x1=crop_x1,
        loss_threshold=5.0,
    )

    assert ok is True, 'refine_detection returned False on a clean match'
    # Centre should have moved closer to (100, 100)
    err = np.hypot(det.cx - 100, det.cy - 100)
    assert err < 3.0, f'centre did not converge: {(det.cx, det.cy)} err={err:.2f}'
    # full_cx/full_cy should include the crop offset
    assert det.full_cx == det.cx + crop_x1
    assert det.full_cy == det.cy + crop_y1


def test_refine_detection_rejects_high_loss():
    """When loss exceeds threshold, the detection is left untouched."""
    bg = _make_white_bg(200, 200)
    # No disc drawn — bg is all white, so the template will match poorly
    tpl = _make_disc_template(16)

    det = _FakeDetection()
    orig_cx, orig_cy = det.cx, det.cy

    ok = na.refine_detection(
        det, bg, tpl,
        crop_y1=0, crop_x1=0,
        loss_threshold=0.1,   # very tight; should reject
    )

    assert ok is False, 'refine_detection should have rejected (high loss)'
    assert det.cx == orig_cx and det.cy == orig_cy, (
        'detection should not have been modified on rejection')


def test_refine_detection_unknown_class_returns_false():
    """A class with no ALIGN_CONFIG entry should be a silent no-op."""
    bg = _make_white_bg(50, 50)
    tpl = _make_disc_template(16)
    det = _FakeDetection()
    det.class_name = 'someClassNotInConfig'

    ok = na.refine_detection(
        det, bg, tpl,
        crop_y1=0, crop_x1=0,
    )
    assert ok is False


# ─────────────────────────────────────────────────────────────────────────────
# load_template
# ─────────────────────────────────────────────────────────────────────────────

def test_load_template_missing_returns_none(tmp_path=None):
    """A non-existent class should return None, not crash."""
    if tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp())
    result = na.load_template('thisClassDoesNotExist', templates_dir=tmp_path)
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Runner — lets you do `python tests/test_note_alignment.py` directly.
# ─────────────────────────────────────────────────────────────────────────────

def _run_all():
    tests = [v for k, v in globals().items()
             if k.startswith('test_') and callable(v)]
    n_pass = 0
    n_fail = 0
    failures = []
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f'  PASS  {name}')
            n_pass += 1
        except AssertionError as e:
            print(f'  FAIL  {name}: {e}')
            failures.append((name, str(e)))
            n_fail += 1
        except Exception as e:
            print(f'  ERROR {name}: {type(e).__name__}: {e}')
            failures.append((name, f'{type(e).__name__}: {e}'))
            n_fail += 1
    print()
    print(f'{n_pass} passed, {n_fail} failed')
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(_run_all())
