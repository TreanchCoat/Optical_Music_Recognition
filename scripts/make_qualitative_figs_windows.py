#!/usr/bin/env python3
r"""
make_qualitative_figs_windows.py  (run locally on Windows)
==========================================================
Regenerates the 9 qualitative-result images for the thesis results chapter:

    success1_original.png  success1_detections.png  success1_rendered.png
    success2_original.png  success2_detections.png  success2_rendered.png
    failure_original.png   failure_detections.png   failure_rendered.png

For each chosen (piece, page) it produces the SAME first staff system as:
  (a) original   - cropped first system from the source page scan
  (b) detections - same crop with every detected symbol boxed.  Boxes come
                   from the generated MusicXML's embedded `omr-coordinates`,
                   so NO YOLO re-run is needed.  (RECTIFY=False on these
                   printed scores => coords share the source-page frame.)
  (c) rendered   - the generated MusicXML rendered with Verovio, first system.

Dependencies (one-time):
    pip install opencv-python numpy verovio
    # for SVG -> PNG, EITHER:
    pip install cairosvg          # needs GTK/Cairo on Windows, OR:
    pip install svglib reportlab  # pure-python fallback (auto-used if no cairosvg)

Then:  python scripts\make_qualitative_figs_windows.py
Output goes to  OUT_DIR  (edit below).  Copy the 9 PNGs into your thesis
images\ folder.
"""
import json, re, hashlib, tempfile
from pathlib import Path
import cv2, numpy as np, verovio

# ----------------------------------------------------------------------
# CONFIG -- edit these paths to match your machine.
# ----------------------------------------------------------------------
PAGES_ROOT = Path(r"S:\training_images\PDF Files\Khalkhuri\Abesadzeebi")
EVAL_ROOT  = Path(r"S:\omr\eval_out\Abesadzeebi_v5")
OUT_DIR    = Path(r"S:\omr\qualitative_figs")
TMP        = Path(tempfile.gettempdir())

# (label, piece-folder, page-image-filename, generated-xml-filename)
CHOICES = [
    ("success1", "30 Gogo midis moxveulshi", "img_13496.png", "page_1.xml"),
    ("success2", "10 Lale",                   "img_13457.png", "page_1.xml"),
    ("failure",  "17 Hadila",                 "img_13468.png", "page_1.xml"),
]
PAD = 28
N_VOICES = 3   # staves in the first system

# ----------------------------------------------------------------------
def load_coords(xml_path):
    text = xml_path.read_text(encoding="utf-8")
    m = re.search(r'name="omr-coordinates">(\[.*?\])</miscellaneous-field>', text, re.S)
    return json.loads(m.group(1)) if m else []

def colour_for(cls):
    h = hashlib.md5(cls.encode()).digest()
    return (int(h[0]) | 60, int(h[1]) | 60, int(h[2]) | 60)

def box_page(d):
    """Reconstruct a page-frame bbox: embedded x1..x2 are crop-local (~137px
    left offset) but cx,cy is the true page-frame centre, so rebuild the box
    around cx,cy using the (frame-invariant) bbox width and height."""
    w = d["x2"] - d["x1"]; h = d["y2"] - d["y1"]
    return (d["cx"] - w/2.0, d["cy"] - h/2.0, d["cx"] + w/2.0, d["cy"] + h/2.0)

def first_system_band(dets):
    staff_keys = sorted({(d.get("part_id"), d.get("staff_in_part", 0)) for d in dets},
                        key=lambda k: min(d["cy"] for d in dets
                                          if (d.get("part_id"), d.get("staff_in_part",0)) == k))
    first = set(staff_keys[:N_VOICES])
    sub = [d for d in dets if (d.get("part_id"), d.get("staff_in_part",0)) in first] or dets
    boxes = [box_page(d) for d in sub]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))

def make_original_and_overlay(page_img, dets, out_orig, out_over):
    h, w = page_img.shape[:2]
    bx1, by1, bx2, by2 = first_system_band(dets)
    cx1, cy1 = int(max(0, bx1-PAD)), int(max(0, by1-PAD))
    cx2, cy2 = int(min(w, bx2+PAD)), int(min(h, by2+PAD))
    crop = page_img[cy1:cy2, cx1:cx2].copy()
    cv2.imwrite(str(out_orig), crop)
    over = crop.copy()
    if over.ndim == 2: over = cv2.cvtColor(over, cv2.COLOR_GRAY2BGR)
    for d in dets:
        bx1, by1, bx2, by2 = box_page(d)
        if by1>=cy1 and by2<=cy2 and bx1>=cx1 and bx2<=cx2:
            cv2.rectangle(over,(int(bx1-cx1),int(by1-cy1)),
                          (int(bx2-cx1),int(by2-cy1)),colour_for(d["class"]),2)
    cv2.imwrite(str(out_over), over)
    return (cx1, cy1, cx2, cy2)

def svg_to_png(svg_str, png_path):
    """Try cairosvg, then svglib+reportlab."""
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_str.encode("utf-8"),
                         write_to=str(png_path), background_color="white")
        return
    except Exception:
        pass
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    tmp_svg = TMP / (png_path.stem + ".svg")
    tmp_svg.write_text(svg_str, encoding="utf-8")
    drawing = svg2rlg(str(tmp_svg))
    renderPM.drawToFile(drawing, str(png_path), fmt="PNG", bg=0xFFFFFF)
    try: tmp_svg.unlink()
    except OSError: pass

def crop_first_system(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_ink = (gray < 200).sum(axis=1)
    thr = max(3, img.shape[1] * 0.003)
    inked = np.where(row_ink > thr)[0]
    if len(inked) == 0: return img
    y0 = inked[0]
    diffs = np.diff(inked); big = np.where(diffs > 1)[0]
    gaps = [(inked[i], inked[i+1], diffs[i]) for i in big]
    cut = inked[-1]
    if gaps:
        max_gap = max(g[2] for g in gaps)
        for gs, ge, gl in gaps:
            if gl >= 0.6 * max_gap and gs > y0 + 20:
                cut = gs; break
    cols = np.where((gray < 200).any(axis=0))[0]
    x0, x1 = max(0, cols[0]-20), min(img.shape[1], cols[-1]+20)
    return img[max(0,y0-20):min(img.shape[0],cut+20), x0:x1]

def render_musicxml(xml_path, out_png):
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth":2100,"scale":42,"adjustPageHeight":True,
                   "footer":"none","header":"none","spacingSystem":12,"evenNoteSpacing":True})
    if not tk.loadFile(str(xml_path)):
        raise RuntimeError(f"Verovio could not load {xml_path}")
    svg = tk.renderToSVG(1)
    tmp = TMP / (out_png.stem + ".full.png")
    svg_to_png(svg, tmp)
    img = cv2.imread(str(tmp))
    img = crop_first_system(img)
    cv2.imwrite(str(out_png), img)
    try: tmp.unlink()
    except OSError: pass

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, piece, page_img_name, xml_name in CHOICES:
        page_path = PAGES_ROOT / piece / page_img_name
        xml_path  = EVAL_ROOT / piece / xml_name
        print(f"[{label}] {piece} page={page_img_name}")
        if not page_path.exists(): print("   !! missing page:", page_path); continue
        if not xml_path.exists():  print("   !! missing xml:", xml_path); continue
        page_img = cv2.imread(str(page_path))
        dets = load_coords(xml_path)
        print("   detections:", len(dets), "| parts:", len({d.get('part_id') for d in dets}))
        make_original_and_overlay(page_img, dets,
                OUT_DIR/f"{label}_original.png", OUT_DIR/f"{label}_detections.png")
        try:
            render_musicxml(xml_path, OUT_DIR/f"{label}_rendered.png"); print("   rendered OK")
        except Exception as e:
            print("   !! render failed:", e)
    print("\nDone ->", OUT_DIR)

if __name__ == "__main__":
    main()
