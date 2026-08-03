# Note Templates

RGBA template PNGs consumed by `src/note_alignment.py` to refine notehead and
rest centres after YOLO detection.

## Naming

One file per YOLO class, exact class-name match:

```
noteheadBlack.png
noteheadHalf.png
noteheadWhole.png
noteheadDoubleWhole.png
restWhole.png
restHalf.png
restQuarter.png
rest8th.png
rest16th.png
rest32nd.png
rest64th.png
```

Missing files are skipped silently — alignment only runs for classes whose
template exists here.

## Format

- RGBA PNG, transparent background, black ink.
- Roughly 60-100 px tall — the alignment search rescales to match the YOLO
  box size anyway, so the absolute size is not critical.
- No staff lines, no stems, no flags. Templates are a single isolated glyph.

## Generating templates

Two options:

### 1. Composite from the labelled corpus (recommended)

The repo ships `scripts/extract_note_templates.py` which picks the cleanest
examples from a YOLO-format dataset and composites them into a median
template per class:

```
python scripts/extract_note_templates.py S:\training_images_crops
```

This will write one PNG per class listed in
`note_alignment.ALIGN_CONFIG` directly into this folder.

### 2. Hand-crafted

You can also drop in templates extracted by other means (e.g. rendered from
a music font, or cleaned up from a single hand-picked crop in any image
editor). The only requirement is the RGBA format above.

## Disabling a class

Delete or rename the corresponding `.png`. The pipeline will fall back to
the connected-component refinement in `bbox_refiner.py` for that class.
