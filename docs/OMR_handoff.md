# OMR Pipeline — Handoff (v2)

> Successor to the original `OMR handoff.md`.  Captures everything done
> in the recent session: pipeline architecture, design decisions,
> file layout, training, and open work.  Pasting this into a fresh
> chat is enough for an assistant to pick up cold.

---

## 0. Quick orientation

End-to-end OMR pipeline at `S:\omr\` (Windows; the user is Vangust).
Input PNG score image → MusicXML.  Stack:

* **Stage 1 — rectify** (`staff_rectifier`) — straighten curved staves.
* **Stage 1b — clef-split** (`clef_splitter`) — handles rows with
  multiple staves side-by-side.
* **Stage 2 — preprocess** (`preprocessing` + `score_analyzer` +
  `staff_remover`) — detect staves, crop, binarise, remove lines.
* **Stage 3 — symbol detection** (`symbol_detector` + `bbox_refiner`)
  — YOLOv8 inference.
* **Stage 4 — assemble** (`build_musicxml_v2` + `xml_builder`) — write
  MusicXML.
* **Stage 4½ — re-bar** (`measure_recalculator`) — split into measures
  by `<time>` signature.

Originally for 3-clarinet Georgian folk music; scope expanded to general
scores.  User is on Windows with an **RTX 3050 (8 GB VRAM)**, Vast.ai
RTX 4090 / 5090 for heavy training.

---

## 1. Project layout (post-reorg)

```
S:\omr\
├── .env                  ← local config (gitignored)
├── .env.example          ← documented template
├── README.md
├── requirements.txt
│
├── src/                  ← every pipeline source module
│   ├── env_loader.py            (.env loader + path bootstrap + path constants)
│   ├── xml_builder.py           (lxml-based MusicXML constructor)
│   ├── score_analyzer.py        (staff detection + row clustering + part assignment)
│   ├── staff_remover.py         (binarize + remove_staff_lines)
│   ├── staff_unet.py            (U-Net architecture; shared by detect + remove)
│   ├── staff_detector_unet.py   (U-Net staff-line detector, tiled inference)
│   ├── staff_detector_yolo.py   (OLA fallback)
│   ├── staff_rectifier.py       (Stage 1)
│   ├── preprocessing.py         (Stage 2 orchestrator)
│   ├── symbol_detector.py       (Stage 3 YOLO inference + visualisation)
│   ├── bbox_refiner.py          (Stage 3 polish — refine notehead boxes)
│   ├── note_alignment.py        (Stage 3 polish — template-DT alignment, OPT-IN)
│   ├── clef_splitter.py         (Stage 1b — multi-clef row splitting)
│   ├── build_musicxml_v2.py     (Stage 4)
│   ├── measure_recalculator.py  (Stage 4½)
│   └── score_to_xml.py          (legacy v1; kept for reference)
│
├── scripts/              ← entry-point CLIs
│   ├── pipeline.py
│   ├── preprocess_for_training.py
│   └── extract_note_templates.py   (build note_alignment templates from corpus)
│
├── api/                  ← REST API + clients + tester UI
│   ├── real_api.py
│   ├── dummy_api.py
│   ├── api_client_example.py
│   ├── process_score.py
│   └── api_tester.html
│
├── training/             ← model-training scripts
│   ├── staff_unet_train.py     (staff-line DETECTION)
│   ├── staff_remover_train.py  (staff-line REMOVAL — tried, parked)
│   ├── dataset_prep.py
│   └── muscima_manifest.json
│
├── diagnostics/          ← ad-hoc dev scripts
├── tests/                ← _smoke_test_v2.py
├── models/               ← trained weights
│   ├── staff_unet.pth                 (~125 MB, IoU 0.85)
│   ├── ola_v2.pt                      (OLA pretrained, ~40 MB)
│   ├── deepscores_crops_v1.pt         (YOLO symbol model, ~52 MB)
│   └── staff_remover.pth              (removal U-Net, val IoU ~0.87 — UNUSED)
├── data/
│   ├── input/                ← input PNGs
│   ├── sample/               ← dummy-API fixtures
│   └── note_templates/       ← RGBA glyph templates for note_alignment
│       ├── README.md         (naming + format)
│       └── <class>.png       (one per refined YOLO class; missing = skip)
├── docs/                 ← thesis_methodology.md, UNET_SETUP.md, this handoff
└── archive/              ← legacy files + old test outputs
```

`env_loader.py` is the single source of truth for paths: it loads
`.env`, adds `src/` to `sys.path`, and exposes typed constants
(`MODEL_PATH`, `MODELS_DIR`, `INPUT_DIR`, `OUTPUT_BASE_DIR`,
`DUMMY_*_PATH`, `HOST`, `PORT`, …).  Every entry point's first line is
`import env_loader`.

`get_num_parts()` / `infer_num_parts()` decide stave-per-system count
— see §4 "Auto-inferred num_parts".

---

## 2. Pipeline stages in detail

### Stage 1 — `staff_rectifier.process_image`

* Multi-strategy detection: **U-Net → YOLO → classical multi-strategy**.
* `return_metadata=True` returns `(rectified_image, staves_meta)`
  where each meta dict carries `top_y / bot_y / left_x / right_x /
  line_spacing / line_positions` in **canvas coordinates**, so Stage 2
  can skip its own classical re-detection.
* **Auto-scaled sizing** from the source's actual line spacing:
  `target_line_spacing = median(detected line spacings)`,
  `vertical_pad = 4× line_spacing` (was 2× — bumped for fermatas),
  `gap = 1.5× line_spacing`.  Canvas height now matches input
  proportions instead of being artificially tall.
* **Single-staff rows extended to full image width**: when a row has
  exactly one detected staff, override its `left_x = 0, right_x =
  img_w - 1` so the homography doesn't clip clefs or end-barlines that
  U-Net missed.  Multi-staff rows keep their detected extents.

### Stage 1b — `clef_splitter`

Runs YOLO once on the rectified page, filtered to clef classes
(`clefG`, `clefF`, `clefCAlto`, `clefCTenor`).  For each Stage-1 staff
that contains **>1 clef**, splits that staff's metadata into N
sub-staves.

Boundary rule (after iteration):
```
sub_left  = clef.x1 − LEFT_PADDING_FRAC * img_w
sub_right = next_sub_left − 1
```

Anchored on the clef's bbox **left edge** (`x1`), not centre, so the
first sub-stave's right edge doesn't bleed into the next clef.
`LEFT_PADDING_FRAC = 0.030` (doubled from the original 0.015 after
visual feedback).  Consecutive sub-staves share boundaries — no gap
where symbols could fall through.

### Stage 2 — `preprocessing.preprocess_image`

Accepts `precomputed_staves=stage1_staves_meta`.  When given, skips
detection and routes through `assemble_score_data(...)` which does
row clustering, part assignment, cropping, and staff-line removal.

Part assignment in `assemble_score_data`:
* **Multi-staff rows** → position-in-row modulo num_parts.
* **All-single-staff rows + num_parts > 1** → "vertically-stacked
  layout" mode: top-to-bottom assignment via global-index modulo.

### Stage 3 — `symbol_detector.detect_page`

* YOLOv8 inference per cleaned staff crop.
* **Filters out `staff`/`staffLine`** YOLO classes — those add noise
  because we have a dedicated staff detector.
* `bbox_refiner` tightens notehead boxes via connected components on
  the cleaned image.
* **Optional `note_alignment.refine_page`** (`apply_alignment_refinement=True`)
  slides per-class glyph templates around each notehead / rest box to
  refine the centre via distance-transform matching.  Most useful for
  hollow noteheads and rests, where the connected-component refiner
  struggles.  Headless port of the v6 GUI tool — see
  `src/note_alignment.py`.  Templates live at `data/note_templates/`;
  the pipeline silently no-ops when no template exists for a class.
  Build templates with `python scripts/extract_note_templates.py
  S:\training_images_crops`.  A/B diagnostic at
  `diagnostics/diagnose_alignment.py` compares two runs and reports
  per-class centre-shift stats.

### Stage 4 — `build_musicxml_v2`

* No barlines / dynamics / lyrics / tie-vs-slur differentiation.
* **Chord grouping at ±10 px**: noteheads within tolerance collapse
  into a single `<note>` chain with `<chord/>`.
* **Configurable divisions** via `auto_divisions`: scans for rest /
  flag classes, picks 1 / 2 / 4 / 8 / 16.
* **Time signatures**: `detect_all_time_signatures` finds compound
  classes (`timeSig3over4`), symbolic (`timeSigCommon` → 4/4), and
  stacked digit pairs.  Multiple time-sig changes within a staff are
  interleaved as `<attributes><time>` events in the event stream.
* **Fermata / trill / turn attachment** by Euclidean distance to the
  nearest notehead.
* **Repeat barlines** detected from `repeatLeft / repeatRight`-style
  classes, emitted as `<barline><repeat …/>`.
* **Melisma framework** ready for 5 not-in-DeepScores types (shake,
  acciaccatura, nachshlang, long appoggiatura, glissando) via
  `apply_melisma_overrides`.
* **Element IDs**: every `<note>` gets a `det_NNNN` id matching the
  detections JSON; full coordinate list embedded as
  `<miscellaneous-field name="omr-coordinates">` JSON.
* `id_map` is computed once in `pipeline.py` and passed to both the
  JSON writer and the XML builder so the two files cross-reference.

### Stage 4½ — `measure_recalculator`

Re-runnable standalone script.  **Time-signature-only splitting** —
no beat counting.  Per the user's decision: measure duration accuracy
is not needed for the final product; what matters is structural
correspondence between bars and visible regions on the page.

Rules:
* Each input measure (= one detected staff) becomes one output measure.
* Any mid-stream `<attributes><time>` triggers an extra split.
* Short parts are padded to the longest part's measure count so
  MuseScore doesn't show red blank boxes.
* Forward / backward repeat barlines migrate to the new measure
  boundaries.
* CLI: `python measure_recalculator.py in.xml out.xml [--time 3/4]`
  — if `--time` is omitted, the recalculator uses the `<time>`
  already embedded in the XML.

---

## 3. Output artefacts (after a pipeline run)

```
output/
├── rectified.png                  Stage 1 output
├── stage1_staves_on_input.png     Stage 1 staves drawn on the ORIGINAL input
├── unet_mask.png                  U-Net binary staff mask (if U-Net used)
├── unet_prob_heatmap.png          U-Net probability JET heatmap
├── stage2_staves_on_rectified.png Stage 2 staves on rectified page
├── symbols_detected.png           YOLO boxes on rectified page (per-class colour)
├── clefs_detected.png             clef_splitter input clefs (red boxes)
├── binarized/                     ONE PNG per staff, BINARISED only
│   └── PN_staffMM_binary.png      …no line removal yet (diagnosis)
├── labeled_crops/                 ONE PNG per staff with YOLO boxes
│   └── PN_staffMM_labeled.png     …color legend below; no per-box text
├── detections.json                grouped by staff, with stable det_NNNN IDs
└── score.xml                      MusicXML, re-barred
```

`detections.json` shape (top-level list of staves; each detection
matches the `omr-coordinates` schema in the XML):

```json
[
  {
    "part_id": "P1", "staff_in_part": 0,
    "top_y": 363, "bot_y": 425,
    "line_positions": [363, 378, 393, 410, 425],
    "line_spacing": 15.5,
    "crop_y1": 305, "crop_x1": 137,
    "total_detections": 37,
    "detections": [
      { "id": "det_0001", "class": "clefG", "conf": 0.96,
        "part_id": "P1", "staff_in_part": 0,
        "cx": 276, "cy": 400,
        "x1": 118, "y1": 343, "x2": 160, "y2": 457 },
      …
    ]
  }
]
```

All coordinates are full-image (already offset by `crop_y1`).

---

## 4. Auto-inferred num_parts

`src/score_analyzer.py::infer_num_parts(staves_meta)` replaces the
old `get_num_parts() = 3` static return.

Algorithm:
1. Sort staves by `top_y`, compute consecutive gaps.
2. Look at the biggest "jump" in sorted gap values.  If it's
   < 30 % of the mean gap, gaps are unimodal → fall back to default.
3. Else split at the midpoint of that jump.  The cluster with
   **fewer members** is the between-system one (true whether
   between-system gap is bigger OR smaller — handles the inverted
   case the user reported).
4. `num_systems = between_count + 1` → `M = N / num_systems`.

`_DEFAULT_NUM_PARTS = 1` (was 3).  Per user preference: solo scores
are common and a wrong "3" invents two empty parts the renderer
pads with measure-rests.

Override paths in priority order:
1. `assemble_score_data(num_parts=X)` programmatically.
2. `OMR_NUM_PARTS=X` env var.
3. Geometric inference (default).
4. Static fallback (1).

---

## 5. Image preprocessing — current state

### Binarisation (`src/staff_remover.binarize`)

**Default method = `'flatten'`** (background-flattened Otsu).
Replaces the legacy hard cutoff at 128 that damaged handwritten /
photographed pages.

Pipeline:
1. Wide Gaussian blur (`ksize ≈ page_width / 14`) → background estimate.
2. `cv2.divide(gray, bg, scale=255)` → illumination-normalised image.
3. Otsu threshold on the flattened image, biased UP by
   `sensitivity` (default 35) to preserve more ink.
4. `MORPH_OPEN(2×2)` to fill 1-px pinholes inside ink strokes.

Alternative methods accessible by `binarize(img, method=...)`:
* `'otsu'` — plain Otsu (no flatten).
* `'adaptive'` — `cv2.adaptiveThreshold(GAUSSIAN_C)`.
* `'fixed'` — legacy hard cutoff at 128 (for comparison only).

### Staff-line removal — kept classical

The user trained a U-Net for removal (val IoU ~0.87 on CVC-MUSCIMA
full multi-variant), but the visualisation showed it **doesn't
actually remove staff lines well** at inference time on real scans.
**Decision: revert to the classical `remove_staff_lines` algorithm.**

The trained weights (`models/staff_remover.pth`) stay on disk in case
the experiment is revisited.  The trainer
(`training/staff_remover_train.py`) is intact and includes:
* `RemovalDataset` — CVC-MUSCIMA `image/` → `symbol/` pairs (the
  symbol path is auto-derived by swapping the directory name).
* `predict_clean(model_path, image)` inference helper with auto-
  detected input convention (real-scan vs MUSCIMA-raw) and a
  threshold parameter.

A bug worth noting if anyone goes back: `predict_clean` used to
unconditionally invert the input, which double-inverted real-scan
binarised images and made the model output ≈ 0 everywhere.  The fix
auto-detects from mean pixel intensity (mean > 127 → real-scan,
no invert).

### Proposed but unimplemented design

User suggested: train separate U-Nets for handwritten vs printed,
switch on the `RECTIFY` flag.  Recommendation written but not yet
applied: skip the printed-specific U-Net (CVC-MUSCIMA is
handwritten-only anyway; classical removal is already strong on
printed) and instead use `RECTIFY` as a method selector:

```
RECTIFY=True  → use U-Net (when re-enabled)
RECTIFY=False → classical
```

Task #38 in the task list, not yet executed.

---

## 6. UI / API

`api_tester.html` has a new **"detections.json" tab** on the
`/process` section that displays exactly the file content (no
envelope wrapping).  "Raw JSON" tab still shows the full envelope.
Stats tab now reads `d.class` (was `d.class_name`).

`real_api.py` / `dummy_api.py` both pull path defaults from
`env_loader` constants — no more `_HERE / 'sample_data'` hardcoding.

---

## 7. Training history & decisions

### Staff DETECTION U-Net (`models/staff_unet.pth`)
* Trained on Vast.ai RTX 4090/5090.
* 15 epochs to IoU 0.85; was still climbing.
* Critical fix earlier: mask polarity (`mask >= 128` not `< 128`),
  tiled inference at native resolution (the U-Net was trained at 512
  but real pages are 2 500+ wide; resizing crushes 1-2 px stafflines
  to sub-pixel).  Tiled inference recovers all staves on both printed
  and handwritten test scores.

### Staff REMOVAL U-Net (`models/staff_remover.pth`)
* 25-epoch full multi-variant run.  Stopped at epoch 17 (val IoU
  0.87, still climbing slowly).
* Per-epoch time on the 3050: ~2400 s (40 min).  Full 25-epoch full-
  variant run ~16 hours.
* Visual A/B showed the model doesn't strongly remove staff lines at
  inference — parked.  See §5 for the diagnosis (double-invert bug
  fixed; result still mediocre, indicating the model itself is the
  bottleneck).
* Decision: stick with classical.

### Training-prep script (`scripts/preprocess_for_training.py`)

Batch processor that walks a folder of source PNGs, runs Stages 1
and 2, and writes one cleaned PNG per detected staff to
`OUTPUT_DIR/images/`, plus a `manifest.csv`.  Used to generate
training data for future YOLO fine-tuning on Georgian-specific
notation (the 15,100-page corpus the user has).  Resumable, multi-
process, with per-page timing.

---

## 8. Thesis

`docs/thesis_methodology.md` is a scaffold for the Methodology
chapter.  Each section starts with text and has `**TO EXPAND**`
checklists for what the user needs to flesh out.  Sections:

1. Overview + pipeline diagram.
2. Datasets (CVC-MUSCIMA, DeepScoresV2, the 15 100-page Georgian
   corpus).
3. Stage-by-stage methodology (rectifier, preprocessing, symbol
   detection, score reconstruction, measure recalc).
4. Training methodology + the polarity-bug debugging story.
5. Evaluation methodology (symbol-level mAP, per-element XML diff,
   listenability).
6. Limitations and future work (multi-page assembly, voice
   separation, beam-stack counting, tuplets, bracket recognition,
   pickup measures, cadenza, printed-score binarisation damage, …).
7. Reproducibility checklist.

User is on Masters thesis; this is the "Methodology" chapter
scaffold, not a finished doc.

---

## 9. Open / not-yet-done items

These are real next steps that were discussed but not implemented:

* **Finish YOLO fine-tune on Georgian corpus.**  Labelling is done
  (`S:\training_images_crops`, 362 666 paired image/label files —
  YOLO-format pseudo-labels generated by a DeepScoresV2-pretrained
  model and then class-merged; **these are training labels, not eval
  ground truth** — see "Evaluation ground truth" below for what
  evaluation actually compares against).  Data files produced by
  `training/build_data_yaml.py`.  After cloud start + 5 nightly epochs
  on the 3050, mAP50 climbed from 0.365 to 0.388, mAP50:95 from 0.216
  to 0.241 (small but steady).  Current weights at
  `runs/detect/nightly-5/weights/last.pt`.  Plan is to keep doing
  one-epoch-per-night runs via `scripts/train_nightly.bat` and/or
  resume on a cloud GPU when faster turnaround matters.  Heads-up:
  Ultralytics `optimizer: auto` silently overrides `--lr0`, so all
  nightly runs to date have actually trained at `lr=0.01` rather than
  the intended `lr=0.001`.  Adding `--optimizer SGD` to the .bat (now
  done) forces `lr0` to take effect on subsequent runs.  Don't forget
  to re-run `build_data_yaml.py` if you move between Windows and
  Linux machines — paths in train.txt are absolute.

* **Class taxonomy is still being narrowed.**  Currently 124 classes
  (down from DeepScoresV2's 135 via merging/exclusion).  Some
  redundant classes remain and may be further consolidated; if you
  drop any, regenerate `data.yaml` / `classes.txt` and re-train.
* **Train note_alignment templates from full corpus.**  9 of 11
  templates already generated under `data/note_templates/`
  (noteheadBlack/Half/Whole/DoubleWhole, restWhole/Half/Quarter/8th/16th).
  rest32nd and rest64th need a longer extractor run — they're rare
  enough that ~50 000+ label files need to be scanned to find them.
  The label-first prescan in the extractor handles this efficiently;
  just remove `--limit-images` and let it run.
* **Regenerate templates from clean source images.**  The current
  templates were composited from corpus crops which carry some real
  glyph damage (partial erasure, scanner artefacts, ink bleed).  They
  work — the multiset matching is robust to a small amount of damage
  — but ideal templates would come from a clean glyph font (Bravura,
  Maestro, etc.) rendered at the same scale as the corpus.  Either
  render fresh templates from a font, or curate the cleanest single
  example per class from the corpus instead of composite-of-N.
* **Voice separation** on polyphonic staves.  Everything goes to
  `voice=1` today.
* **Tuplets** (`<time-modification>`).  Not handled.
* **Pickup measures / cadenza / unmetered passages** — undocumented
  but real cases in the Georgian corpus.
* **Bracket / brace recognition** for instrument grouping.
* **Pitch validation** for percussion / drum clefs (the
  `CLEF_MIDDLE_PITCH` table only covers G / F / CAlto / CTenor).
* **Cross-staff beaming** (piano notation).
* **Tests for note_alignment** — no unit tests cover the alignment
  module right now.
* **Use the handwritten Georgian corpus.**  A separate collection of
  hand-written (rather than printed) Georgian sheet music exists,
  beyond the 15 100 printed pages already in
  `S:\training_images_crops`.  Currently unused — the model still
  struggles on printed pages, so handwritten generalisation is
  parked as future work.  Don't accidentally claim it was used in
  results / methodology writeups until it actually is.
* **Migrate existing detections.json files to the new page-relative
  schema** — only matters if you want to keep old runs comparable;
  fresh pipeline runs use the new schema automatically.

### Done since the last revision of this doc
* **Page-relative coords in detections.json** — every coordinate field
  (cx/cy/x1/y1/x2/y2) is now page-relative.  Was a mixed schema
  previously (x1/x2 crop-relative, others page-relative) which caused
  silent overlay misalignment on multi-clef pages where `crop_x1 > 0`.
* **Beam-stack counting for 16th/32nd/64th notes.**  Implemented in
  `build_musicxml_v2.count_beams_at` + `BEAM_COUNT_TO_DURATION`.  When
  a noteheadBlack has no detected flag, the number of beam detections
  spanning its x-position decides the duration (1=eighth, 2=16th,
  3=32nd, 4+=64th).  Auto-scales `divisions` accordingly so the
  representation is exact.
* **Multi-page assembly** — deliberately deferred; not on the roadmap
  for the current Masters-thesis deliverable.
* **U-Net staff-line removal** — experiment killed; classical removal
  is the permanent solution.  The detection U-Net stays (it provides
  staff-line coordinates for rectification); only the removal U-Net is
  shelved.  Do not revive without a paired-data generation strategy.
* **Two-sided Chamfer matching in note_alignment.**  `align_note_distance_transform`
  now computes BOTH `dist_to_dark` (ink-side, existing) and `dist_to_light`
  (surround-side, new) and combines them as
  `total = ink_loss + inv_weight * inv_loss`.  Eliminates the failure mode
  where a candidate inside a thick black region (beam, smudge, barline)
  scored a perfect ink_loss=0 because every mask pixel was on a dark
  pixel.  Per-class `inv_weight` in `ALIGN_CONFIG` (1.5 for hollow
  noteheads — the hole+surround distinction is crucial for Half/Whole/
  DoubleWhole; 1.0 for noteheadBlack; 0.8–1.0 for rests).
* **Per-step API endpoints.**  `api/real_api.py` now exposes the
  individual pipeline stages independently of `/process`:
  `POST /staff_detect/unet`, `POST /staff_detect/classical`,
  `POST /note_align`, `POST /note_align/click`,
  `POST /xml_build`.  Designed for the editor frontend that overlays
  XML on the rectified image and lets the user edit detections;
  `/note_align/click` is the single-click snap-to-glyph gesture
  (works on raw-binarized local crops, no cleaned-image dependency).
  `/xml_build` runs the full MusicXML construction from a JSON body —
  feeds straight back into the editor's "save edits" flow.
  `api/api_tester.html` has UI tabs for each, including a "↻ Prefill
  from /detections" helper that builds the `/xml_build` request body
  automatically from the latest `/process` result.
* **YOLO inference defaults tuned for dense OMR.**
  `CONF_THRESHOLD = 0.03`, `IOU_THRESHOLD = 0.8`, `MAX_DET = 5000`
  module-level constants in `scripts/pipeline.py`.  Trades a few
  extra false positives (which the downstream dedup/refine/align
  stages clean up) for substantially better recall on faint glyphs.
  Conservative alternatives documented in the constants' docstring.
* **MusicXML evaluation tool.**  `evaluation/eval_xml.py` produces
  three F1 metrics (pitch+duration / pitch-only / rhythm-only) via
  measureless multiset matching, with a `--aggregate-parts` flag for
  cases where the pipeline collapses multi-voice music into one
  part.  Batch mode walks a piece-organized pred-dir and pairs each
  subfolder to a same-named GT XML.  Pre-Georgian baseline on the
  Abesadzeebi 31-piece subset: F1 = 0.687 (pitch+dur), 0.817 (pitch),
  0.844 (rhythm).
* **Batch pipeline runner.**  `scripts/batch_eval_prep.py` walks a
  piece-organized image folder, runs `pipeline.run_pipeline` per
  page, and arranges per-page XMLs into the layout `eval_xml.py
  batch` expects.  Resumable (skips pages already done) and tolerant
  of per-page errors.
* **YOLO training conveniences.**  `training/yolo_train.py` gained
  `--lr0`, `--lrf`, `--optimizer`, `--warmup-epochs` flags.  Cloud
  fine-tune at default `lr0=0.01` oscillates wildly — use 0.001 for
  fine-tunes.  `scripts/train_nightly.bat` wraps the resume command
  for double-click overnight training on the local 3050.

### Evaluation ground truth — what counts as GT

Two completely different artefacts are sometimes both called "ground
truth" in this project and it matters to keep them distinct:

* **YOLO training labels** at `S:\training_images_crops\*.txt` are
  *pseudo-labels* generated by running a DeepScoresV2-pretrained
  detector over the page-image staff crops.  They are good enough to
  fine-tune the next model from, but they are NOT trustworthy for
  evaluating model quality (the eval would partly be measuring
  agreement-with-itself).

* **Evaluation ground truth** is the Finale-exported MusicXML files
  at `E:\Thesis\Xml-s\xalxuri simRerebi (nawilobriv)\...\*.xml` —
  human-authored representations of the same musical pieces.  These
  are what `evaluation/eval_xml.py` compares pipeline output against,
  and they are what should be cited as "ground truth" in writeups.

The thesis methodology section in particular should be careful with
this distinction — a reviewer asking "how do you know the model is
good?" wants to hear about the Finale XML comparison, not the YOLO
pseudo-labels.

---

## 10. Key files to read first

If picking up cold, read in this order:

1. `S:\omr\README.md` — layout + quick-start commands.
2. `scripts\pipeline.py` — full pipeline entry point with all the
   top-level toggles (RECTIFY, VISUALIZE_DETECTIONS,
   SAVE_BINARIZED_CROPS).
3. `src\score_analyzer.py` — staff detection, row clustering, part
   assignment, `infer_num_parts`.
4. `src\staff_rectifier.py` — Stage 1 details + canvas-coord
   metadata builder.
5. `src\build_musicxml_v2.py` — XML construction logic.
6. `src\measure_recalculator.py` — re-barring logic.
7. This document.

---

## 11. Notable user preferences

* **Measure accuracy doesn't matter for the final product** — splits
  on time-sig changes only, no beat counting.
* **Default `num_parts = 1`** for ambiguous geometry (single-part
  scores are common; over-counting invents empty parts).
* **Labels in `labeled_crops` should not clutter** the image —
  default `show_labels=False`, colour legend at the bottom.
* **No-gap boundaries** between sub-staves after clef-splitting.
* **U-Net staff-line removal is shelved** — classical works.
* **Repo layout matters** — `src/`, `scripts/`, `api/`, `training/`,
  `diagnostics/`, `tests/`, `data/`, `models/`, `docs/`, `archive/`.
* **`.env`-driven paths** — `env_loader.py` is the source of truth.

---

## 12. Known sandbox caveats (only relevant if reading via Linux mount)

When an assistant works on this project via a Linux sandbox, the
bash `wc -l` / `python3 -c "ast.parse..."` calls sometimes return
**truncated views of files that are intact on the Windows host**.
This shows up as fake "SyntaxError: unterminated string" at
specific line numbers; the Read tool (which sees the host) confirms
the file is complete.  Ignore the bash-mount truncation reports;
trust the Read tool.

---

## 13. Quick reference — pipeline, endpoints, files

A flat catalogue.  Skim before diving in.

### 13.1  Pipeline overview (one paragraph)

Input is a score image (PNG/JPG).  **Stage 1** (`staff_rectifier`)
detects every staff via U-Net → YOLO → classical, and warps each
to a flat coordinate frame, producing `rectified.png` plus per-staff
geometry metadata.  **Stage 1b** (`clef_splitter`) runs YOLO once
for clef classes and splits any staff that contains more than one
clef (side-by-side sub-staves).  **Stage 2** (`preprocessing`) takes
the rectified page + the metadata, crops each staff, binarises
(background-flattened Otsu), and removes staff lines via the
classical algorithm — output is a list of `ProcessedStaff` objects
with both colour crops and cleaned (staff-line-removed) crops.
**Stage 3** (`symbol_detector`) runs YOLOv8 once per cleaned crop,
filters out `staff`/`staffLine` classes, runs a same-class
overlap dedup, optionally tightens noteheads via `bbox_refiner`,
optionally runs `note_alignment.refine_page` for template-based
centre refinement.  **Stage 4** (`build_musicxml_v2`) assembles
the detections + staff geometry into MusicXML — pitch from
notehead position relative to clef + key signature, duration from
notehead class + flag/beam analysis, chords from per-staff x-
clustering of noteheads, time signatures interleaved as
`<attributes>` events.  **Stage 4½** (`measure_recalculator`)
splits the single-measure-per-staff output into proper measures
based on time signature changes only (no beat counting).
Output: `score.xml` plus `detections.json` (every coordinate
page-relative; both files share `det_NNNN` IDs).

### 13.2  API endpoints

`api/real_api.py` exposes the pipeline as Flask routes.  The
existing `/process` runs the whole pipeline; the per-step
endpoints let an editor frontend re-run individual stages without
re-uploading the full image every time.

| Method | Path | Input | Output | Purpose |
|---|---|---|---|---|
| GET  | `/`                       | —                                      | `api_tester.html`                   | Browser UI for testing every endpoint |
| GET  | `/health`                 | —                                      | JSON status + readiness             | Used by the sidebar status dot |
| POST | `/process`                | image (multipart `image` or raw body)  | JSON envelope: base64 rectified PNG, detections, MusicXML | Run the full pipeline end-to-end; caches result for the GET endpoints below |
| GET  | `/rectified`              | —                                      | PNG                                 | Last `/process` rectified image |
| GET  | `/detections`             | —                                      | JSON (per-staff schema)             | Last `/process` detections |
| GET  | `/xml`                    | —                                      | `application/xml`                   | Last `/process` MusicXML |
| GET  | `/full`                   | —                                      | JSON envelope of all three          | Last `/process` combined response |
| POST | `/staff_detect/unet`      | image; `?threshold=0.5&inference_size=512` | JSON: list of staves with line positions | Run only the U-Net staff detector |
| POST | `/staff_detect/classical` | image; `?strategy=auto\|printed\|handwritten\|uneven_scan\|permissive` | Same shape as `/staff_detect/unet` | Algorithmic staff detector (no U-Net, no YOLO) — for A/B comparison |
| POST | `/note_align`             | multipart `image` (cleaned staff) + `params` (JSON with detections list, crop_x1, crop_y1) | JSON: each detection echoed with refined cx/cy/x1..y2 and `aligned: bool` | Batch refinement of YOLO detections |
| POST | `/note_align/click`       | multipart `image` (rectified) + `params` (JSON with x, y, class, init_w, init_h, window) | JSON: refined position + box + loss + accepted flag | Single-click "snap to glyph" for editor UIs.  Local-window matching, no cleaned-crop dependency |
| POST | `/xml_build`              | JSON body: `staves` list + optional num_parts/instrument/divisions; `?format=json` to wrap response | `application/xml` (default) or JSON envelope `{xml, summary}` | Build MusicXML from edited detections — closes the editor "save" loop |

The per-step endpoints DO NOT touch the `/process` cache; each call
is independent.  All POST endpoints return
`{"error": "<code>", "message": "..."}` with an appropriate HTTP
status on failure.

### 13.3  File reference

Every Python file in the repo, one line each: role / input / output.

#### `src/` — pipeline modules

* `env_loader.py` — Loads `.env`, exposes path constants (`MODEL_PATH`, `DATA_DIR`, `OUTPUT_BASE_DIR`, etc.), bootstraps `sys.path`. Imported by every entry point.
* `staff_rectifier.py` — **Stage 1**. Input: image. Output: rectified image + per-staff geometry. Multi-strategy detector (U-Net → YOLO → classical) + homography warp.
* `staff_detector_unet.py` — U-Net staff-line detection. Input: grayscale page. Output: list of staff dicts. Used by staff_rectifier and the `/staff_detect/unet` endpoint.
* `staff_detector_yolo.py` — OLA YOLO staff-line detection fallback. Same input/output schema as the U-Net one.
* `staff_unet.py` — U-Net architecture (PyTorch `nn.Module`). Shared by detect + the shelved remove model.
* `clef_splitter.py` — **Stage 1b**. Input: rectified image + Stage 1 staves. Output: same staves list, possibly with multi-clef staves split into sub-staves.
* `preprocessing.py` — **Stage 2**. Input: rectified image + optional precomputed staves. Output: `ProcessedScore` with per-staff cleaned crops.
* `score_analyzer.py` — Staff classification: row clustering, part assignment, `infer_num_parts`. Input: staff list. Output: organized `ScoreData` with per-part lists.
* `staff_remover.py` — Image processing utilities. `binarize()` (background-flattened Otsu), `remove_staff_lines()` (classical algorithm), `clean_staff_image()` (the pipeline composition).
* `symbol_detector.py` — **Stage 3**. Input: `ProcessedScore`. Output: `PageDetections` (and serialized `detections.json`). Wraps YOLOv8 inference + filtering + dedup. Optional `apply_alignment_refinement` calls `note_alignment.refine_page`.
* `bbox_refiner.py` — Connected-components box tightening, run after YOLO. Input: `PageDetections` + `ProcessedScore`. Output: same detections with boxes snapped to actual ink. In-place modification.
* `note_alignment.py` — Template-based distance-transform alignment for notehead/rest centres. Pure-numpy/cv2 (no torch). Functions: `align_note_distance_transform`, `align_note_with_variants`, `refine_detection`, `refine_page`, `load_template`. Two-sided Chamfer matching with per-class `inv_weight` config.
* `build_musicxml_v2.py` — **Stage 4**. Input: `ProcessedScore` + `PageDetections` + output path. Writes MusicXML with embedded `omr-coordinates` JSON. Handles chords, rests, fermatas, time-signature changes, melisma overrides.
* `measure_recalculator.py` — **Stage 4½**. Input: a MusicXML file. Output: same XML re-barred by time-signature events. Re-runnable standalone or CLI: `python measure_recalculator.py in.xml out.xml [--time 3/4]`.
* `xml_builder.py` — lxml-based MusicXML construction helpers. Internal API used by `build_musicxml_v2`.
* `score_to_xml.py` — Legacy v1 XML builder. Kept for reference; not on the pipeline path.

#### `scripts/` — entry-point CLIs

* `pipeline.py` — **End-to-end pipeline orchestrator**. Top-level constants (`RECTIFY`, `VISUALIZE_DETECTIONS`, `SAVE_BINARIZED_CROPS`, `ALIGN_NOTES`, `CONF_THRESHOLD`, `IOU_THRESHOLD`, `MAX_DET`) drive default behaviour. CLI: `python pipeline.py page.png out/ [model.pt] [--debug] [--no-align]`.
* `preprocess_for_training.py` — Batch processor that runs Stage 1+2 on a folder of PNGs, writes one cleaned PNG per staff into `OUTPUT_DIR/images/` plus `manifest.csv`. Multi-process, resumable. Used to prepare training data for YOLO fine-tuning.
* `extract_note_templates.py` — Builds `data/note_templates/<class>.png` from a YOLO-labeled corpus. Median-aspect normalisation + median consensus across N best crops per class. Label-first prescan supports rare classes via `--limit-images N`.
* `batch_eval_prep.py` — Runs `pipeline.run_pipeline` over a piece-organized input folder (one subfolder per piece, each containing page images), arranges per-page XMLs for `eval_xml.py batch`. Resumable, per-page error tolerance.
* `train_nightly.bat` — Windows .bat wrapper for `yolo_train.py --resume`-style overnight training on the 3050 with the right batch=2, lr=0.001 settings. Double-click to run or drag-drop a custom `last.pt` to override the starting checkpoint.

#### `api/` — REST API + clients

* `real_api.py` — Production Flask app. Wraps `pipeline.run_pipeline` plus the per-step endpoints. Single `_PIPELINE_LOCK` serialises GPU inference. CORS enabled. `_LAST_RESULT` cache for the GET endpoints.
* `dummy_api.py` — Fixtures-backed API for frontend dev without the model. Same response shapes as `real_api.py`.
* `api_tester.html` — Browser UI testing all endpoints. Tab per endpoint, "↻ Prefill from /detections" helper for `/xml_build`. Loaded via `GET /` on `real_api.py`.
* `process_score.py` — CLI client that POSTs an image to `/process` and saves the response artefacts. Useful for scripting.
* `api_client_example.py` — Example Python client showing how to call `/process` and walk the response.

#### `training/` — model training

* `yolo_train.py` — YOLO fine-tune entry point. Accepts `--data`, `--weights`, `--name`, `--epochs`, `--batch`, `--imgsz`, `--device`, `--lr0`, `--lrf`, `--optimizer`, `--warmup-epochs`, `--resume`. Writes to `runs/detect/<name>/weights/`.
* `build_data_yaml.py` — Generates `data.yaml` + `train.txt` + `val.txt` + `classes.txt` from a flat folder of paired image/label files. Run this once per machine (paths differ between Windows local and Linux cloud).
* `dataset_prep.py` — Utilities for preparing YOLO training data from CVC-MUSCIMA + DeepScores.
* `pseudo_label.py` — Generates YOLO-format labels by running an existing model over unlabeled images. Used to bootstrap labels for new datasets.
* `staff_unet_train.py` — Trains the staff-line DETECTION U-Net (the one currently in production at `models/staff_unet.pth`).
* `staff_remover_train.py` — Trains the staff-line REMOVAL U-Net. Shelved (the experiment didn't work at inference) but the trainer is kept.
* `muscima_manifest.json` — Manifest of CVC-MUSCIMA images used by `staff_unet_train.py`.

#### `evaluation/` — eval tools

* `eval_xml.py` — Compare pipeline-output MusicXML to Finale-exported ground truth. Multiset matching (no measure alignment), three F1 metrics (pitch+dur / pitch only / rhythm only). Modes: `single` (one pred ↔ one GT) and `batch` (piece-organized pred-dir ↔ GT root). `--aggregate-parts` flag flattens part assignment.

#### `diagnostics/` — ad-hoc dev scripts

* `diagnose_alignment.py` — A/B comparison of two `detections.json` files (one with alignment off, one with it on). Prints per-class centre-shift stats; optional `--overlay` PNG with off→on arrows.
* `diagnose_loaded.py` — Sanity-checks what the dataset loader produces (verifies binarization, mask polarity, etc.).
* `diagnose_mask.py` — Visualisation of mask polarity for the staff U-Net.
* `diagnose_unet_output.py` — Inspects U-Net prediction quality on a sample page.
* `visualizer.py` — General-purpose detection visualiser (drops bounding boxes on an image).

#### `tests/` — unit + smoke tests

* `test_note_alignment.py` — Unit tests for the alignment module. Synthetic data, no YOLO. 12 tests covering mask conversion, distance-transform alignment, variant search, refinement mutation, two-sided figure-ground matching.
* `_smoke_test_v2.py` — Smoke test for the v2 XML builder. Reconstructs detections from an embedded `omr-coordinates` blob and re-runs `build_score_xml_v2`.
* `_reuse_user_score.py` — Replays a previously-saved score through the v2 builder.

#### `data/` — runtime data

* `data/input/` — Test input PNGs.
* `data/sample/` — Dummy-API fixtures (rectified.png, score.xml, detections.json).
* `data/note_templates/` — RGBA glyph templates consumed by `note_alignment`. Currently 9 of 11 classes (rest32nd / rest64th too rare in the current limited extractor scan; full corpus extraction will add them).

#### `models/` — trained weights

* `staff_unet.pth` — Staff-line detection U-Net (~125 MB, val IoU 0.85). Used by `staff_detector_unet`.
* `ola_v2.pt` — Pretrained YOLOv8 staff detector (~40 MB). Fallback used by `staff_rectifier` when U-Net is unavailable.
* `deepscores_crops_v1.pt` — YOLOv8 symbol detector (~52 MB, 124 classes from DeepScoresV2). Stage 3's workhorse. Being fine-tuned to Georgian notation.
* `staff_remover.pth` — Removed-staff-lines U-Net. Trained but UNUSED at inference (the experiment failed; kept for archeology).

---

*Generated as a session-handoff document — supersedes `OMR handoff.md`.*
