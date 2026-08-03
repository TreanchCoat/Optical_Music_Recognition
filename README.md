# OMR Pipeline

An end-to-end Optical Music Recognition (OMR) pipeline that converts scanned sheet music images (PNG/JPG) into structured MusicXML files.

## How It Works

The pipeline processes score images in multiple stages:
* **Stage 1 (Rectify):** Detects every staff using a multi-strategy approach (U-Net → YOLO → classical) and warps each to a flat coordinate frame, producing a rectified image.
* **Stage 1b (Clef-split):** Runs YOLO to detect clefs and splits any staff containing multiple side-by-side clefs into sub-staves.
* **Stage 2 (Preprocess):** Crops each staff from the rectified page, binarizes the image (background-flattened Otsu), and removes staff lines algorithmically to produce cleaned crops.
* **Stage 3 (Symbol Detection):** Runs a YOLOv8 object detector on each cleaned crop to find musical symbols, followed by deduplication, bounding-box refinement, and optional template-based center alignment for noteheads and rests.
* **Stage 4 (Assembly & MusicXML):** Assembles detections and staff geometry into MusicXML. It determines pitch from notehead positions relative to clefs/key signatures, and duration from notehead classes and beam analysis. Chords are grouped by horizontal clustering, and time signatures are interleaved as events.
* **Stage 4½ (Re-barring):** Splits the single-measure-per-staff output into proper measures based purely on time signature changes.

## Quick Start

### Command Line Interface (CLI)

```cmd
:: Process a single score image (End-to-End)
python scripts\pipeline.py data\input\page1.png output\

:: Generate training crops from a folder of pages
python scripts\preprocess_for_training.py data\input\ output\staves\

:: Re-bar an XML after manual edits
python src\measure_recalculator.py score.xml score_rebar.xml --time 3/4
```

### REST API Server

Start the API server to expose the pipeline via HTTP endpoints:
```cmd
python api\real_api.py
```
*(Or use `python api\dummy_api.py` for fixture-based responses without loading models.)*

Once running, you can access the browser testing UI at `http://localhost:5000/`.

**Key API Routes:**
- `POST /process`: Submit a score image (multipart form with `image` field) to run the full pipeline. Returns a JSON envelope with a base64 rectified PNG, raw detections, and the MusicXML string.
- `GET /xml`: Download the generated MusicXML file from the last `/process` job.
- **Per-step endpoints** (`/staff_detect/unet`, `/note_align`, `/xml_build`, etc.) are also available for interactive editor frontends.

## Configuration

All paths and runtime settings are read from a `.env` file, loaded automatically by `src\env_loader.py`. 
- The supported variables are documented in `.env.example`.
- Default values derived from the project root keep everything working with zero `.env` setup. You only need to create a `.env` file if you want to store data or models outside the repository.

## Output Artifacts

After a successful pipeline run, the output directory will contain:
- `score.xml` — The final reconstructed MusicXML file.
- `detections.json` — Raw YOLO detections with page-relative coordinates, cross-referenced with `det_NNNN` IDs in the XML.
- `rectified.png` — The straightened version of the input page.
- `stage1_staves_on_input.png` / `symbols_detected.png` — Visualizations of bounding boxes.
- `labeled_crops/` & `binarized/` — Per-staff intermediate crops for diagnostics and training.

## Project Layout

- `src/` — Pipeline source modules (rectifier, analyzer, XML builder, U-Net, etc.).
- `scripts/` — Entry-point CLIs (pipeline, batch processors, model training prep).
- `api/` — REST API application, clients, and tester UI.
- `training/` — Model-training scripts (YOLO, U-Net, dataset prep).
- `models/` — Trained weights (.pth, .pt).
- `docs/` — Further documentation (Methodology, system handoff notes).
- `diagnostics/` / `tests/` — Ad-hoc dev scripts and regression tests.

## Note
- because of the file size, the U-net model(staff detector for rectified mode) isn't uploaded. On regular mode system should work just fine
