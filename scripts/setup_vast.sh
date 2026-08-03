#!/usr/bin/env bash
# ============================================================
# setup_vast.sh — provision a vast.ai (Linux + CUDA) instance and run the
# OMR eval-prep over the full corpus.  Resumable: re-run to continue.
#
# Upload to the instance BEFORE running (see folder layout in the chat):
#   /workspace/detector.pt        the 54-class model (crops54-2 best.pt)
#   /workspace/pages/             corpus images (mirror of "PDF Files")
#   /workspace/note_templates/    alignment templates (your Bravura set)
# Then:  bash setup_vast.sh
# ============================================================
set -e

# ---- CONFIG: edit if your paths differ --------------------
REPO_URL="https://github.com/TreanchCoat/OMR.git"
WORK="/workspace"
MODEL_FILE="$WORK/detector.pt"          # the 54-class detector you uploaded
TEMPLATES="$WORK/note_templates"        # your regenerated alignment templates
INPUT_DIR="$WORK/pages"                 # corpus images (mirror of PDF Files)
OUTPUT_DIR="$WORK/eval_out"             # predictions written here
# -----------------------------------------------------------

cd "$WORK"

# 1. code
[ -d "$WORK/OMR" ] || git clone "$REPO_URL"
cd "$WORK/OMR"

# 2. dependencies — install a CUDA build of torch FIRST (skip if the
#    base image already ships torch+CUDA), then everything else.
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. model + templates into place
mkdir -p models data/note_templates
cp "$MODEL_FILE" models/detector.pt
cp "$TEMPLATES"/*.png data/note_templates/ 2>/dev/null || true

# 4. config via env vars (these win over the committed .env)
export MODEL_PATH="$WORK/OMR/models/detector.pt"
export DATA_DIR="$WORK/OMR/data"
unset ALIGN_LOSS_LOG

# 5. confirm the model is the 54-class one before a long run
python -c "from ultralytics import YOLO; n=len(YOLO('$MODEL_PATH').names); print('detector classes:', n); assert n==54, 'NOT the 54-class model!'"

# 6. run the full corpus (resumable — re-run to continue)
python scripts/batch_eval_prep.py --input "$INPUT_DIR" --output "$OUTPUT_DIR"

echo "Done.  Download $OUTPUT_DIR to your machine and run eval_xml locally."
