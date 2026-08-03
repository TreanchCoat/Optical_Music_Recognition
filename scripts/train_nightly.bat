@echo off
REM ─────────────────────────────────────────────────────────────────────
REM train_nightly.bat — resume YOLO training overnight on the 3050.
REM
REM Usage
REM   1. Double-click to run with the default checkpoint, OR
REM   2. Drag and drop a custom last.pt onto this file to use that one.
REM
REM Stop with Ctrl+C any time after at least one epoch has completed.
REM Tomorrow night, edit DEFAULT_WEIGHTS below to point at the newest
REM last.pt (under S:\omr\runs\detect\nightly-N\weights\).
REM ─────────────────────────────────────────────────────────────────────

setlocal

REM The starting weights for the very first night (your cloud-trained
REM checkpoint).  After that, update this to the latest nightly run's
REM last.pt — or drop the .pt file onto this batch file as a drag-drop.
set DEFAULT_WEIGHTS=S:\omr\runs\detect\georgian_v2-3\weights\last.pt

REM Resolve weights path — drag-drop arg overrides default
if "%~1"=="" (
    set WEIGHTS=%DEFAULT_WEIGHTS%
) else (
    set WEIGHTS=%~1
)

if not exist "%WEIGHTS%" (
    echo ERROR: weights file not found:  %WEIGHTS%
    echo Edit DEFAULT_WEIGHTS in this .bat to point at the latest checkpoint.
    pause
    exit /b 1
)

cd /d S:\omr

echo Starting nightly training from:
echo   %WEIGHTS%
echo.
echo Press Ctrl+C after at least one epoch completes to save and quit.
echo.

REM NOTE: --optimizer SGD is required for --lr0 to actually take effect.
REM Ultralytics' default `optimizer=auto` mode SILENTLY IGNORES the --lr0
REM argument and picks its own (typically 0.01).  We need SGD explicitly
REM forced so the lr=0.001 fine-tune setting holds.  Check the first
REM line of results.csv after epoch 1 — lr/pg0 should read ~0.001, not
REM ~0.01.  If it reads 0.01, the .bat's --optimizer flag is broken.
python training\yolo_train.py ^
    --data S:\training_images_crops\data.yaml ^
    --weights "%WEIGHTS%" ^
    --name nightly ^
    --epochs 200 ^
    --batch 2 ^
    --imgsz 1280 ^
    --device 0 ^
    --optimizer SGD ^
    --lr0 0.001 ^
    --warmup-epochs 0.5

echo.
echo Training stopped.  Latest weights at:
echo   S:\omr\runs\detect\nightly\weights\last.pt   (or nightly-N for repeated runs)
pause
