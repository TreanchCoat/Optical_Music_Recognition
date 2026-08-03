"""
yolo_train.py — Fine-tune a YOLOv8 detection model for OMR symbol detection.

Pipeline position
-----------------
1. ``scripts/preprocess_for_training.py``  →  staff crops in PREP_DIR/images/
2. ``training/pseudo_label.py``            →  YOLO labels + data.yaml
3. **this script**                          →  fine-tuned .pt model

By default this initialises from ``deepscores_crops_v1.pt`` (the model the
pipeline already uses), so the class set and head dimensions match and the
new weights are drop-in compatible with ``src/symbol_detector.py``.

Usage
-----
    python yolo_train.py --data DATA.YAML [options]

Examples
--------
    # Fine-tune on the handwritten dataset, single GPU
    python yolo_train.py --data S:\\omr\\training_data\\handwritten\\data.yaml

    # Custom run name, more epochs, larger image size
    python yolo_train.py \\
        --data S:\\omr\\training_data\\printed\\data.yaml \\
        --name printed_v1 --epochs 200 --imgsz 1536 --batch 6

    # Train from a generic YOLOv8s checkpoint instead of the existing model
    python yolo_train.py --data DATA.yaml --weights yolov8s.pt --epochs 100

    # Resume an interrupted run (re-uses the same project/name)
    python yolo_train.py --data DATA.yaml --name handwritten_v1 --resume

Output
------
Ultralytics writes everything to::

    <project>/runs/detect/<name>/
        weights/best.pt        ← copy this into S:\\omr\\models\\
        weights/last.pt
        results.png, results.csv, args.yaml, etc.

Notes
-----
* ``imgsz=1280`` matches the pipeline's inference size; thin glyphs need it.
* On an 8 GB GPU (RTX 3050 / 4060), ``batch=4`` at 1280 px is usually safe.
  Drop to ``batch=2`` if you OOM.  Use ``--device cpu`` for a sanity run.
* Vast.ai 4090 / 5090 (24 GB) can comfortably handle ``batch=16`` at 1280 px.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Bootstrap so env_loader is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
import env_loader  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog='yolo_train',
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--data', required=True,
                    help='Path to data.yaml (produced by pseudo_label.py).')
    ap.add_argument('--weights', default=str(env_loader.MODEL_PATH),
                    help=f'Starting weights (.pt).  Defaults to '
                         f'env_loader.MODEL_PATH ({env_loader.MODEL_PATH}).')
    ap.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs (default 100).')
    ap.add_argument('--imgsz', type=int, default=1280,
                    help='Training image size (default 1280, matches '
                         'pipeline inference).')
    ap.add_argument('--batch', type=int, default=8,
                    help='Batch size (default 8).  Drop to 2-4 on 8 GB GPUs.')
    ap.add_argument('--device', default='0',
                    help="Ultralytics device string (e.g. '0', '0,1', "
                         "'cpu').  Default '0'.")
    ap.add_argument('--workers', type=int, default=4,
                    help='DataLoader workers (default 4).')
    ap.add_argument('--name', default='omr_yolo',
                    help='Run name; results go to runs/detect/<name>/.')
    ap.add_argument('--project', default=str(_PROJECT_ROOT / 'runs' / 'detect'),
                    help='Project dir for Ultralytics outputs.')
    ap.add_argument('--patience', type=int, default=30,
                    help='Early-stopping patience in epochs (default 30).')
    ap.add_argument('--resume', action='store_true',
                    help='Resume the run named --name from its last checkpoint.')
    ap.add_argument('--seed', type=int, default=0,
                    help='Random seed (default 0).')
    ap.add_argument('--lr0', type=float, default=None,
                    help='Initial learning rate (default: Ultralytics auto, '
                         'which is 0.01 for SGD-like optimisers).  Override '
                         'when FINE-TUNING from a near-converged checkpoint '
                         '— 0.01 is too aggressive and produces oscillation. '
                         'Try 0.001 for a moderate fine-tune, 0.0001 for a '
                         'gentle one.')
    ap.add_argument('--lrf', type=float, default=None,
                    help='Final learning rate as a multiplier of lr0 '
                         '(default: Ultralytics auto = 0.01).  Lower values '
                         'mean stronger cosine decay toward the end.')
    ap.add_argument('--optimizer', default='auto',
                    help="Optimiser name: 'auto', 'SGD', 'Adam', 'AdamW', "
                         "'NAdam', 'RAdam', 'RMSProp' (default 'auto', which "
                         "Ultralytics picks via heuristic).")
    ap.add_argument('--warmup-epochs', type=float, default=None,
                    help='Number of warmup epochs (default Ultralytics 3.0).  '
                         'Lower for fine-tuning since the model is already '
                         'close to good.')
    args = ap.parse_args(argv)

    data_path    = Path(args.data).resolve()
    weights_path = Path(args.weights).resolve()
    project_dir  = Path(args.project).resolve()

    if not data_path.is_file():
        print(f'data.yaml not found: {data_path}', file=sys.stderr)
        return 2

    if not args.resume and not weights_path.is_file():
        # Ultralytics also accepts bare names like 'yolov8s.pt' which it
        # auto-downloads.  Only warn when the user passed an explicit path
        # that doesn't exist.
        if any(sep in args.weights for sep in ('/', '\\')):
            print(f'Weights file not found: {weights_path}', file=sys.stderr)
            return 2

    project_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — Ultralytics is heavy.
    from ultralytics import YOLO

    if args.resume:
        # Ultralytics will pick up <project>/<name>/weights/last.pt
        ckpt = project_dir / args.name / 'weights' / 'last.pt'
        if not ckpt.is_file():
            print(f'Cannot resume — no checkpoint at {ckpt}', file=sys.stderr)
            return 2
        print(f'Resuming from {ckpt}')
        model = YOLO(str(ckpt))
    else:
        print(f'Initialising model from {weights_path}')
        model = YOLO(str(weights_path))

    print(f'Data:    {data_path}')
    print(f'Epochs:  {args.epochs}    imgsz={args.imgsz}    batch={args.batch}')
    print(f'Device:  {args.device}    workers={args.workers}')
    print(f'Project: {project_dir}    name={args.name}')

    train_kwargs = dict(
        data       = str(data_path),
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        device     = args.device,
        workers    = args.workers,
        name       = args.name,
        project    = str(project_dir),
        patience   = args.patience,
        seed       = args.seed,
        resume     = args.resume,
        exist_ok   = args.resume,
        optimizer  = args.optimizer,
    )
    # Forward optional LR / warmup overrides only when given so we don't
    # override Ultralytics' defaults with None.
    if args.lr0 is not None:
        train_kwargs['lr0'] = args.lr0
    if args.lrf is not None:
        train_kwargs['lrf'] = args.lrf
    if args.warmup_epochs is not None:
        train_kwargs['warmup_epochs'] = args.warmup_epochs

    if args.lr0 is not None:
        print(f'LR:      lr0={args.lr0}'
              f'{" lrf=" + str(args.lrf) if args.lrf is not None else ""}')
    if args.optimizer != 'auto':
        print(f'Optim:   {args.optimizer}')

    results = model.train(**train_kwargs)

    # Print final summary + best weights path.
    save_dir = getattr(results, 'save_dir', project_dir / args.name)
    best = Path(save_dir) / 'weights' / 'best.pt'
    if best.is_file():
        print()
        print(f'Best weights: {best}')
        print('To use the new model in the pipeline, copy or symlink it '
              f'into {env_loader.MODELS_DIR}\\ and update MODEL_PATH '
              'in .env (or pass --model in symbol_detector calls).')
    else:
        print()
        print(f'Training finished.  Inspect {save_dir} for results.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
