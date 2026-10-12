"""
train_ball_specialist.py -- train the 640x640 ball specialist (docs/path_b/plans/BALL_TEMPORAL_PLAN.md §5).

Standalone (no gsr_pipeline import). Needs ultralytics>=8.4.0 (YOLO26).

The model is built from a YOLO26 yaml with nc=1, then the backbone/neck weights of the main
athlete/ball detector are transferred (layers whose shapes don't match, i.e. the class head,
stay freshly initialised). This works for both single-GPU and DDP runs: in DDP the worker
processes rebuild the model from the yaml and reload the same weights via `pretrained`.

Baseline, both T4s (DDP):
    python scripts/train_ball_specialist.py --data /kaggle/working/ball_crops/dataset.yaml \
        --weights /kaggle/working/kit/best.pt --device 0,1 --batch 64 --name s_baseline

Two ablations in parallel, one per GPU:
    python scripts/train_ball_specialist.py ... --device 0 --batch 32 --model yolo26s-p2.yaml --name s_p2 &
    python scripts/train_ball_specialist.py ... --device 1 --batch 32 --model yolo26n.yaml   --name n_base &

Resume after a Kaggle session ended:
    python scripts/train_ball_specialist.py --resume /kaggle/working/runs/s_baseline/weights/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# Plan §5.2. Position and scale come from the dataset, and the runtime feeds crops 1:1, so
# geometric augments that would move the ball off native scale are off or small.
TRAIN_CFG = dict(
    imgsz=640, epochs=100, patience=25, amp=True, optimizer="auto", cos_lr=True,
    close_mosaic=0, max_det=10,         # no single_cls: the data is already 1-class, and
                                        # single_cls would rename the class 'ball' -> 'item'
    seed=0, deterministic=True, cache=False,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, scale=0.1, translate=0.05, degrees=0.0,
    shear=0.0, perspective=0.0, fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
    plots=True,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="dataset.yaml from build_ball_crop_dataset.py")
    ap.add_argument("--model", default="yolo26s.yaml",
                    help="architecture yaml: yolo26s.yaml, yolo26s-p2.yaml, yolo26n.yaml")
    ap.add_argument("--weights", default=None,
                    help="weights to transfer: the main detector's best.pt (baseline), or "
                         "yolo26s.pt for the COCO ablation")
    ap.add_argument("--device", default="0,1")
    ap.add_argument("--batch", type=int, default=64, help="total batch across all GPUs in --device")
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers per GPU process")
    ap.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    ap.add_argument("--patience", type=int, default=TRAIN_CFG["patience"])
    ap.add_argument("--project", default="/kaggle/working/runs")
    ap.add_argument("--name", default="s_baseline")
    ap.add_argument("--resume", type=Path, default=None, help="path to a run's weights/last.pt")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override any TRAIN_CFG / Ultralytics train arg, e.g. --set scale=0.3 freeze=10")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        YOLO(str(args.resume)).train(resume=True)
        return
    if not args.data:
        ap.error("--data is required unless --resume")

    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)          # transfers matching layers; sets `pretrained` for DDP workers
    cfg = {**TRAIN_CFG, "epochs": args.epochs, "patience": args.patience}
    for kv in args.set:
        k, _, v = kv.partition("=")
        cfg[k] = yaml.safe_load(v)        # "0.3" -> float, "10" -> int, "true" -> bool
    model.train(data=str(args.data), device=args.device, batch=args.batch, workers=args.workers,
                project=args.project, name=args.name, exist_ok=False, **cfg)


if __name__ == "__main__":
    main()
