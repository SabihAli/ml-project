"""
snap_ball_labels.py -- tighten the noisy GSR ball boxes in a ball-crop dataset (BALL_TEMPORAL_PLAN.md).

For every label in --splits (default train, val), each ball box is replaced by the nearest
detection of --weights if that detection is close (centre within max(MIN_PX, SNAP_FRAC * size))
and similar in size (0.5x .. 2x). Otherwise the original box is kept. Test labels are left
alone so the gates stay comparable.

Originals are copied to labels_orig/<split>/ once (never overwritten), and every run snaps from
those originals, so re-running is safe. Ultralytics' labels/*.cache files are deleted.

    python snap_ball_labels.py --data /kaggle/working/ball_crops \
        --weights /kaggle/working/runs/s_baseline/weights/best.pt --device 0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

MIN_PX = 12.0      # minimum snap radius, just under the gate's 15 px match distance
SNAP_FRAC = 0.8    # snap radius as a fraction of the ball's larger side
SIZE_RATIO = 2.0   # detection size must be within 1/SIZE_RATIO .. SIZE_RATIO of the label's
CROP = 640


def snap(gt: np.ndarray, det: np.ndarray) -> tuple[np.ndarray, int]:
    """gt, det: (N,4) / (M,4) cx,cy,w,h in px. Returns snapped gt and how many changed."""
    out, n, used = gt.copy(), 0, np.zeros(len(det), bool)
    for i, (cx, cy, w, h) in enumerate(gt):
        if not len(det):
            break
        d = np.hypot(det[:, 0] - cx, det[:, 1] - cy)
        s = np.maximum(det[:, 2], det[:, 3]) / max(w, h, 1e-6)
        ok = (d <= max(MIN_PX, SNAP_FRAC * max(w, h))) & (s >= 1 / SIZE_RATIO) & (s <= SIZE_RATIO) & ~used
        if ok.any():
            j = int(np.flatnonzero(ok)[d[ok].argmin()])
            out[i], used[j], n = det[j], True, n + 1
    return out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="ball_crops dir from build_ball_crop_dataset.py")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--imgsz", type=int, default=CROP,
                    help="640 for a specialist; 416 for the main detector (its 0.667 frame scale)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    # A multi-class model (the main athlete/ball detector) is filtered to its 'ball' class.
    ball = [k for k, v in model.names.items() if v == "ball"] if len(model.names) > 1 else None
    kw = dict(imgsz=args.imgsz, conf=args.conf, max_det=10, device=args.device, verbose=False, classes=ball)
    if args.half:
        kw["half"] = True

    for split in args.splits:
        lab_dir, orig_dir = args.data / "labels" / split, args.data / "labels_orig" / split
        if not orig_dir.exists():
            shutil.copytree(lab_dir, orig_dir)
        todo = [p for p in sorted(orig_dir.glob("*.txt")) if p.stat().st_size]
        boxes = changed = 0
        for k in range(0, len(todo), args.batch):
            chunk = todo[k:k + args.batch]
            imgs = [str(args.data / "images" / split / (p.stem + ".jpg")) for p in chunk]
            for p, r in zip(chunk, model.predict(imgs, **kw)):
                rows = np.loadtxt(p, ndmin=2)
                gt = rows[:, 1:5] * CROP
                det = r.boxes.xywh.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
                new, n = snap(gt, det)
                boxes, changed = boxes + len(gt), changed + n
                rows[:, 1:5] = np.clip(new / CROP, 0, 1)
                (lab_dir / p.name).write_text("".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                                                      for c, x, y, w, h in rows))
            print(f"\r{split}: {min(k + args.batch, len(todo))}/{len(todo)} files", end="", flush=True)
        print(f"\n{split}: snapped {changed}/{boxes} boxes ({100 * changed / max(boxes, 1):.1f}%)")
    for c in (args.data / "labels").glob("*.cache"):
        c.unlink()


if __name__ == "__main__":
    main()
