"""
infer_athlete_video.py — run the athlete/ball detector over a clip with per-class confidence
thresholds and write an annotated mp4.

The model (runs/yolo26s_athlete_ball) has two classes, `athlete` and `ball`. The
ball is tiny and hard, so it gets its own, much lower threshold than the athletes.

Works on a SoccerNet-GSR sequence folder (img1/ or images/ of frames), a plain folder of
frames, or a video file. Runs on Kaggle or locally; the device is picked automatically.

Usage:
    python scripts/infer_athlete_video.py --source <path>/SNGS-033
    python scripts/infer_athlete_video.py --source clip.mp4 --athlete_conf 0.4 --ball_conf 0.10

Needs ultralytics>=8.4.0 (YOLO26).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "runs" / "yolo26s_athlete_ball" / "best.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ATHLETE_COLOR = (0, 90, 255)   # BGR -> orange-red
BALL_COLOR = (0, 255, 255)     # BGR -> yellow


def iter_frames(source: Path) -> Iterator[np.ndarray]:
    if source.is_file():
        capture = cv2.VideoCapture(str(source))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
        capture.release()
        return
    frame_dir = next((source / name for name in ("img1", "images") if (source / name).is_dir()), source)
    for path in sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        yield cv2.imread(str(path))


def draw(frame: np.ndarray, boxes: List[Tuple[str, float, np.ndarray]]) -> np.ndarray:
    scale = frame.shape[1] / 1920
    thickness = max(1, round(2 * scale))
    for name, conf, (x1, y1, x2, y2) in boxes:
        colour = BALL_COLOR if name == "ball" else ATHLETE_COLOR
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, colour, thickness)
        label = f"{name} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale + 0.3, thickness)
        top = max(p1[1] - th - base - 2, 0)
        cv2.rectangle(frame, (p1[0], top), (p1[0] + tw + 4, top + th + base + 2), colour, -1)
        cv2.putText(frame, label, (p1[0] + 2, top + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale + 0.3,
                    (0, 0, 0), max(1, thickness - 1), cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Athlete/ball inference with per-class thresholds -> annotated mp4")
    parser.add_argument("--source", required=True, help="sequence folder (img1/), folder of frames, or a video file")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--athlete_conf", type=float, default=0.4)
    parser.add_argument("--ball_conf", type=float, default=0.10, help="kept low on purpose: the ball is the hard class")
    parser.add_argument("--imgsz", type=int, default=1280, help="the model was trained at 1280")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--out", default=None, help="output mp4 (default: <weights dir>/inference_out/<source>_athlete<conf>.mp4)")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights)
    name_of = model.names
    ids = {v: k for k, v in name_of.items()}
    if "athlete" not in ids or "ball" not in ids:
        raise SystemExit(f"expected classes athlete + ball, model has {name_of}")
    threshold = {ids["athlete"]: args.athlete_conf, ids["ball"]: args.ball_conf}

    source = Path(args.source)
    out = Path(args.out) if args.out else (
        Path(args.weights).parent / "inference_out" / f"{source.stem}_athlete{args.athlete_conf:g}_ball{args.ball_conf:g}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    n_frames = n_athletes = n_ball_frames = 0
    started = time.perf_counter()

    def flush(batch: List[np.ndarray]) -> None:
        nonlocal writer, n_frames, n_athletes, n_ball_frames
        results = model.predict(batch, imgsz=args.imgsz, conf=min(threshold.values()), device=device, verbose=False)
        for frame, result in zip(batch, results):
            kept = []
            for box in result.boxes:
                cls, conf = int(box.cls[0]), float(box.conf[0])
                if conf >= threshold[cls]:
                    kept.append((name_of[cls], conf, box.xyxy[0].cpu().numpy()))
            n_athletes += sum(1 for k in kept if k[0] == "athlete")
            n_ball_frames += int(any(k[0] == "ball" for k in kept))
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(draw(frame, kept))
            n_frames += 1
        print(f"  {n_frames} frames  ({(time.perf_counter() - started) / n_frames:.2f}s/frame)", flush=True)

    pending: List[np.ndarray] = []
    for frame in iter_frames(source):
        pending.append(frame)
        if len(pending) == args.batch:
            flush(pending)
            pending = []
        if args.max_frames and n_frames + len(pending) >= args.max_frames:
            break
    if pending:
        flush(pending[: args.max_frames - n_frames] if args.max_frames else pending)
    if writer is not None:
        writer.release()

    print(f"\nwrote {out}")
    print(f"device={device}  imgsz={args.imgsz}  athlete conf >= {args.athlete_conf}  ball conf >= {args.ball_conf}")
    print(f"{n_frames} frames | {n_athletes / max(n_frames, 1):.1f} athletes/frame | ball in {n_ball_frames}/{n_frames} frames")


if __name__ == "__main__":
    main()
