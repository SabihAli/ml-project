"""
pseudo_label.py -- run the adopted Phase 3b teacher on unlabelled footage and write ball labels for the students
(docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md, Phase 3b, "After the gate").

Teacher (Phase 3b.2, closed 2026-11-17): T2 (`t2m_s1920_ft`, YOLO26s, 5 ch) + T1 (`t1_r34`, 5-frame heatmap),
top-5 candidates each, fused 0.7 / 0.3 (`fuse_candidates.fuse_frame`), decoded per clip with the frozen
`decode_trajectory.py` knobs. The same code path as the evaluated teacher: on a GSR clip it must reproduce the
candidate-dump numbers (`--check`).

Input: a folder of consecutive 1920x1080 frames (`--frames_dir`), or a video window (`--video`, `--start_s`,
`--seconds`), extracted at native frame rate to `<out>/frames/`. Not 1920x1080 -> refused (the knobs are in px/frame
at 1080p 25 fps; see the plan).

Output `<out>/labels.json`: "frames" {name: [x, y, score] | null} at `--min_score` (thresholded, then gaps filled,
as evaluated; default 0.06, the evaluated operating point), and "path" {name: [x, y, score]}, every decoded path node
before thresholding, so a stricter set can be rebuilt without rerunning the networks. `<out>/yolo/*.txt`: one class-0
box per labelled frame, size from T2's box when it saw the ball, else `--box_px`.

    python scripts/pipeline_improvement_1/phase3b/pseudo_label.py --frames_dir data/gsr/test/SNGS-021/img1 --out <dir> --check
    python scripts/pipeline_improvement_1/phase3b/pseudo_label.py --video match/1_HQ.mkv --start_s 600 --seconds 120 --out <dir>
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_trajectory as dt  # noqa: E402
import fuse_candidates as fc  # noqa: E402
import train_teacher_heatmap as t1m  # noqa: E402
import train_teacher_yolo as base  # noqa: E402
import train_teacher_yolo_motion as t2m  # noqa: E402

T2_WEIGHTS = base.ROOT / "runs" / "teacher" / "t2m_s1920_ft" / "weights" / "best.pt"
T1_WEIGHTS = base.ROOT / "runs" / "teacher" / "t1_r34" / "best.pt"
ALPHA = 0.7
CFG = (0.08, 0.05, 8.0, 25.0, 60.0, 10, 0.0, 0.2, 1, 10)       # decode_trajectory knobs chosen on select
THR = 0.06                                                      # the evaluated operating point
TOPK, CONF = 5, 0.001
JPEG_Q = 90                                                     # ~ GSR's own frames (~300 KB at 1080p)


def extract(video: Path, start_s: float, seconds: float, out: Path) -> Path:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    out.mkdir(parents=True, exist_ok=True)
    n = int(round(seconds * fps))
    for i in range(n):
        ok, im = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out / f"{i + 1:06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    cap.release()
    print(f"extracted {i + 1} frames at {fps:.2f} fps -> {out}", flush=True)
    if abs(fps - 25) > 0.5:
        print(f"WARNING: {fps:.2f} fps, the knobs assume 25", flush=True)
    return out


class FrameSeq(t1m.Seq):
    """A frame folder as T1's Seq (frames numbered 1..n in sorted file order)."""

    def __init__(self, files: list[Path]):
        self.name = files[0].parent.name
        self.frames = list(range(1, len(files) + 1))
        self.paths = {f: str(p) for f, p in zip(self.frames, files)}
        self.balls = {}


@functools.lru_cache(maxsize=None)
def _t2_model():
    from ultralytics import YOLO
    return YOLO(str(T2_WEIGHTS))


@functools.lru_cache(maxsize=None)
def _t1_model(device: str):
    import torch
    dev = torch.device(device if device == "cpu" else "cuda")
    ck = torch.load(T1_WEIGHTS, map_location=dev, weights_only=False)
    model = t1m.MultiFrameHeatmapNet(ck.get("backbone", "resnet34"), pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    return model, dev


def t2_candidates(files: list[Path], device: str, batch: int) -> tuple[dict, dict]:
    """{frame: [(x, y, conf)]}, {frame: [(w, h)]} from T2, best first."""
    model = _t2_model()
    cache, cands, sizes = {}, {}, {}
    n = len(files)

    def get(k):                                    # 1-based index; outside the clip -> None (stack uses t)
        if k < 1 or k > n:
            return None
        if k not in cache:
            cache[k] = cv2.imread(str(files[k - 1]), cv2.IMREAD_COLOR)
        return cache[k]

    for s in range(1, n + 1, batch):
        fs = list(range(s, min(s + batch, n + 1)))
        ims = [t2m.stack(get(f), [get(f + o) for o in t2m.OFFS]) for f in fs]
        for k in [k for k in cache if k < fs[0] - max(t2m.DTS)]:
            del cache[k]
        res = model.predict(ims, imgsz=1920, conf=CONF, iou=0.5, max_det=50, half=device != "cpu",
                            device=device, verbose=False)
        for f, r in zip(fs, res):
            b = r.boxes
            if b is None or not len(b):
                cands[f], sizes[f] = [], []
                continue
            order = b.conf.argsort(descending=True)[:TOPK].tolist()
            xy, cf = b.xyxy.tolist(), b.conf.tolist()
            cands[f] = [((xy[j][0] + xy[j][2]) / 2, (xy[j][1] + xy[j][3]) / 2, cf[j]) for j in order]
            sizes[f] = [(xy[j][2] - xy[j][0], xy[j][3] - xy[j][1]) for j in order]
    return cands, sizes


def t1_candidates(files: list[Path], device: str, batch: int) -> dict:
    import torch
    model, dev = _t1_model(device)
    with torch.no_grad():
        return t1m.infer_seq(model, FrameSeq(files), dev, batch, TOPK, CONF)


def label(files: list[Path], device: str, batch: int) -> tuple[dict, dict]:
    t0 = time.time()
    c2, sz = t2_candidates(files, device, batch)
    print(f"T2 {len(files) / (time.time() - t0):.1f} fps", flush=True)
    t0 = time.time()
    c1 = t1_candidates(files, device, max(1, batch // 2))
    print(f"T1 {len(files) / (time.time() - t0):.1f} fps", flush=True)
    frames = sorted(c2)
    fused = [fc.fuse_frame(c2.get(f, []), c1.get(f, []), ALPHA, 15.0, TOPK) for f in frames]
    fr, xy, cf = dt.arrays(frames, fused)
    path = dt.decode(fr, xy, cf, *CFG[:8])
    return (fr, xy, cf, path), {f: (c2[f], sz[f]) for f in frames}


def box_size(p, t2, box_px: float) -> tuple[float, float]:
    cands, sizes = t2
    for (x, y, _), wh in zip(cands, sizes):
        if np.hypot(x - p[0], y - p[1]) <= 15:
            return wh
    return box_px, box_px


def label_dir(frames_dir: Path, out: Path, device: str, batch: int, min_score: float = THR,
              box_px: float = 12.0) -> tuple[list[Path], dict]:
    """Label one folder of consecutive frames; writes out/labels.json and out/yolo/*.txt."""
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    h, w = cv2.imread(str(files[0])).shape[:2]
    if (w, h) != (1920, 1080):
        raise SystemExit(f"{w}x{h}: the teacher's knobs assume 1920x1080")
    (fr, xy, cf, path), t2 = label(files, device, batch)
    # threshold before gap fill, exactly as evaluated (a low path node inside a gap must not block the fill)
    preds = dt.outputs(fr, xy, cf, path, CFG[8], CFG[9], min_score)
    raw = dt.outputs(fr, xy, cf, path, CFG[8], 0, 0.0)            # every path node with its score, no fill
    out.mkdir(parents=True, exist_ok=True)
    doc = {"teacher": {"t2": str(T2_WEIGHTS), "t1": str(T1_WEIGHTS), "alpha": ALPHA, "decode": CFG},
           "min_score": min_score,
           "frames": {files[f - 1].name: (list(p) if p is not None else None) for f, p in preds.items()},
           "path": {files[f - 1].name: list(p) for f, p in raw.items() if p is not None}}
    (out / "labels.json").write_text(json.dumps(doc))
    yd = out / "yolo"
    yd.mkdir(exist_ok=True)
    n = 0
    for f, p in preds.items():
        txt = ""
        if p is not None:
            bw, bh = box_size(p, t2[f], box_px)
            txt = f"0 {p[0] / w:.6f} {p[1] / h:.6f} {bw / w:.6f} {bh / h:.6f}\n"
            n += 1
        (yd / f"{files[f - 1].stem}.txt").write_text(txt)
    print(f"{n}/{len(files)} frames labelled at score >= {min_score} -> {out}", flush=True)
    return files, preds


def main() -> None:
    ap = argparse.ArgumentParser(description="Pseudo-label ball positions with the Phase 3b teacher")
    ap.add_argument("--frames_dir", type=Path)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--start_s", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min_score", type=float, default=THR, help="score floor for the YOLO txt labels")
    ap.add_argument("--box_px", type=float, default=12.0)
    ap.add_argument("--check", action="store_true", help="frames_dir is a GSR clip: score against its labels")
    a = ap.parse_args()
    if a.video:
        a.frames_dir = extract(a.video, a.start_s, a.seconds, a.out / "frames")
    files, preds = label_dir(a.frames_dir, a.out, a.device, a.batch, a.min_score, a.box_px)
    if a.check:
        from gsr_io import parse_labels
        lab = a.frames_dir.parent / "Labels-GameState.json"
        _, fmap, balls, *_ = parse_labels(json.loads(lab.read_text()))
        name_to_f = {Path(v).name: k for k, v in fmap.items()}
        gt = {f: [(cx, cy) for cx, cy, *_ in balls.get(name_to_f[files[f - 1].name], [])] for f in preds}
        r = base.score({"clip": preds}, {"clip": gt}, a.min_score)
        print(f"check vs GSR labels: recall {r['recall']:.2%} precision {r['precision']:.2%}", flush=True)


if __name__ == "__main__":
    main()
