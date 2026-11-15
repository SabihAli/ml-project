"""
train_teacher_yolo_motion.py -- Phase 3b teacher T2 with motion: YOLO26 on 5 channels per frame,
BGR(t) + grey of neighbours (default t-1, t+1; --dts 1,2 for t-2..t+2, 7 channels) (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md, Phase 3b). Offline, so the
teacher may look one frame ahead, which the live system cannot.

Uses the mirror `train_teacher_yolo.py build` makes (every frame is linked there, so neighbours exist even
when a frame is not listed). No extra files: the channels are stacked when a frame is read, by replacing
Ultralytics' imread in its dataset module (top level of this script, so Windows dataloader workers, which
re-import it, get the same reader). At a clip end the missing neighbour is the frame itself.

Channel order is B, G, R, grey-, grey+ in training and in eval; Ultralytics only reverses BGR->RGB for
3-channel input, so nothing reorders it. The first conv is new (5 inputs): the RGB slice takes the --init
weights (reversed to BGR), the two grey channels start at zero, so step 0 is the single-frame detector.

    python scripts/pipeline_improvement_1/phase3b/train_teacher_yolo_motion.py train --init runs/yolo26s_athlete_ball/best.pt
    python scripts/pipeline_improvement_1/phase3b/train_teacher_yolo_motion.py eval --weights runs/teacher/t2m_s1920_ft/weights/best.pt
    python scripts/pipeline_improvement_1/phase3b/train_teacher_yolo_motion.py dump --weights runs/teacher/t2m_s1920_ft/weights/best.pt

`dump` writes the top-k candidates per frame (and GT) on select and test for trajectory_pass.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_teacher_yolo as base  # noqa: E402
from gsr_io import SeqReader, discover, parse_labels  # noqa: E402

# Neighbour offsets in frames, e.g. "1" -> t-1, t+1 (5 channels) or "1,2" -> t-2, t-1, t+1, t+2 (7).
# An environment variable, not argv, so spawned dataloader workers (which re-import this file) agree.
DTS = tuple(sorted(int(d) for d in os.environ.get("TEACHER_DTS", "1").split(",")))
OFFS = tuple(-d for d in reversed(DTS)) + DTS     # channel order after B, G, R: t-max .. t-1, t+1 .. t+max
CH = 3 + len(OFFS)
_STEM = re.compile(r"^(.*)_(\d{6})$")


def stack(cur_bgr: np.ndarray, neighbours) -> np.ndarray:
    """B, G, R of t, then grey of each neighbour in OFFS order (None -> the frame itself)."""
    g = lambda im: cv2.cvtColor(im if im is not None else cur_bgr, cv2.COLOR_BGR2GRAY)  # noqa: E731
    return np.dstack([cur_bgr] + [g(n) for n in neighbours])


def _read(p: Path):
    return cv2.imread(str(p), cv2.IMREAD_COLOR) if p.exists() else None


def read_stacked(path) -> np.ndarray | None:
    p = Path(path)
    cur = _read(p)
    m = _STEM.match(p.stem)
    if cur is None or m is None:
        return cur
    seq, f = m.group(1), int(m.group(2))
    nb = lambda k: _read(p.with_name(f"{seq}_{k:06d}{p.suffix}"))  # noqa: E731
    return stack(cur, [nb(f + o) for o in OFFS])


def _patch_imread() -> None:
    import ultralytics.data.base as ub

    orig = ub.imread

    def imread(filename, flags=cv2.IMREAD_COLOR):
        if flags == cv2.IMREAD_UNCHANGED and str(filename).lower().endswith(".jpg"):
            return read_stacked(filename)
        return orig(filename, flags)

    ub.imread = imread


_patch_imread()                                   # also runs in spawned dataloader workers


def dataset_yaml(data: Path) -> Path:
    src = (data / "dataset.yaml").read_text()
    out = data / f"dataset_c{CH}.yaml"
    out.write_text(src.rstrip("\n") + f"\nchannels: {CH}\n")
    return out


def model_of(trainer):
    return trainer.model.module if hasattr(trainer.model, "module") else trainer.model


def _first_conv(model):
    return model.model[0].conv


def train(a) -> None:
    import torch
    from ultralytics import YOLO
    run_dir = a.project / a.name
    if a.resume:
        YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True)
        return
    init = a.init or Path(f"{a.arch}.pt")
    model = YOLO(f"{a.arch}{'-p2' if a.p2 else ''}.yaml")
    model = model.load(str(init))                 # everything but the first conv (shape) and class head
    src_w = YOLO(str(init)).model.model[0].conv.weight.detach().float()  # (c_out, 3, k, k), RGB order

    def inflate(trainer):
        conv = _first_conv(model_of(trainer))
        w = conv.weight
        if w.shape[1] != CH or w.shape[0] != src_w.shape[0]:
            raise SystemExit(f"first conv {tuple(w.shape)} does not match init {tuple(src_w.shape)}")
        with torch.no_grad():
            w.zero_()
            w[:, :3] = src_w.flip(1).to(w.device, w.dtype)   # RGB weights onto B, G, R
        print(f"first conv inflated: {tuple(w.shape)}, grey channels zero, offsets {OFFS}", flush=True)
        ema = getattr(trainer, "ema", None)
        if ema is not None:                       # EMA was copied before this callback
            ema.ema.load_state_dict(model_of(trainer).state_dict())

    model.add_callback("on_pretrain_routine_end", inflate)
    if (cb := base.ball_row_callback(init, a.src_class)) is not None:
        model.add_callback("on_pretrain_routine_end", cb)
    model.train(
        data=str(dataset_yaml(a.data)), imgsz=a.imgsz, epochs=a.epochs, patience=a.patience,
        batch=a.batch, workers=a.workers, device=a.device, project=str(a.project), name=a.name,
        exist_ok=True, seed=a.seed, amp=True, cos_lr=True, close_mosaic=a.close_mosaic,
        mosaic=1.0, scale=a.scale, translate=0.1, fliplr=0.5, degrees=0.0, mixup=0.0,
        single_cls=True, plots=True, fraction=a.fraction, max_det=a.max_det,
        **({"time": a.time_h} if a.time_h else {}),
    )


def _top_ball_per_frame(model, refs, a) -> dict:
    """{seq: {frame: (x, y, conf) or None}} -- the top candidate of _candidates_per_frame."""
    cands = _candidates_per_frame(model, refs, a, topk=1, conf=0.01)
    return {seq: {f: (tuple(c[0]) if c else None) for f, c in fr.items()} for seq, fr in cands.items()}


def _candidates_per_frame(model, refs, a, topk: int, conf: float) -> dict:
    """{seq: {frame: [(x, y, conf), ...]}}, best first, at most topk per frame."""
    out = {}
    for i, r in enumerate(refs, 1):
        rd = SeqReader(r)
        try:
            im_dir, files, *_ = parse_labels(rd.labels())
        finally:
            rd.close()
        frames = sorted(files)
        path = lambda f: Path(r.container) / im_dir / files[f] if f in files else None  # noqa: E731
        cache, preds, t0 = {}, {}, time.time()

        def get(f):
            if f not in cache:
                p = path(f)
                cache[f] = _read(p) if p is not None else None
            return cache[f]

        for s in range(0, len(frames), a.eval_batch):
            fs = frames[s:s + a.eval_batch]
            ims = [stack(get(f), [get(f + o) for o in OFFS]) for f in fs]
            for k in [k for k in cache if k < fs[0] - max(DTS)]:
                del cache[k]
            res = model.predict(ims, imgsz=a.imgsz, conf=conf, iou=0.5, max_det=a.max_det,
                                half=True, device=a.device, verbose=False)
            for f, rr in zip(fs, res):
                b = rr.boxes
                if b is None or not len(b):
                    preds[f] = []
                    continue
                order = b.conf.argsort(descending=True)[:topk].tolist()
                xy, cf = b.xyxy.tolist(), b.conf.tolist()
                preds[f] = [((xy[j][0] + xy[j][2]) / 2, (xy[j][1] + xy[j][3]) / 2, cf[j]) for j in order]
        out[r.name] = preds
        print(f"  [{i}/{len(refs)}] {r.name}: {len(frames) / (time.time() - t0):.1f} fps", flush=True)
    return out


def dump(a) -> None:
    """Top-k candidates per frame + GT on select and test, for trajectory_pass.py (one JSON per split)."""
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    out = base.DATA_OUT / "candidates" / a.weights.parent.parent.name
    out.mkdir(parents=True, exist_ok=True)
    for split in ("select", "test"):
        refs = [r for r in discover(a.src) if r.split == split]
        if a.limit_seqs:
            refs = refs[:a.limit_seqs]
        print(f"{split}: {len(refs)} sequences", flush=True)
        cands, gt = _candidates_per_frame(model, refs, a, a.topk, a.dump_conf), base._gt(refs)
        doc = {"weights": str(a.weights), "topk": a.topk, "conf": a.dump_conf,
               "seqs": {s: {"cands": {str(f): c for f, c in cands[s].items()},
                            "gt": {str(f): g for f, g in gt[s].items()}} for s in cands}}
        (out / f"{split}.json").write_text(json.dumps(doc))
        print(f"  -> {out / f'{split}.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("step", choices=["train", "eval", "dump"])
    ap.add_argument("--src", type=Path, default=base.ROOT / "data" / "gsr")
    ap.add_argument("--data", type=Path, default=base.DATA_OUT / "data", help="mirror from train_teacher_yolo build")
    ap.add_argument("--arch", default="yolo26s")
    ap.add_argument("--init", type=Path, default=base.ROOT / "runs" / "yolo26s_athlete_ball" / "best.pt")
    ap.add_argument("--p2", action="store_true", help="add the stride-4 head (default off, as run A)")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--close_mosaic", type=int, default=5)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--max_det", type=int, default=50)
    ap.add_argument("--time_h", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", type=Path, default=base.ROOT / "runs" / "teacher")
    ap.add_argument("--name", default="t2m_s1920_ft")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--src_class", default="ball",
                    help="init class whose head row starts the ball head ('sports ball' for a COCO init)")
    ap.add_argument("--dts", default=None, help="neighbour offsets, e.g. 1,2 for 7 channels (sets TEACHER_DTS)")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--min_precision", type=float, default=0.90)
    ap.add_argument("--limit_seqs", type=int, default=0)
    ap.add_argument("--eval_out", type=Path, default=None)
    ap.add_argument("--topk", type=int, default=5, help="dump: candidates kept per frame")
    ap.add_argument("--dump_conf", type=float, default=0.001, help="dump: lowest candidate score kept")
    a = ap.parse_args()
    if a.dts and a.dts != os.environ.get("TEACHER_DTS", "1"):
        import subprocess                           # re-run with it set (the module-level DTS reads it);
        env = {**os.environ, "TEACHER_DTS": a.dts}   # dataloader workers inherit it too
        raise SystemExit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)
    print(f"{CH} input channels, neighbour offsets {OFFS}", flush=True)
    if a.step == "train":
        train(a)
    elif a.step == "dump":
        if a.weights is None:
            raise SystemExit("dump needs --weights")
        dump(a)
    else:
        if a.weights is None:
            raise SystemExit("eval needs --weights")
        a.eval_out = a.eval_out or base.DATA_OUT / "eval" / a.weights.parent.parent.name
        base._top_ball_per_frame = _top_ball_per_frame   # base.evaluate scores whatever this returns
        base.evaluate(a)


if __name__ == "__main__":
    main()
