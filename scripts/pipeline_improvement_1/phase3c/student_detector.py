"""
student_detector.py -- Phase 3c (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md): the live detector student, YOLO26s,
2 classes (athlete 0, ball 1), 5 input channels: B, G, R of frame t + grey of t-2 and t-1 (past only, so it can
run live; decided 2026-11-19), trained on GSR + the pseudo-labelled SoccerNet windows (build_student_data.py
dataset_all.yaml). The channel stacking is train_teacher_yolo_motion.py's (TEACHER_OFFS=-2,-1).

  train  init runs/yolo26s_athlete_ball/best.pt: every layer but the first conv carries over, the class head
         included (same 2 classes); the first conv takes the init's RGB weights on B, G, R, the grey channels
         start at zero (step 0 = the live detector). imgsz 1280 (the live input is 736x1280). Resumable.
  eval   on select and test, the same scorers as the teachers: ball = top class-1 box per frame (15 px match;
         threshold = highest select recall at precision >= 90%, frozen for test); athletes = class-0 boxes vs GSR
         persons at IoU >= 0.5 (confidence = best select F1, frozen). --baseline scores a 3-channel model (the
         live detector) the same way.

    python scripts/pipeline_improvement_1/phase3c/student_detector.py train --batch 16
    python scripts/pipeline_improvement_1/phase3c/student_detector.py eval --weights runs/student/d1_s1280_c5/weights/best.pt
    python scripts/pipeline_improvement_1/phase3c/student_detector.py eval --baseline --weights runs/yolo26s_athlete_ball/best.pt
"""

from __future__ import annotations

import os

os.environ.setdefault("TEACHER_OFFS", "-2,-1")      # before the motion module is imported (workers too)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase3b"))
import athlete_teacher as at  # noqa: E402
import train_teacher_yolo as base  # noqa: E402
import train_teacher_yolo_motion as tm  # noqa: E402
from gsr_io import SeqReader, discover, parse_labels  # noqa: E402

ROOT = base.ROOT
DATA = ROOT / "data" / "outputs" / "pipeline_improvement_1" / "phase3c" / "student_data"
EVAL = ROOT / "data" / "outputs" / "pipeline_improvement_1" / "phase3c" / "eval"


def head_callback(init: Path):
    """Copy the init's class-head output convs (same classes) after the trainer rebuilt the model."""
    import torch
    from ultralytics import YOLO
    src_head = YOLO(str(init)).model.model[-1]

    def cb(trainer):
        m = tm.model_of(trainer)
        head, n = m.model[-1], 0
        for attr in ("cv3", "one2one_cv3", "cv2", "one2one_cv2"):
            dst, src = getattr(head, attr, None), getattr(src_head, attr, None)
            if dst is None or src is None:
                continue
            for d, s in zip(dst, src):
                if d.state_dict().keys() == s.state_dict().keys() and all(
                        v.shape == s.state_dict()[k].shape for k, v in d.state_dict().items()):
                    with torch.no_grad():
                        d.load_state_dict({k: v.to(d.state_dict()[k]) for k, v in s.state_dict().items()})
                    n += 1
        ema = getattr(trainer, "ema", None)
        if ema is not None:
            ema.ema.load_state_dict(m.state_dict())
        print(f"head: {n} branches copied from {init.name}", flush=True)
    return cb


def train(a) -> None:
    import torch
    from ultralytics import YOLO
    if a.resume:
        YOLO(str(a.project / a.name / "weights" / "last.pt")).train(resume=True)
        return
    yaml_src = (a.data / a.dataset).read_text()
    data_yaml = a.data / a.dataset.replace(".yaml", f"_c{tm.CH}.yaml")
    data_yaml.write_text(yaml_src.rstrip("\n") + f"\nchannels: {tm.CH}\n")
    model = YOLO("yolo26s.yaml").load(str(a.init))
    src_w = YOLO(str(a.init)).model.model[0].conv.weight.detach().float()

    def inflate(trainer):
        w = tm._first_conv(tm.model_of(trainer)).weight
        with torch.no_grad():
            w.zero_()
            w[:, :3] = src_w.flip(1).to(w.device, w.dtype)
        print(f"first conv inflated: {tuple(w.shape)}, grey offsets {tm.OFFS} start at zero", flush=True)

    model.add_callback("on_pretrain_routine_end", inflate)
    model.add_callback("on_pretrain_routine_end", head_callback(a.init))
    model.train(
        data=str(data_yaml), imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=a.batch, workers=a.workers,
        device=a.device, project=str(a.project), name=a.name, exist_ok=True, seed=0, amp=True, cos_lr=True,
        close_mosaic=a.close_mosaic, mosaic=1.0, scale=0.25, translate=0.1, fliplr=0.5, degrees=0.0, mixup=0.0,
        plots=True, fraction=a.fraction, max_det=300, **({"time": a.time_h} if a.time_h else {}))


def run_split(model, refs, a):
    """ball {seq: {frame: (x, y, conf) | None}}, athlete preds [(N,5)], athlete gts [(M,4)], seq per frame."""
    balls, ap, ag, seqs = {}, [], [], []
    for r in refs:
        rd = SeqReader(r)
        try:
            im_dir, files, _, persons, *_ = parse_labels(rd.labels())
        finally:
            rd.close()
        frames = sorted(files)
        cache, bp = {}, {}

        def get(f):
            if f not in cache:
                p = Path(r.container) / im_dir / files[f] if f in files else None
                cache[f] = tm._read(p) if p is not None else None
            return cache[f]

        for s in range(0, len(frames), a.eval_batch):
            fs = frames[s:s + a.eval_batch]
            ims = [get(f) if a.baseline else tm.stack(get(f), [get(f + o) for o in tm.OFFS]) for f in fs]
            for k in [k for k in cache if k < fs[0] - 3]:
                del cache[k]
            for f, rr in zip(fs, model.predict(ims, imgsz=a.imgsz, conf=0.01, iou=0.6, max_det=300, half=True,
                                               device=a.device, verbose=False)):
                b = rr.boxes
                cls, cf = b.cls.cpu().numpy(), b.conf.cpu().numpy()
                xywh = b.xywh.cpu().numpy()
                bi = np.flatnonzero(cls == 1)
                if len(bi):
                    j = bi[np.argmax(cf[bi])]
                    bp[f] = (float(xywh[j, 0]), float(xywh[j, 1]), float(cf[j]))
                else:
                    bp[f] = None
                ai = cls == 0
                ap.append(np.hstack([xywh[ai], cf[ai, None]]))
                ag.append(np.array([p[:4] for p in persons.get(f, [])]).reshape(-1, 4))
                seqs.append(r.name)
        balls[r.name] = bp
        print(f"  {r.split} {r.name}", flush=True)
    return balls, ap, ag, seqs


def evaluate(a) -> None:
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    out = EVAL / ("baseline_" if a.baseline else "student_")
    name = a.weights.parent.parent.name if a.weights.parent.name == "weights" else a.weights.parent.name
    refs = {s: [r for r in discover(a.src) if r.split == s] for s in ("select", "test")}
    if a.limit_seqs:
        refs = {s: v[:a.limit_seqs] for s, v in refs.items()}
    res = {s: run_split(model, v, a) for s, v in refs.items()}
    gt = {s: base._gt(v) for s, v in refs.items()}
    thrs = np.round(np.arange(0.01, 0.96, 0.01), 2)
    curve = [base.score(res["select"][0], gt["select"], float(t)) for t in thrs]
    ok = [c for c in curve if c["precision"] >= 0.90]
    bsel = max(ok, key=lambda c: c["recall"]) if ok else max(curve, key=lambda c: c["f1"])
    no_leak = {r.name for r in refs["test"]} - {base.LEAK_SEQ}
    btest = base.score(res["test"][0], gt["test"], bsel["thr"])
    btest_nl = base.score(res["test"][0], gt["test"], bsel["thr"], seqs=no_leak)
    acurve = [at.score(res["select"][1], res["select"][2], c) for c in at.CONFS]
    asel = max(acurve, key=lambda c: c["f1"])
    _, tp_, tg, ts = res["test"]
    atest = at.score(tp_, tg, asel["conf"])
    keep = [s != base.LEAK_SEQ for s in ts]
    atest_nl = at.score([p for p, k in zip(tp_, keep) if k], [g for g, k in zip(tg, keep) if k], asel["conf"])
    f = lambda s: f"{s['recall']:.2%} | {s['precision']:.2%} | {s['f1']:.2%}"  # noqa: E731
    L = [f"## {'Baseline' if a.baseline else 'Student'} `{a.weights}` (imgsz {a.imgsz}, "
         f"{3 if a.baseline else tm.CH} channels)", "",
         f"Ball (15 px; threshold {bsel['thr']:.2f}, highest select recall at precision >= 90%, frozen):", "",
         "| split | recall | precision | F1 |", "|---|---|---|---|", f"| select | {f(bsel)} |", f"| test | {f(btest)} |",
         f"| test w/o {base.LEAK_SEQ} | {f(btest_nl)} |", "",
         f"Athletes (IoU 0.5; confidence {asel['conf']}, best select F1, frozen):", "",
         "| split | recall | precision | F1 |", "|---|---|---|---|", f"| select | {f(asel)} |", f"| test | {f(atest)} |",
         f"| test w/o {base.LEAK_SEQ} | {f(atest_nl)} |"]
    print("\n".join(L))
    EVAL.mkdir(parents=True, exist_ok=True)
    stem = Path(str(out) + name)
    stem.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps({"ball": {"select": bsel, "test": btest, "test_without_leak": btest_nl},
                                                     "athletes": {"select": asel, "test": atest, "test_without_leak": atest_nl},
                                                     "limit_seqs": a.limit_seqs}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["train", "eval"])
    ap.add_argument("--src", default=str(ROOT / "data" / "gsr"))
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--dataset", default="dataset_all.yaml")
    ap.add_argument("--init", type=Path, default=ROOT / "runs" / "yolo26s_athlete_ball" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--close_mosaic", type=int, default=2)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--time_h", type=float, default=0.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", type=Path, default=ROOT / "runs" / "student")
    ap.add_argument("--name", default="d1_s1280_c5")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--baseline", action="store_true", help="eval: 3-channel model (no stacking)")
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--limit_seqs", type=int, default=0, help="eval smoke test: first N sequences per split")
    a = ap.parse_args()
    print(f"{tm.CH} input channels, neighbour offsets {tm.OFFS}", flush=True)
    train(a) if a.step == "train" else evaluate(a)


if __name__ == "__main__":
    main()
