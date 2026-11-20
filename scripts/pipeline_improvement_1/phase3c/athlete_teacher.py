"""
athlete_teacher.py -- Phase 3c (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md): the offline athlete teacher and its
labels on the pseudo-labelled SoccerNet windows.

  train   YOLO26s (the ball teacher's size) at imgsz 1920, 3 channels, fine-tuned from the live detector
          runs/yolo26s_athlete_ball/best.pt, athlete class only (1 class; build_athlete_data.py) on the GSR mirror
          of build_student_data.py (train + pool -> train, select -> val). Resumable (--resume).
  choose  athlete boxes on every `select` frame vs GSR person boxes (IoU >= 0.5, greedy by confidence):
          precision / recall / F1 per confidence; the confidence with the best F1 is frozen; `test` is scored
          once at it (and without SNGS-050). Written to <eval>/athlete_teacher.{json,md}.
  label   athlete boxes (conf >= --store_conf, class athlete only; the ball comes from the Phase 3b ball
          teacher) on every frame of every pseudo window: <window>/athletes.json {frame file: [[cx,cy,w,h,conf]]}.
          Resumable per window.

    python scripts/pipeline_improvement_1/phase3c/athlete_teacher.py train --batch 8
    python scripts/pipeline_improvement_1/phase3c/athlete_teacher.py choose
    python scripts/pipeline_improvement_1/phase3c/athlete_teacher.py label
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "ball_temporal"))
from gsr_io import SeqReader, discover, parse_labels  # noqa: E402

DATA = ROOT / "data" / "outputs" / "pipeline_improvement_1" / "phase3c" / "athlete_data"
EVAL = ROOT / "data" / "outputs" / "pipeline_improvement_1" / "phase3c" / "eval"
PSEUDO = ROOT / "data" / "pseudo_labels" / "phase3b_teacher"
LEAK_SEQ = "SNGS-050"
CONFS = np.round(np.arange(0.10, 0.91, 0.05), 2)


def train(a) -> None:
    from ultralytics import YOLO
    if a.resume:
        YOLO(str(a.project / a.name / "weights" / "last.pt")).train(resume=True)
        return
    YOLO(str(a.init)).train(
        data=str(a.data / "dataset.yaml"), imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=a.batch,
        workers=a.workers, device=a.device, project=str(a.project), name=a.name, exist_ok=True, seed=0, amp=True,
        cos_lr=True, close_mosaic=a.close_mosaic, mosaic=1.0, scale=0.25, translate=0.1, fliplr=0.5, mixup=0.0,
        plots=True, fraction=a.fraction, max_det=300)


def predict(model, paths, a):
    """[(N, 5) array of cx, cy, w, h, conf] per path, athlete class only."""
    out = []
    for s in range(0, len(paths), a.eval_batch):
        for r in model.predict(paths[s:s + a.eval_batch], imgsz=a.imgsz, conf=a.store_conf, iou=0.6, max_det=300,
                               half=True, device=a.device, verbose=False, classes=[0]):
            b = r.boxes
            out.append(np.hstack([b.xywh.cpu().numpy(), b.conf.cpu().numpy()[:, None]]) if len(b) else np.zeros((0, 5)))
    return out


def iou(a, b):
    a1, a2 = a[:, None, :2] - a[:, None, 2:4] / 2, a[:, None, :2] + a[:, None, 2:4] / 2
    b1, b2 = b[None, :, :2] - b[None, :, 2:4] / 2, b[None, :, :2] + b[None, :, 2:4] / 2
    inter = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None).prod(-1)
    return inter / (a[:, None, 2:4].prod(-1) + b[None, :, 2:4].prod(-1) - inter + 1e-9)


def score(preds, gts, conf):
    tp = fp = n = 0
    for p, g in zip(preds, gts):
        p = p[p[:, 4] >= conf]
        p = p[np.argsort(-p[:, 4])]
        n += len(g)
        if not len(p):
            continue
        if not len(g):
            fp += len(p)
            continue
        m = iou(p, g)
        used = np.zeros(len(g), bool)
        for i in range(len(p)):
            j = np.where(~used & (m[i] >= 0.5))[0]
            if len(j):
                used[j[np.argmax(m[i, j])]] = True
                tp += 1
            else:
                fp += 1
    r, pr = tp / max(n, 1), tp / max(tp + fp, 1)
    return {"conf": float(conf), "recall": r, "precision": pr, "f1": 2 * r * pr / max(r + pr, 1e-9), "gt": n}


def split_preds(model, split, a):
    preds, gts, seqs = [], [], []
    for r in [r for r in discover(a.src) if r.split == split]:
        rd = SeqReader(r)
        try:
            im_dir, files, _, persons, *_ = parse_labels(rd.labels())
        finally:
            rd.close()
        fr = sorted(files)
        preds += predict(model, [str(Path(r.container) / im_dir / files[f]) for f in fr], a)
        gts += [np.array([p[:4] for p in persons.get(f, [])]).reshape(-1, 4) for f in fr]
        seqs += [r.name] * len(fr)
        print(f"  {split} {r.name}", flush=True)
    return preds, gts, seqs


def choose(a) -> None:
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    EVAL.mkdir(parents=True, exist_ok=True)
    sp, sg, _ = split_preds(model, "select", a)
    curve = [score(sp, sg, c) for c in CONFS]
    best = max(curve, key=lambda s: s["f1"])
    tp_, tg, ts = split_preds(model, "test", a)
    t_all = score(tp_, tg, best["conf"])
    keep = [s != LEAK_SEQ for s in ts]
    t_nl = score([p for p, k in zip(tp_, keep) if k], [g for g, k in zip(tg, keep) if k], best["conf"])
    f = lambda s: f"{s['recall']:.2%} | {s['precision']:.2%} | {s['f1']:.2%}"  # noqa: E731
    L = [f"Athlete teacher {a.weights}, imgsz {a.imgsz}; conf chosen on select (best F1): {best['conf']}", "",
         "| split | recall | precision | F1 |", "|---|---|---|---|", f"| select | {f(best)} |", f"| test | {f(t_all)} |",
         f"| test w/o {LEAK_SEQ} | {f(t_nl)} |"]
    print("\n".join(L))
    stem = EVAL / f"athlete_teacher_{a.weights.parent.parent.name}"
    stem.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps({"conf": best["conf"], "select_curve": curve, "test": t_all,
                                                     "test_without_leak": t_nl}, indent=1))


def label(a) -> None:
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    wins = sorted(p.parent for p in a.pseudo.glob("*/*/labels.json"))
    t0, n = time.time(), 0
    for i, w in enumerate(wins, 1):
        out = w / "athletes.json"
        if out.exists():
            continue
        files = sorted((w / "frames").glob("*.jpg"))
        boxes = predict(model, [str(p) for p in files], a)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({p.name: np.round(b, 4).tolist() for p, b in zip(files, boxes)}))
        tmp.replace(out)
        n += len(files)
        print(f"[{i}/{len(wins)}] {w.parent.name[:50]}/{w.name}: {n / (time.time() - t0):.1f} fps", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["train", "choose", "label"])
    ap.add_argument("--src", default=str(ROOT / "data" / "gsr"))
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--init", type=Path, default=ROOT / "runs" / "yolo26s_athlete_ball" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--close_mosaic", type=int, default=5)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", type=Path, default=ROOT / "runs" / "teacher")
    ap.add_argument("--name", default="a1_s1920_ft")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "teacher" / "a1_s1920_ft" / "weights" / "best.pt")
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--store_conf", type=float, default=0.10)
    ap.add_argument("--pseudo", type=Path, default=PSEUDO)
    a = ap.parse_args()
    {"train": train, "choose": choose, "label": label}[a.step](a)


if __name__ == "__main__":
    main()
