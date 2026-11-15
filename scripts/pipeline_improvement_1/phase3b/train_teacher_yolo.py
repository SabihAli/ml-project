"""
train_teacher_yolo.py -- Phase 3b teacher candidate T2 (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md, Phase 3b):
YOLO26 with a stride-4 P2 head, large backbone, ball only, full resolution (imgsz 1920).

Splits are gsr_io's, as every other experiment in this plan: train on `train` + `pool`, validate on `select`,
`test` held out (scored once, also without SNGS-050). data/gsr is never modified: `build` makes a mirror of
hard links (same volume) laid out as Ultralytics expects (images/ <-> labels/).

Steps:
  build  mirror + ball-only YOLO labels (class 0; frames without a ball get an empty label file, they are
         the negatives) + dataset.yaml. Train uses every --train_stride-th frame (25 fps neighbours are
         near-duplicates), validation every --val_stride-th.
  train  YOLO(<arch>-p2.yaml) with the COCO <arch>.pt weights loaded where layers match (backbone, neck);
         the P2 branch and the 1-class head start cold. Ultralytics downloads <arch>.pt from its GitHub
         releases on first use. Mosaic keeps frames at native scale at imgsz 1920; --scale is kept small
         (a 6 px ball must not be shrunk away). Resumable (--resume).
  eval   per frame, the top ball box (conf >= 0.01) on every frame of `select` and `test`, at imgsz 1920,
         scored as scripts/ball_eval/eval_ball_temporal.py scores the live system: 15 px centre match,
         recall = hit frames / ball-labelled frames, precision = hits / outputs on ball-labelled frames
         (outputs on unlabelled frames counted apart), plus F1 and recall @5 px. The threshold is chosen
         on `select` (highest recall at precision >= --min_precision) and frozen for `test`. Phase 3b gate:
         recall >= 88% at precision >= 90% on `test`.

    python scripts/pipeline_improvement_1/phase3b/train_teacher_yolo.py build
    python scripts/pipeline_improvement_1/phase3b/train_teacher_yolo.py train --no_p2 --batch 8
    python scripts/pipeline_improvement_1/phase3b/train_teacher_yolo.py eval --weights runs/teacher/t2_yolo26l_p2/weights/best.pt

Outputs: data/outputs/pipeline_improvement_1/phase3b/teacher_yolo/data (mirror), runs/teacher/<name>/
(Ultralytics run), <eval out>/teacher_eval.json + teacher_eval.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "ball_temporal"))

from gsr_io import SeqReader, discover, parse_labels  # noqa: E402

DATA_OUT = ROOT / "data" / "outputs" / "pipeline_improvement_1" / "phase3b" / "teacher_yolo"
MATCH_PX, FINE_PX = 15.0, 5.0
LEAK_SEQ = "SNGS-050"            # overlaps a pool clip (Phase 0); test is also reported without it
SPLIT_ROLE = {"train": "train", "pool": "train", "select": "val", "test": "test"}


# -- build --------------------------------------------------------------------------------

def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:                     # other volume / no hard links: copy
        import shutil
        shutil.copy2(src, dst)


def build(a) -> Path:
    out = a.data
    refs = [r for r in discover(a.src) if r.split in SPLIT_ROLE]
    if any(r.kind != "dir" for r in refs):
        raise SystemExit("build needs extracted sequence folders (data/gsr/<split>/SNGS-xxx/img1)")
    lists = {"train": [], "val": [], "test": []}
    counts = {k: [0, 0] for k in lists}                 # frames, frames with a ball
    for i, r in enumerate(refs, 1):
        role = SPLIT_ROLE[r.split]
        rd = SeqReader(r)
        try:
            im_dir, files, balls, _, _, (W, H) = parse_labels(rd.labels())
        finally:
            rd.close()
        img_dir, lab_dir = out / "images" / role, out / "labels" / role
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        stride = a.train_stride if role == "train" else a.val_stride if role == "val" else 1
        for k, f in enumerate(sorted(files)):
            stem = f"{r.name}_{f:06d}"
            _link(Path(r.container) / im_dir / files[f], img_dir / f"{stem}.jpg")
            rows = [f"0 {cx / W:.6f} {cy / H:.6f} {max(w, 1.0) / W:.6f} {max(h, 1.0) / H:.6f}"
                    for cx, cy, w, h, _ in balls.get(f, [])]
            (lab_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            if k % stride == 0:
                lists[role].append(str((img_dir / f"{stem}.jpg").resolve()))
                counts[role][0] += 1
                counts[role][1] += bool(rows)
        print(f"  [{i}/{len(refs)}] {r.name} ({r.split} -> {role})", flush=True)
    for role, files in lists.items():
        (out / f"{role}.txt").write_text("\n".join(files) + "\n")
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(f"path: {out.resolve().as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
                         "nc: 1\nnames:\n  0: ball\n")
    for role, (n, nb) in counts.items():
        print(f"{role}: {n} frames listed, {nb} with a ball")
    print(f"dataset -> {yaml_path}")
    return yaml_path


# -- train --------------------------------------------------------------------------------

def ball_row_callback(init: Path, src_class: str = "ball"):
    """Trainer callback: start the 1-class head from the init model's `ball` row.

    Ultralytics builds the trainer's model before it knows the dataset's class names, so a 2-class init
    (athlete, ball) is not remapped by name and the head starts as the athlete classifier: every athlete
    is then a confident false ball (smoke test: cls loss ~64-190, 0 hits). Heads are matched by stride
    from the coarsest, so a P2 model takes the init's P3-P5 rows on its P3-P5 heads.
    """
    import torch
    from ultralytics import YOLO
    src_model = YOLO(str(init)).model
    names = src_model.names
    if src_class not in names.values():
        return None
    row = [k for k, v in names.items() if v == src_class][0]
    src_head = src_model.model[-1]

    def cb(trainer):
        m = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        head, n = m.model[-1], 0
        for attr in ("cv3", "one2one_cv3"):
            dst, src = getattr(head, attr, None), getattr(src_head, attr, None)
            if dst is None or src is None:
                continue
            for d, s in zip(reversed(list(dst)), reversed(list(src))):
                dc, sc = d[-1], s[-1]
                if dc.out_channels == 1 and dc.in_channels == sc.in_channels:
                    with torch.no_grad():
                        dc.weight.copy_(sc.weight[row:row + 1].to(dc.weight))
                        dc.bias.copy_(sc.bias[row:row + 1].to(dc.bias))
                    n += 1
        ema = getattr(trainer, "ema", None)
        if ema is not None:                       # EMA was copied before this callback
            ema.ema.load_state_dict(m.state_dict())
        print(f"class head: {n} output convs set from the init's '{src_class}' row", flush=True)
    return cb


def train(a) -> None:
    from ultralytics import YOLO
    run_dir = a.project / a.name
    if a.resume:
        model = YOLO(str(run_dir / "weights" / "last.pt"))
        model.train(resume=True)
        return
    model = YOLO(f"{a.arch}{'' if a.no_p2 else '-p2'}.yaml")
    if not a.no_pretrained:                             # COCO <arch>.pt, or --init (e.g. our football detector)
        model = model.load(str(a.init) if a.init else f"{a.arch}.pt")   # wherever the layers match
        if a.init and (cb := ball_row_callback(a.init)) is not None:
            model.add_callback("on_pretrain_routine_end", cb)
    model.train(
        data=str(a.data / "dataset.yaml"), imgsz=a.imgsz, epochs=a.epochs, patience=a.patience,
        batch=a.batch, workers=a.workers, device=a.device, project=str(a.project), name=a.name,
        exist_ok=True, seed=a.seed, amp=True, cos_lr=True, close_mosaic=a.close_mosaic,
        mosaic=1.0, scale=a.scale, translate=0.1, fliplr=0.5, degrees=0.0, mixup=0.0,
        single_cls=True, plots=True, fraction=a.fraction, max_det=a.max_det,
        **({"time": a.time_h} if a.time_h else {}),
    )


# -- eval ---------------------------------------------------------------------------------

def _top_ball_per_frame(model, refs, a) -> dict:
    """{seq: {frame: (x, y, conf) or None}} -- the highest-confidence box per frame."""
    out = {}
    for i, r in enumerate(refs, 1):
        rd = SeqReader(r)
        try:
            im_dir, files, *_ = parse_labels(rd.labels())
        finally:
            rd.close()
        paths = [str(Path(r.container) / im_dir / files[f]) for f in sorted(files)]
        frames = sorted(files)
        preds, t0 = {}, time.time()
        for s in range(0, len(paths), a.eval_batch):
            res = model.predict(paths[s:s + a.eval_batch], imgsz=a.imgsz, conf=0.01, iou=0.5, max_det=a.max_det,
                                half=True, device=a.device, verbose=False)
            for f, rr in zip(frames[s:s + a.eval_batch], res):
                b = rr.boxes
                if b is None or not len(b):
                    preds[f] = None
                    continue
                j = int(b.conf.argmax())
                x1, y1, x2, y2 = b.xyxy[j].tolist()
                preds[f] = ((x1 + x2) / 2, (y1 + y2) / 2, float(b.conf[j]))
        out[r.name] = preds
        print(f"  [{i}/{len(refs)}] {r.name}: {len(paths) / (time.time() - t0):.1f} fps", flush=True)
    return out


def _gt(refs) -> dict:
    gt = {}
    for r in refs:
        rd = SeqReader(r)
        try:
            _, files, balls, *_ = parse_labels(rd.labels())
        finally:
            rd.close()
        gt[r.name] = {f: [(cx, cy) for cx, cy, *_ in balls.get(f, [])] for f in files}
    return gt


def score(preds: dict, gt: dict, thr: float, seqs=None) -> dict:
    n_gt = hit = hit5 = out_gt = out_nogt = 0
    for seq, frames in gt.items():
        if seqs is not None and seq not in seqs:
            continue
        for f, balls in frames.items():
            p = preds[seq].get(f)
            p = p if p is not None and p[2] >= thr else None
            if balls:
                n_gt += 1
                if p is not None:
                    out_gt += 1
                    d = min(np.hypot(p[0] - bx, p[1] - by) for bx, by in balls)
                    hit += int(d <= MATCH_PX)
                    hit5 += int(d <= FINE_PX)
            elif p is not None:
                out_nogt += 1
    rec, prec = hit / max(n_gt, 1), hit / max(out_gt, 1)
    return {"thr": thr, "gt_frames": n_gt, "recall": rec, "precision": prec,
            "f1": 2 * rec * prec / max(rec + prec, 1e-9), "recall_5px": hit5 / max(n_gt, 1),
            "outputs_unlabelled": out_nogt}


def evaluate(a) -> None:
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    a.eval_out.mkdir(parents=True, exist_ok=True)
    refs = {s: [r for r in discover(a.src) if r.split == s] for s in ("select", "test")}
    if a.limit_seqs:                                   # smoke tests only: not a result
        refs = {s: rs[:a.limit_seqs] for s, rs in refs.items()}
    res = {}
    for split, rs in refs.items():
        print(f"{split}: {len(rs)} sequences", flush=True)
        res[split] = (_top_ball_per_frame(model, rs, a), _gt(rs))
    thrs = np.round(np.arange(0.01, 0.96, 0.01), 2)
    sel_curve = [score(*res["select"], float(t)) for t in thrs]
    ok = [s for s in sel_curve if s["precision"] >= a.min_precision]
    chosen = max(ok, key=lambda s: s["recall"]) if ok else max(sel_curve, key=lambda s: s["f1"])
    thr = chosen["thr"]
    test_seqs = {r.name for r in refs["test"]}
    test = score(*res["test"], thr)
    test_wo = score(*res["test"], thr, seqs=test_seqs - {LEAK_SEQ})
    best_f1_test = max((score(*res["test"], float(t)) for t in thrs), key=lambda s: s["f1"])
    gate = bool(test["recall"] >= 0.88 and test["precision"] >= 0.90)
    report = {"weights": str(a.weights), "imgsz": a.imgsz, "min_precision": a.min_precision,
              "threshold_rule": "highest select recall at precision >= min_precision" if ok else
              "no threshold reaches min_precision on select: best select F1",
              "select": chosen, "test": test, "test_without_" + LEAK_SEQ: test_wo,
              "test_best_f1_oracle": best_f1_test, "gate_pass": gate, "select_curve": sel_curve}
    (a.eval_out / "teacher_eval.json").write_text(json.dumps(report, indent=1))
    f = lambda s: (f"{s['recall']:.2%} | {s['precision']:.2%} | {s['f1']:.2%} | {s['recall_5px']:.2%} | "  # noqa: E731
                   f"{s['outputs_unlabelled']}")
    L = ["# Teacher T2 evaluation", "", f"`{a.weights}`, imgsz {a.imgsz}. Threshold **{thr:.2f}** ({report['threshold_rule']}, "
         f"precision floor {a.min_precision:.0%}), frozen for test.", "",
         "| split | recall @15 px | precision | F1 | recall @5 px | outputs on unlabelled frames |",
         "|---|---|---|---|---|---|",
         f"| select ({len(refs['select'])} seq) | {f(chosen)} |",
         f"| **test ({len(refs['test'])} seq)** | {f(test)} |",
         f"| test without {LEAK_SEQ} | {f(test_wo)} |",
         f"| test, best-F1 threshold {best_f1_test['thr']:.2f} (oracle, not a result) | {f(best_f1_test)} |", "",
         "Live system (Phase 1): test 76.29% / 90.21%. Gate (Phase 3b): test recall >= 88% at precision >= 90%: "
         f"**{'PASS' if gate else 'not met'}**."]
    (a.eval_out / "teacher_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "train", "eval"])
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "gsr")
    ap.add_argument("--data", type=Path, default=DATA_OUT / "data")
    ap.add_argument("--train_stride", type=int, default=3)
    ap.add_argument("--val_stride", type=int, default=10)
    ap.add_argument("--arch", default="yolo26s", help="yolo26s / m / l / x; the -p2 variant is built from it")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--scale", type=float, default=0.25, help="Ultralytics scale aug (+-); small keeps the ball visible")
    ap.add_argument("--close_mosaic", type=int, default=5)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the train list (smoke tests)")
    ap.add_argument("--max_det", type=int, default=50)
    ap.add_argument("--time_h", type=float, default=0.0, help="Ultralytics time budget in hours (0: off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--init", type=Path, default=ROOT / "runs" / "yolo26s_athlete_ball" / "best.pt", help="weights to start from instead of COCO <arch>.pt, e.g. "
                    "runs/yolo26s_athlete_ball/best.pt (must match --arch)")
    ap.add_argument("--no_p2", action="store_true", help="plain <arch>.yaml, no stride-4 head")
    ap.add_argument("--project", type=Path, default=ROOT / "runs" / "teacher")
    ap.add_argument("--name", default="t2_s1920_ft")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--min_precision", type=float, default=0.90)
    ap.add_argument("--limit_seqs", type=int, default=0, help="eval: first N sequences per split (smoke test)")
    ap.add_argument("--eval_out", type=Path, default=DATA_OUT / "eval")
    a = ap.parse_args()
    if a.step == "build":
        build(a)
    elif a.step == "train":
        train(a)
    else:
        if not a.weights:
            raise SystemExit("eval needs --weights")
        evaluate(a)


if __name__ == "__main__":
    main()
