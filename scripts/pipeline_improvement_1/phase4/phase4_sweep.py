"""
phase4_sweep.py -- Phase 4 of docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md: the robust ball tracker
(gsr_pipeline/ball_tracker_robust.py) driving main + c5c, against the Phase 1 system
(plausibility gate + threshold 0.45).

Per frame of a sequence (the runtime order):
  1. tracker.predict(t, camera motion t-1 -> t, athlete boxes of t)
  2. main candidates: cached balls at conf >= main_floor; tracker.match()
  3. no match and the tracker has a search position: c5c on a 640 window at the PREDICTION
     (memoised per window); a peak >= thr becomes a specialist candidate
  4. tracker.update(candidates) -> main | specialist | predicted | nothing
Scored two ways: "measured" (main + specialist outputs only, comparable to Phase 1) and
"+predicted" (coasted / occluded positions also output). Conditions, from the labels (eval only):
  occluded   GT ball centre inside a GT person box
  abrupt     |p(t+1) - 2 p(t) + p(t-1)| > --abrupt_px (kick, deflection, bounce)
  post_cut   within 25 frames after a frame with no camera-motion estimate (cut, close-up)

Search on --search_split (select): coordinate descent over SPACE from the defaults, --passes
passes; objective = measured recall with precision >= Phase-1 precision on that split - --prec_drop.
Then frozen on test: Phase 1 baseline, the tracker (measured and +predicted), and ablations
(no occlusion model, no camera motion, one motion model). Kill criterion as Phase 1.

Outputs (--out): report_phase4.md, phase4.json, search.csv, phase4_log.txt,
frames_tracker/frames.jsonl (measured outputs, diagnose format, for phase0_report.py).

    python scripts/pipeline_improvement_1/phase4/phase4_sweep.py --src data/datasets/SoccerNetGS \
        --cache <main cache with --athletes> --motion <camera_motion_cache out> \
        --ckpt runs/ball_temporal/c5c/best.pt --out data/outputs/pipeline_improvement_1/phase4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
for p in (ROOT / "gsr_pipeline", ROOT / "scripts" / "ball_temporal", ROOT / "scripts" / "ball_eval",
          ROOT / "scripts" / "pipeline_improvement_1" / "phase1"):
    sys.path.insert(0, str(p))
from ball_gate import GateParams  # noqa: E402
from ball_tracker_robust import RobustBallTracker, RobustTrackerConfig  # noqa: E402
from eval_ball_temporal import SeqData, Specialist, clamp_window, load_model  # noqa: E402
from gsr_io import discover, parse_labels  # noqa: E402
import phase1_sweep  # noqa: E402

MATCH = 15.0
TOLS = (5.0, 10.0, 15.0)
PHASE1_GATE = GateParams(enabled=True, r0=30.0, speed=50.0, conf_hi=0.6, max_rejects=15)
PHASE1_THR = 0.45
CONDS = ("occluded", "abrupt", "post_cut", "other")
# knob -> candidate values; "thr" and "main_floor" belong to the replay, the rest to RobustTrackerConfig
SPACE = {
    "thr": [0.35, 0.45, 0.55],
    "main_floor": [0.10, 0.15, 0.25],
    "sigma_man": [6.0, 12.0, 20.0],
    "sigma_calm": [0.8, 1.5, 3.0],
    "gate_chi2": [9.2, 13.8, 20.0],
    "min_gate_px": [15.0, 25.0, 40.0],
    "lam": [0.25, 0.5, 1.0],
    "max_coast_frames": [5, 10, 20],
    "max_occluded_frames": [25, 50, 100],
    "max_search_frames": [15, 30, 60],
    "restart_conf": [0.4, 0.5, 0.7],
    "cut_restart_conf": [0.2, 0.3, 0.5],
}
REPLAY_KNOBS = ("thr", "main_floor")


class Seq:
    """SeqData plus athlete boxes, camera motion and per-frame conditions."""

    def __init__(self, ref, cache: Path, motion: Path, abrupt_px: float):
        self.sd = SeqData(ref, cache)
        frames = json.loads((cache / ref.split / f"{ref.name}.json").read_text())["frames"]
        self.athletes = {int(k): v.get("athletes", []) for k, v in frames.items()}
        self.has_athletes = any("athletes" in v for v in frames.values())
        mp = motion / ref.split / f"{ref.name}.json"
        mv = json.loads(mp.read_text())["frames"] if mp.exists() else {}
        self.motion = {int(k): (None if v is None else np.asarray(v, float).reshape(2, 3)) for k, v in mv.items()}
        self.has_motion = bool(mv)
        persons = parse_labels(self.sd.reader.labels())[3]
        ts = sorted(self.sd.files)
        gt = {t: self.sd.gt[t][0] for t in ts if self.sd.gt.get(t)}
        breaks = [t for t in ts[1:] if self.has_motion and t in self.motion and self.motion[t] is None]
        self.cond = {}
        for t, g in gt.items():
            if any(abs(g[0] - p[0]) <= p[2] / 2 and abs(g[1] - p[1]) <= p[3] / 2 for p in persons.get(t, [])):
                c = "occluded"
            elif t - 1 in gt and t + 1 in gt and np.hypot(gt[t + 1][0] - 2 * g[0] + gt[t - 1][0],
                                                          gt[t + 1][1] - 2 * g[1] + gt[t - 1][1]) > abrupt_px:
                c = "abrupt"
            elif any(0 <= t - b <= 25 for b in breaks):
                c = "post_cut"
            else:
                c = "other"
            self.cond[t] = c

    @property
    def name(self):
        return self.sd.ref.name


def run_tracker(q: Seq, spec: Specialist, knobs: dict, a) -> list[dict]:
    sd = q.sd
    tr = RobustBallTracker(RobustTrackerConfig(**{k: v for k, v in knobs.items() if k not in REPLAY_KNOBS}))
    thr, floor = knobs["thr"], knobs["main_floor"]
    out = []
    for t in sorted(sd.files)[: a.max_frames or None]:
        tr.predict(t, q.motion.get(t), q.athletes.get(t, []))
        cands = [(b[0], b[1], b[4], "main") for b in sd.dets.get(t, []) if b[4] >= floor]
        rec = dict(t=t, out=None, pred=None, src="", win=None, sconf=None, gap=None, occ=False)
        if tr.match(cands) is None:
            s = tr.search()
            if s is not None:
                x0, y0, _ = clamp_window(s[0], s[1], sd.W, sd.H)
                key = (sd.ref.name, t, x0, y0)
                px, py, conf = spec(key, None if key in spec.memo else sd.window_frames(t), x0, y0)
                rec.update(win=(int(x0), int(y0)), sconf=float(conf), gap=tr.gap)
                if conf >= thr:
                    cands.append((px, py, conf, "specialist"))
        e = tr.update(cands)
        if e is not None:
            if e.source == "predicted":
                rec.update(pred=(e.x, e.y), src="pred", occ=e.occluded)
            else:
                rec.update(out=(e.x, e.y), src="main" if e.source == "main" else "spec")
        out.append(rec)
    return out


def run_phase1(q: Seq, spec: Specialist, a) -> list[dict]:
    outs = phase1_sweep.simulate(q.sd, spec, PHASE1_THR, PHASE1_GATE, a)
    for r in outs:
        r["pred"] = None
    return outs


def _d(p, gts):
    return min((float(np.hypot(p[0] - g[0], p[1] - g[1])) for g in gts), default=np.inf) if p else np.inf


def score(seqs, outs, exclude=()) -> dict:
    agg = {k: 0 for k in ("n", "pred_m", "hit_m", "pred_p", "hit_p", "unl_m")}
    hit = {t: 0 for t in TOLS}
    cond = {c: [0, 0, 0] for c in CONDS}       # n, hit measured, hit +predicted
    runs, longest = [], 0
    for q, out in zip(seqs, outs):
        if q.name in exclude:
            continue
        run = 0
        for r in out:
            gts = q.sd.gt.get(r["t"])
            if not gts:
                agg["unl_m"] += r["out"] is not None
                continue
            agg["n"] += 1
            dm = _d(r["out"], gts)
            dp = dm if r["out"] is not None else _d(r["pred"], gts)
            agg["pred_m"] += r["out"] is not None
            agg["pred_p"] += r["out"] is not None or r["pred"] is not None
            hm, hp = dm <= MATCH, dp <= MATCH
            agg["hit_m"] += hm
            agg["hit_p"] += hp
            for t in TOLS:
                hit[t] += dm <= t
            c = cond[q.cond.get(r["t"], "other")]
            c[0] += 1
            c[1] += hm
            c[2] += hp
            if hm:
                if run:
                    runs.append(run)
                run = 0
            else:
                run += 1
                longest = max(longest, run)
        if run:
            runs.append(run)
    n = max(agg["n"], 1)
    return {"ball_frames": agg["n"], "recall": agg["hit_m"] / n, "precision": agg["hit_m"] / max(agg["pred_m"], 1),
            "recall5": hit[5.0] / n, "recall10": hit[10.0] / n,
            "recall_pred": agg["hit_p"] / n, "precision_pred": agg["hit_p"] / max(agg["pred_p"], 1),
            "unlabelled_outputs": agg["unl_m"], "longest_miss": longest,
            "miss_run_p95": float(np.percentile(runs, 95)) if runs else 0.0,
            "cond": {c: {"n": v[0], "recall": v[1] / max(v[0], 1), "recall_pred": v[2] / max(v[0], 1)}
                     for c, v in cond.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="main cache, built with --athletes")
    ap.add_argument("--motion", type=Path, required=True, help="camera_motion_cache.py output")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--search_split", default="select")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--prec_drop", type=float, default=0.01)
    ap.add_argument("--min_gain", type=float, default=0.005)
    ap.add_argument("--abrupt_px", type=float, default=6.0)
    ap.add_argument("--leak_seqs", default="SNGS-050")
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--max_miss", type=int, default=15, help="Phase 1 baseline only")
    ap.add_argument("--main_conf", type=float, default=0.15, help="Phase 1 baseline only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    logf = open(a.out / "phase4_log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    import torch
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model, channels, epoch = load_model(a.ckpt, dev)
    spec = Specialist(model, channels, dev)
    refs = discover(a.src)
    leak = tuple(s for s in a.leak_seqs.split(",") if s)

    def load(split):
        rs = [r for r in refs if r.split == split][: a.max_seqs or None]
        return [Seq(r, a.cache, a.motion, a.abrupt_px) for r in rs]

    sel = load(a.search_split)
    log(f"c5c epoch {epoch} on {dev}; {a.search_split}: {len(sel)} sequences; athletes cached: "
        f"{all(q.has_athletes for q in sel)}; motion: {all(q.has_motion for q in sel)}")
    if not all(q.has_athletes for q in sel):
        log("WARNING: no athlete boxes in the cache: the occlusion model has nothing to follow")

    def evaluate(seqs, knobs):
        t0, c0 = time.time(), spec.calls
        outs = [run_tracker(q, spec, knobs, a) for q in seqs]
        s = score(seqs, outs)
        s.update(new_calls=spec.calls - c0, secs=round(time.time() - t0, 1))
        return s, outs

    base_s = score(sel, [run_phase1(q, spec, a) for q in sel])
    floor = base_s["precision"] - a.prec_drop
    log(f"[phase1 on {a.search_split}] recall {base_s['recall']:.2%} precision {base_s['precision']:.2%}; "
        f"floor {floor:.2%}")
    cfg0 = RobustTrackerConfig()
    defaults = {**{k: getattr(cfg0, k) for k in SPACE if k not in REPLAY_KNOBS}, "thr": PHASE1_THR, "main_floor": 0.15}
    rows, seen = [], {}

    def trial(knobs, tag):
        key = json.dumps(knobs, sort_keys=True)
        if key not in seen:
            s, _ = evaluate(sel, knobs)
            seen[key] = s
            rows.append(dict(tag=tag, **knobs, **{k: v for k, v in s.items() if k != "cond"}))
            log(f"  {tag:>28}: recall {s['recall']:.2%} prec {s['precision']:.2%} | +pred {s['recall_pred']:.2%} / "
                f"{s['precision_pred']:.2%} | +{s['new_calls']} calls {s['secs']}s")
        return seen[key]

    def obj(s):
        return (s["recall"] if s["precision"] >= floor else -1.0, s["precision"])

    best = dict(defaults)
    best_s = trial(best, "defaults")
    for p in range(a.passes):
        log(f"[pass {p + 1}]")
        improved = False
        for k, vals in SPACE.items():
            for v in vals:
                if v == best[k]:
                    continue
                s = trial({**best, k: v}, f"{k}={v}")
                if obj(s) > obj(best_s):
                    best, best_s, improved = {**best, k: v}, s, True
        if not improved:
            break
    log(f"chosen on {a.search_split}: {best}")
    for q in sel:
        q.sd.reader.close()

    test = load(a.eval_split)
    variants = {"tracker": best, "no_occlusion": {**best, "occlusion": False},
                "no_camera": {**best, "camera": False}, "one_model": {**best, "sigma_man": best["sigma_calm"]}}
    res, outs_of = {}, {}
    p1 = [run_phase1(q, spec, a) for q in test]
    res["phase1"] = score(test, p1)
    ex = score(test, p1, leak)
    res["phase1"].update(recall_ex=ex["recall"], recall_pred_ex=ex["recall_pred"])
    for name, kn in variants.items():
        s, outs = evaluate(test, kn)
        ex = score(test, outs, leak)
        s.update(recall_ex=ex["recall"], recall_pred_ex=ex["recall_pred"])
        res[name], outs_of[name] = s, outs
        log(f"[{a.eval_split}] {name:>13}: recall {s['recall']:.2%} prec {s['precision']:.2%} | +pred "
            f"{s['recall_pred']:.2%} / {s['precision_pred']:.2%} | longest {s['longest_miss']}")
    b, t = res["phase1"], res["tracker"]
    keep = t["recall"] - b["recall"] >= a.min_gain and b["precision"] - t["precision"] <= a.prec_drop
    keep_pred = t["recall_pred"] - t["recall"] >= a.min_gain and t["precision_pred"] >= b["precision"] - a.prec_drop

    fdir = a.out / "frames_tracker"
    fdir.mkdir(parents=True, exist_ok=True)
    with open(fdir / "frames.jsonl", "w") as f:
        for q, out in zip(test, outs_of["tracker"]):
            for r in out:
                gts = q.sd.gt.get(r["t"], [])
                f.write(json.dumps(dict(split=a.eval_split, seq=q.name, t=r["t"], gt=[list(g) for g in gts],
                                        out=list(r["out"]) if r["out"] else None, src=r["src"] if r["out"] else "",
                                        win=r["win"], sconf=r["sconf"], gap=r["gap"],
                                        hit=float(bool(gts) and _d(r["out"], gts) <= MATCH))) + "\n")
    for q in test:
        q.sd.reader.close()

    cols = ["tag", *SPACE, "recall", "precision", "recall_pred", "precision_pred", "longest_miss", "miss_run_p95",
            "new_calls", "secs"]
    with open(a.out / "search.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump({"chosen": best, "search_split": a.search_split, "floor": floor, "phase1_search": base_s,
               "eval": res, "keep_tracker": keep, "keep_predicted_output": keep_pred, "leak_seqs": leak,
               "spec_calls": spec.calls, "spec_ms_mean": spec.ms / max(spec.calls, 1)},
              open(a.out / "phase4.json", "w"), indent=1, default=float)

    pc = lambda x: f"{x:.2%}"  # noqa: E731
    L = ["# Phase 4: robust ball tracker (PIPELINE_IMPROVEMENT_PLAN_1.md)", "",
         f"Tracker knobs chosen on `{a.search_split}` (coordinate descent, {len(rows)} runs; precision floor "
         f"{pc(floor)} = Phase 1 minus {a.prec_drop:.0%}), frozen for `{a.eval_split}`.", "",
         f"## `{a.eval_split}`", "",
         f"| system | recall @15 | @10 | @5 | precision | w/o {','.join(leak)} | +predicted recall | "
         "+predicted precision | longest miss | miss run p95 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in res.items():
        L.append(f"| {name} | {pc(s['recall'])} | {pc(s['recall10'])} | {pc(s['recall5'])} | {pc(s['precision'])} | "
                 f"{pc(s['recall_ex'])} | {pc(s['recall_pred'])} | {pc(s['precision_pred'])} | {s['longest_miss']} | "
                 f"{s['miss_run_p95']:.0f} |")
    L += ["", f"**Tracker vs Phase 1 (measured outputs): {'KEEP' if keep else 'REVERT'}** (kill criterion: "
          f"+{a.min_gain:.1%} recall, <= {a.prec_drop:.0%} precision loss). "
          f"**Output predicted positions: {'yes' if keep_pred else 'no'}.**", "",
          "## Recall by condition (measured / +predicted)", "",
          "| system | " + " | ".join(f"{c} (n={res['tracker']['cond'][c]['n']})" for c in CONDS) + " |",
          "|---|" + "---|" * len(CONDS)]
    for name, s in res.items():
        L.append(f"| {name} | " + " | ".join(f"{pc(s['cond'][c]['recall'])} / {pc(s['cond'][c]['recall_pred'])}"
                                             for c in CONDS) + " |")
    L += ["", "Conditions come from the labels: occluded = GT ball inside a GT person box; abrupt = second "
          f"difference of the GT track > {a.abrupt_px:g} px; post_cut = within 25 frames of a frame without a "
          "camera-motion estimate.", "", "## Chosen knobs", "", "```", json.dumps(best, indent=1), "```", "",
          f"Search log: `search.csv`. Specialist calls: {spec.calls}, mean {spec.ms / max(spec.calls, 1):.1f} ms."]
    (a.out / "report_phase4.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"wrote {a.out / 'report_phase4.md'}")


if __name__ == "__main__":
    main()
