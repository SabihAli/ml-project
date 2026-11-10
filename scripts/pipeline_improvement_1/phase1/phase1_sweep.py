"""
phase1_sweep.py -- Phase 1 of docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md: the detector-ball
plausibility gate (gsr_pipeline/ball_gate.py, task 1) and the c5c threshold sweep (task 2).

Replays main + c5c on cached main detections, exactly as BallFuser.update runs it (fused chain;
with the gate: first plausible candidate, none plausible = detector miss). The specialist is
memoised per window, so configurations that share windows share GPU work.

Search, on `select` only:
  A. gate grid at threshold 0.70 (plus gate off);
  B. thresholds 0.40-0.85 for gate off and the best --top_gates gates of A.
  Rule: highest select recall with select precision >= gate-off@0.70 precision - --prec_drop.
Then, unchanged on `test`: baseline, gate only, threshold only, gate + threshold, each also
without --leak_seqs (SNGS-050 overlaps a `pool` clip c5c trained on). A candidate is kept only
if test recall gains >= --min_gain and test precision drops <= --prec_drop (the plan's kill
criterion); of the kept ones the highest test recall is chosen.

Outputs (--out): report_phase1.md, phase1.json, sweep.csv, phase1_log.txt, and
frames_<chosen>/frames.jsonl in diagnose_ball_misses.py's format (feed it to
scripts/pipeline_improvement_1/phase0/phase0_report.py --diag for the new miss buckets).

    python scripts/pipeline_improvement_1/phase1/phase1_sweep.py --src data/datasets/SoccerNetGS \
        --cache data/outputs/pipeline_improvement_1/phase0/main_cache \
        --ckpt runs/ball_temporal/c5c/best.pt --out data/outputs/pipeline_improvement_1/phase1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "gsr_pipeline"))          # ball_gate.py alone, without the package
sys.path.insert(0, str(ROOT / "scripts" / "ball_temporal"))
sys.path.insert(0, str(ROOT / "scripts" / "ball_eval"))
from ball_gate import BallGate, GateParams  # noqa: E402
from eval_ball_temporal import SeqData, Specialist, clamp_window, load_model  # noqa: E402
from gsr_io import discover  # noqa: E402

TOLS = (5.0, 10.0, 15.0)
MATCH = 15.0
BASE_THR = 0.70
THRESHOLDS = [round(float(x), 2) for x in np.arange(0.40, 0.851, 0.05)]
GRID = dict(r0=[30.0], speed=[30.0, 50.0, 80.0], conf_hi=[0.6, 0.8, 1.01], max_rejects=[5, 15])


def gate_grid():
    return [GateParams(enabled=True, **dict(zip(GRID, v))) for v in itertools.product(*GRID.values())]


def simulate(sd: SeqData, spec: Specialist, thr: float, gp: GateParams, a) -> list[dict]:
    """BallFuser.update (fused chain) with the gate. Per frame: t, out, src, win, sconf, gap, rej."""
    gate = BallGate(gp, a.max_miss)
    out, last = [], None
    for t in sorted(sd.files)[: a.max_frames or None]:
        cands = sorted((b for b in sd.dets.get(t, []) if b[4] >= a.main_conf), key=lambda b: -b[4])
        rec = dict(t=t, out=None, src="", win=None, sconf=None, gap=None, rej=None)
        i = gate.pick([(b[0], b[1], b[4]) for b in cands], t, last)
        if i is not None:
            last = (cands[i][0], cands[i][1], t)
            rec.update(out=(cands[i][0], cands[i][1]), src="main")
            out.append(rec)
            continue
        if cands:
            rec["rej"] = (cands[0][0], cands[0][1])
        if last is not None and t - last[2] <= a.max_miss:
            x0, y0, _ = clamp_window(last[0], last[1], sd.W, sd.H)
            rec.update(gap=t - last[2], win=(int(x0), int(y0)))
            key = (sd.ref.name, t, x0, y0)
            px, py, conf = spec(key, None if key in spec.memo else sd.window_frames(t), x0, y0)
            rec["sconf"] = float(conf)
            if conf >= thr:
                rec.update(out=(px, py), src="spec")
                last = (px, py, t)
        out.append(rec)
    return out


def _d(p, gts):
    return min((float(np.hypot(p[0] - g[0], p[1] - g[1])) for g in gts), default=np.inf) if p else np.inf


def score(sds, outs, exclude=()) -> dict:
    n = pred = unl = rej = rej_good = calls = 0
    hit = {t: 0 for t in TOLS}
    longest = []
    for sd, out in zip(sds, outs):
        if sd.ref.name in exclude:
            continue
        run = lg = 0
        for r in out:
            gts = sd.gt.get(r["t"])
            calls += r["win"] is not None
            if r["rej"] is not None:
                rej += 1
                rej_good += bool(gts) and _d(r["rej"], gts) <= MATCH
            if not gts:
                unl += r["out"] is not None
                continue
            n += 1
            d = _d(r["out"], gts)
            pred += r["out"] is not None
            for t in TOLS:
                hit[t] += d <= t
            run = 0 if d <= MATCH else run + 1
            lg = max(lg, run)
        longest.append(lg)
    return {"ball_frames": n, "recall": hit[15.0] / max(n, 1), "recall5": hit[5.0] / max(n, 1),
            "recall10": hit[10.0] / max(n, 1), "precision": hit[15.0] / max(pred, 1),
            "unlabelled_outputs": unl, "spec_calls": calls, "gate_rejects": rej,
            "gate_rejects_correct": rej_good, "longest_miss_frames": max(longest, default=0)}


def run_configs(sds, spec, configs, a, log, exclude=()):
    rows = []
    for gp, thr in configs:
        t0, c0 = time.time(), spec.calls
        outs = [simulate(sd, spec, thr, gp, a) for sd in sds]
        s = score(sds, outs)
        row = dict(gate=gp.tag(), thr=thr, **s, new_calls=spec.calls - c0, secs=round(time.time() - t0, 1),
                   params=gp.as_dict(), _outs=outs)
        if exclude:
            ex = score(sds, outs, exclude)
            row.update(recall_ex=ex["recall"], precision_ex=ex["precision"])
        rows.append(row)
        log(f"  {gp.tag():>22} thr {thr:.2f}: recall {s['recall']:.2%}  precision {s['precision']:.2%}  "
            f"rejects {s['gate_rejects']} ({s['gate_rejects_correct']} correct)  +{row['new_calls']} calls  {row['secs']}s")
    return rows


def best(rows, floor):
    ok = [r for r in rows if r["precision"] >= floor]
    return max(ok, key=lambda r: (r["recall"], r["precision"])) if ok else None


def write_frames(fdir: Path, split, sds, outs):
    with open(fdir / "frames.jsonl", "a") as f:
        for sd, out in zip(sds, outs):
            for r in out:
                gts = sd.gt.get(r["t"], [])
                hit = bool(gts) and _d(r["out"], gts) <= MATCH
                f.write(json.dumps(dict(split=split, seq=sd.ref.name, t=r["t"], gt=[list(g) for g in gts],
                                        out=list(r["out"]) if r["out"] else None, src=r["src"], win=r["win"],
                                        sconf=r["sconf"], gap=r["gap"], hit=float(hit))) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="GSR root (gsr_io.discover)")
    ap.add_argument("--cache", type=Path, required=True, help="main-detector cache (Phase 0 main_cache/)")
    ap.add_argument("--ckpt", type=Path, required=True, help="c5c best.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--main_conf", type=float, default=0.15)
    ap.add_argument("--max_miss", type=int, default=15)
    ap.add_argument("--top_gates", type=int, default=3)
    ap.add_argument("--prec_drop", type=float, default=0.01, help="precision guardrail (fraction)")
    ap.add_argument("--min_gain", type=float, default=0.005, help="kill criterion: minimum test recall gain")
    ap.add_argument("--leak_seqs", default="SNGS-050", help="comma list, test also reported without them")
    ap.add_argument("--max_seqs", type=int, default=0, help="smoke: first N sequences per split")
    ap.add_argument("--max_frames", type=int, default=0, help="smoke: first N frames per sequence")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    logf = open(a.out / "phase1_log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    import torch
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model, channels, epoch = load_model(a.ckpt, dev)
    spec = Specialist(model, channels, dev)
    log(f"c5c {a.ckpt} (epoch {epoch}) on {dev}")
    refs = discover(a.src)
    leak = tuple(s for s in a.leak_seqs.split(",") if s)

    def load(split):
        return [SeqData(r, a.cache) for r in [r for r in refs if r.split == split][: a.max_seqs or None]]

    sel = load("select")
    log(f"select: {len(sel)} sequences\n[A] gate grid at threshold {BASE_THR}")
    rows_a = run_configs(sel, spec, [(GateParams(), BASE_THR)] + [(g, BASE_THR) for g in gate_grid()], a, log)
    base_sel = rows_a[0]
    floor = base_sel["precision"] - a.prec_drop
    gp_of = {r["gate"]: GateParams(**r["params"]) for r in rows_a}
    top = [r["gate"] for r in sorted((r for r in rows_a[1:] if r["precision"] >= floor), key=lambda r: -r["recall"])[: a.top_gates]]
    log(f"[B] thresholds {THRESHOLDS} for gate off + {top}")
    rows_b = run_configs(sel, spec, [(gp_of[g], t) for g in ["off"] + top for t in THRESHOLDS if t != BASE_THR], a, log)
    allrows = rows_a + rows_b
    picks = {"baseline": base_sel,
             "threshold_only": best([r for r in allrows if r["gate"] == "off"], floor),
             "gate_only": best([r for r in allrows if r["gate"] != "off" and r["thr"] == BASE_THR], floor),
             "gate_threshold": best(allrows, floor)}
    picks = {k: v for k, v in picks.items() if v is not None}
    for sd in sel:
        sd.reader.close()

    test = load("test")
    log(f"test: {len(test)} sequences; frozen picks: " + ", ".join(f"{k}={v['gate']}@{v['thr']}" for k, v in picks.items()))
    test_of = dict(zip(picks, run_configs(test, spec, [(gp_of[v["gate"]], v["thr"]) for v in picks.values()],
                                          a, log, exclude=leak)))
    bt = test_of["baseline"]
    verdict = {k: ("keep" if r["recall"] - bt["recall"] >= a.min_gain and bt["precision"] - r["precision"] <= a.prec_drop
                   else "revert") for k, r in test_of.items() if k != "baseline"}
    kept = [k for k, v in verdict.items() if v == "keep"]
    chosen = max(kept, key=lambda k: test_of[k]["recall"]) if kept else "baseline"

    fdir = a.out / f"frames_{chosen}"
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / "frames.jsonl").unlink(missing_ok=True)
    sel = load("select")
    write_frames(fdir, "select", sel, [simulate(sd, spec, picks[chosen]["thr"], gp_of[picks[chosen]["gate"]], a) for sd in sel])
    write_frames(fdir, "test", test, test_of[chosen]["_outs"])
    for sd in sel + test:
        sd.reader.close()

    cols = ["split", "gate", "thr", "recall", "recall5", "recall10", "precision", "recall_ex", "precision_ex",
            "unlabelled_outputs", "spec_calls", "gate_rejects", "gate_rejects_correct", "longest_miss_frames",
            "new_calls", "secs"]
    with open(a.out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in allrows:
            w.writerow(dict(r, split="select"))
        for k, r in test_of.items():
            w.writerow(dict(r, split=f"test:{k}"))

    strip = lambda r: {k: v for k, v in r.items() if k != "_outs"}  # noqa: E731
    json.dump({"select": {k: strip(v) for k, v in picks.items()}, "test": {k: strip(v) for k, v in test_of.items()},
               "verdict": verdict, "chosen": chosen, "chosen_params": gp_of[picks[chosen]["gate"]].as_dict(),
               "chosen_threshold": picks[chosen]["thr"], "leak_seqs": leak, "select_precision_floor": floor,
               "spec_calls_total": spec.calls, "spec_ms_mean": spec.ms / max(spec.calls, 1)},
              open(a.out / "phase1.json", "w"), indent=1)

    pc = lambda x: f"{x:.2%}"  # noqa: E731
    L = ["# Phase 1: plausibility gate + c5c threshold (PIPELINE_IMPROVEMENT_PLAN_1.md)", "",
         f"Chosen on `select` (precision floor {pc(floor)} = gate off @ {BASE_THR} minus {a.prec_drop:.0%}), frozen "
         f"for `test`. Kill criterion: test recall gain >= {a.min_gain:.1%} and precision drop <= {a.prec_drop:.0%}.", "",
         "## Test", "",
         f"| config | gate | thr | recall @15 | @10 | @5 | precision | recall w/o {','.join(leak)} | rejects (correct) | spec calls | longest miss | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in test_of.items():
        L.append(f"| {k} | `{r['gate']}` | {r['thr']} | {pc(r['recall'])} | {pc(r['recall10'])} | {pc(r['recall5'])} | "
                 f"{pc(r['precision'])} | {pc(r['recall_ex'])} | {r['gate_rejects']} ({r['gate_rejects_correct']}) | "
                 f"{r['spec_calls']} | {r['longest_miss_frames']} | {verdict.get(k, '—')} |")
    L += ["", f"**Chosen: `{chosen}`.** Replay: `frames_{chosen}/frames.jsonl`; buckets: `buckets_{chosen}/`.", "",
          "\"rejects (correct)\": frames where the gate rejected the detector's top ball, and how many of those "
          "were actually on the ball (the gate's cost).", "",
          "## Select search (top 25 by recall)", "", "| gate | thr | recall | precision | rejects (correct) |",
          "|---|---|---|---|---|"]
    for r in sorted(allrows, key=lambda r: -r["recall"])[:25]:
        L.append(f"| `{r['gate']}` | {r['thr']} | {pc(r['recall'])} | {pc(r['precision'])} | "
                 f"{r['gate_rejects']} ({r['gate_rejects_correct']}) |")
    L += ["", f"Full grid: `sweep.csv`. Specialist calls run: {spec.calls}, mean {spec.ms / max(spec.calls, 1):.1f} ms."]
    (a.out / "report_phase1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"chosen: {chosen}; wrote {a.out / 'report_phase1.md'}")


if __name__ == "__main__":
    main()
