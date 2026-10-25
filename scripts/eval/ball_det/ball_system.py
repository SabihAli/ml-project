"""
ball_system.py -- main detector + c5c + ball tracker, simulated frame by frame, for
docs/plans/BALL_IMPROVEMENT_PLAN.md items 2-3 (the `track` stage of eval_ball_temporal.py, and
diagnose_ball_misses.py). Builds on eval_ball_temporal.py's Specialist (memoised windows) and
SeqData (labels, cached main detections, decoded-frame cache).

Per frame t of a sequence:
  1. tracker.predict(t, camera motion t-1 -> t)                (motion cache; None = no estimate)
  2. main candidates: every cached main ball at conf >= the tracker's main floor (up to 5)
  3. the specialist runs when the tracker has a track whose last measurement is at most max_miss
     frames old AND main's best candidate is below spec_run_below (0.15 = "main missed", today's
     rule). Window centre: the last measurement (`last_seen`, today) or the tracker's prediction
     (`predicted`). n_windows 3 adds windows shifted +-half a crop along the predicted velocity
     (or horizontally when it is ~0) from gap >= 2; the highest peak wins.
  4. tracker.update(candidates) -> this frame's live output (main | specialist | predicted).

Batch on top: ball_tracker.best_path over the same per-frame candidates, then interpolate.

Every output is scored with eval_ball_temporal.score (15 px centre match), so numbers compare
directly with gate 3. Sources map to score's `src`: specialist -> "spec" (so spec_precision means
the same thing), predicted -> "pred", interpolated -> "interp".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "scripts" / "ball_temporal", ROOT / "scripts" / "eval" / "ball_det", ROOT / "gsr_pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import ball_tracker as bt  # noqa: E402  (plain module import: no gsr_pipeline package init)
from build_ball_temporal_dataset import CROP, clamp_window  # noqa: E402
from eval_ball_temporal import score  # noqa: E402


@dataclass
class SimConfig:
    tracker: bt.TrackerConfig = field(default_factory=bt.TrackerConfig)
    window_centre: str = "last_seen"      # last_seen | predicted
    n_windows: int = 1                    # 1 | 3 (from gap >= 2)
    spec_run_below: float = 0.15          # run the specialist when main's best candidate is below this
    max_miss: int = 15

    def describe(self) -> str:
        t = self.tracker
        return (f"centre={self.window_centre} windows={self.n_windows} run_below={self.spec_run_below} "
                f"lam={t.lam} coast={t.max_coast_frames} spec_floor={t.min_conf_spec} main_floor={t.min_conf_main} "
                f"gate={t.gate_a}+{t.gate_b}g")


def _windows(cx, cy, vel, gap, cfg: SimConfig, W, H):
    tops = [clamp_window(cx, cy, W, H)[:2]]
    if cfg.n_windows >= 3 and gap >= 2:
        vx, vy = vel
        n = float(np.hypot(vx, vy))
        ux, uy = (vx / n, vy / n) if n > 1.0 else (1.0, 0.0)
        for s in (-1, 1):
            tops.append(clamp_window(cx + s * ux * CROP / 2, cy + s * uy * CROP / 2, W, H)[:2])
    return list(dict.fromkeys(tops))       # dedupe windows clamped to the same place


def simulate_tracked(sd, spec, motion: dict, cfg: SimConfig) -> list[dict]:
    """-> per frame {t, live: (x, y, src) | None, cands: [BallCandidate], gap, spec_ran}."""
    tr = bt.BallTracker(cfg.tracker)
    out = []
    for t in sorted(sd.files):
        pred = tr.predict(t, motion.get(t))
        cands = [bt.BallCandidate(b[0], b[1], b[4], "main", b[2], b[3])
                 for b in sd.dets.get(t, []) if b[4] >= cfg.tracker.min_conf_main]
        main_best = max((c.conf for c in cands), default=0.0)
        gap = tr.gap if tr.last_meas is not None else None
        ran = False
        if spec is not None and tr.active and gap is not None and 1 <= gap <= cfg.max_miss \
                and main_best < cfg.spec_run_below:
            if cfg.window_centre == "predicted" and pred is not None:
                cx, cy = pred
            else:
                cx, cy = tr.last_meas[0], tr.last_meas[1]
            vel = (float(tr.x[2]), float(tr.x[3])) if tr.x is not None else (0.0, 0.0)
            best = None
            for x0, y0 in _windows(cx, cy, vel, gap, cfg, sd.W, sd.H):
                key = (sd.ref.name, t, x0, y0)
                frames = None if key in spec.memo else sd.window_frames(t)
                p = spec(key, frames, x0, y0)
                if best is None or p[2] > best[2]:
                    best = p
            ran = True
            cands.append(bt.BallCandidate(best[0], best[1], best[2], "specialist"))
        est = tr.update(cands)
        live = None
        if est is not None:
            live = (est.x, est.y, {"main": "main", "specialist": "spec", "predicted": "pred"}[est.source])
        out.append(dict(t=t, live=live, cands=cands, gap=gap, spec_ran=ran))
    return out


def _records(sim, mode: str, path_cfg: Optional[bt.PathConfig] = None, motion=None, max_interp=25):
    """Per-frame records in eval_ball_temporal.score's format for one output mode:
    measured | live (measured + coasted) | batch (best_path picks) | batch_interp."""
    recs = []
    if mode in ("measured", "live"):
        for r in sim:
            x = y = None
            src = ""
            if r["live"] is not None and (mode == "live" or r["live"][2] != "pred"):
                x, y, src = r["live"]
            recs.append(dict(t=r["t"], x=x, y=y, src=src, gap=r["gap"] if r["spec_ran"] else None))
        return recs
    ts = [r["t"] for r in sim]
    affines = [None] + [motion.get(t) if motion else None for t in ts[1:]]
    picks = bt.best_path([r["cands"] for r in sim], affines, path_cfg)
    pos = bt.interpolate(picks, affines, max_interp) if mode == "batch_interp" else \
        [(p.x, p.y) if p is not None else None for p in picks]
    for r, p, q in zip(sim, picks, pos):
        if q is None:
            recs.append(dict(t=r["t"], x=None, y=None, src="", gap=None))
        else:
            src = "interp" if p is None else ("spec" if p.source == "specialist" else "main")
            recs.append(dict(t=r["t"], x=q[0], y=q[1], src=src, gap=r["gap"] if r["spec_ran"] else None))
    return recs


def score_modes(sds, sims, motions, path_cfg=None, modes=("measured", "live", "batch", "batch_interp")):
    """{mode: score dict} over a split."""
    res = {}
    for m in modes:
        recs = [_records(sim, m, path_cfg, mo) for sim, mo in zip(sims, motions)]
        res[m] = score(sds, recs)
    return res


def run_split(sds, spec, motions, cfg: SimConfig):
    return [simulate_tracked(sd, spec, mo, cfg) for sd, mo in zip(sds, motions)]


def variant(cfg: SimConfig, **kw) -> SimConfig:
    """Copy of cfg with SimConfig fields or tracker fields (prefixed t_) replaced."""
    tk = {k[2:]: v for k, v in kw.items() if k.startswith("t_")}
    sk = {k: v for k, v in kw.items() if not k.startswith("t_")}
    return replace(cfg, tracker=replace(cfg.tracker, **tk), **sk)


# --------------------------------------------------------------------------- the `track` stage

# Coordinate search on `select` (plan §3.7): each tracker knob in turn, keeping the best, twice.
TRACKER_GRID = {
    "t_min_conf_main": [0.05, 0.10, 0.15],
    "t_min_conf_spec": [0.30, 0.50, 0.70],
    "t_lam": [0.25, 0.5, 1.0],
    "t_max_coast_frames": [0, 5, 10],
    "t_gate_b": [20.0, 30.0, 45.0],
    "t_restart_conf": [0.4, 0.6],
}
# Item 3 window variants (plan §4.1-4.2), a full grid on top of the best tracker.
WINDOW_GRID = [dict(window_centre=c, n_windows=n, spec_run_below=b)
               for c in ("last_seen", "predicted") for n in (1, 3) for b in (0.15, 0.30)]
PATH_GRID = [dict(start_pen=s, c0=c) for s in (0.2, 0.4, 0.8) for c in (0.05, 0.10, 0.20)]


def _objective(res: dict, mode: str, floor: float) -> tuple:
    """Higher is better: meet the precision floor first, then recall."""
    r = res[mode]
    return (r["precision"] >= floor, r["recall"] if r["precision"] >= floor else r["precision"])


def _cfg_dict(cfg: SimConfig, path: Optional[bt.PathConfig]) -> dict:
    from dataclasses import asdict
    d = {"sim": {k: v for k, v in asdict(cfg).items() if k != "tracker"}, "tracker": asdict(cfg.tracker)}
    if path is not None:
        d["path"] = {k: v for k, v in asdict(path).items() if k != "tracker"}
    return d


def stage_track(a, model, channels, dev):
    """Item 2 (+ item 3's window variants): tune on `select`, freeze, report `test`. Writes
    <out>/report_track.md, results_track.json, tracker_config.json, search_log.csv."""
    import csv
    import json
    import time

    import camera_motion_cache as cmc
    from eval_ball_temporal import SeqData, Specialist, discover, simulate

    spec = Specialist(model, channels, dev)
    splits = {}
    for split in ("select", "test"):
        refs = [r for r in discover(a.src) if r.split == split]
        if a.max_seqs:
            refs = refs[:a.max_seqs]
        sds = [SeqData(r, a.cache) for r in refs]
        motions = [cmc.load(a.motion, r) if a.motion else {} for r in refs]
        splits[split] = (sds, motions)
        print(f"[track] {split}: {len(sds)} sequences", flush=True)

    def baseline(split):
        sds, _ = splits[split]
        main = score(sds, [simulate(sd, None, 1.0, a) for sd in sds])
        c5c = score(sds, [simulate(sd, spec, a.system_thr, a) for sd in sds])
        return main, c5c

    sel_main, sel_c5c = baseline("select")
    floor = sel_c5c["precision"]
    print(f"[track] select baseline: main {sel_main['recall']:.2%}/{sel_main['precision']:.2%}, "
          f"main+c5c (thr {a.system_thr}) {sel_c5c['recall']:.2%}/{sel_c5c['precision']:.2%}; "
          f"precision floor {floor:.2%}", flush=True)

    log_rows = []

    def evaluate(cfg, path=None, tag=""):
        t0 = time.time()
        sds, motions = splits["select"]
        sims = run_split(sds, spec, motions, cfg)
        res = score_modes(sds, sims, motions, path)
        row = {"tag": tag, "config": cfg.describe(), **{f"{m}_{k}": round(res[m][k], 4)
               for m in res for k in ("recall", "precision")}, "spec_calls": spec.calls,
               "seconds": round(time.time() - t0, 1)}
        log_rows.append(row)
        print(f"[track] {tag:<10} {cfg.describe()} | live {res['live']['recall']:.2%}/{res['live']['precision']:.2%} "
              f"measured {res['measured']['recall']:.2%}/{res['measured']['precision']:.2%} "
              f"batch+interp {res['batch_interp']['recall']:.2%}/{res['batch_interp']['precision']:.2%} "
              f"({row['seconds']}s)", flush=True)
        return res, sims

    best_cfg = SimConfig()
    best_res, best_sims = evaluate(best_cfg, tag="start")
    for p in range(a.search_passes):
        for knob, values in TRACKER_GRID.items():
            for v in values:
                cand = variant(best_cfg, **{knob: v})
                if cand == best_cfg:
                    continue
                res, sims = evaluate(cand, tag=f"p{p + 1} {knob[2:]}")
                if _objective(res, "live", floor) > _objective(best_res, "live", floor):
                    best_cfg, best_res, best_sims = cand, res, sims
    tracker_only = best_cfg
    for w in WINDOW_GRID:
        cand = variant(best_cfg, **w)
        if cand == best_cfg:
            continue
        res, sims = evaluate(cand, tag="window")
        if _objective(res, "live", floor) > _objective(best_res, "live", floor):
            best_cfg, best_res, best_sims = cand, res, sims

    sds, motions = splits["select"]
    best_path_cfg = bt.PathConfig(tracker=best_cfg.tracker)
    best_path_res = score_modes(sds, best_sims, motions, best_path_cfg, ("batch_interp",))
    for g in PATH_GRID:
        pc = bt.PathConfig(tracker=best_cfg.tracker, **g)
        res = score_modes(sds, best_sims, motions, pc, ("batch_interp",))
        print(f"[track] path {g} | batch+interp {res['batch_interp']['recall']:.2%}/"
              f"{res['batch_interp']['precision']:.2%}", flush=True)
        if _objective(res, "batch_interp", floor) > _objective(best_path_res, "batch_interp", floor):
            best_path_cfg, best_path_res = pc, res

    print(f"[track] chosen on select: {best_cfg.describe()}; path start_pen={best_path_cfg.start_pen} "
          f"c0={best_path_cfg.c0}", flush=True)

    # ---- frozen, on test
    test_main, test_c5c = baseline("test")
    sds, motions = splits["test"]
    spec.calls, spec.ms = 0, 0.0
    sims = run_split(sds, spec, motions, best_cfg)
    test = score_modes(sds, sims, motions, best_path_cfg)
    calls, ms = spec.calls, spec.ms / max(spec.calls, 1)
    sims_t = run_split(sds, spec, motions, tracker_only) if tracker_only != best_cfg else sims
    test_tracker_only = score_modes(sds, sims_t, motions, best_path_cfg, ("live",))["live"]

    live = test["live"]
    gate_t = (live["recall"] > test_c5c["recall"] and live["precision"] >= test_c5c["precision"]
              and live["longest_miss_mean_s"] < test_c5c["longest_miss_mean_s"])

    def row(name, r):
        return (f"| {name} | {r['recall']:.2%} | {r['precision']:.2%} | {r['outputs_unlabelled']} | "
                f"{r['longest_miss_mean_s']:.1f} s / {r['longest_miss_max_s']:.1f} s |")

    L = ["# Ball tracker: test results (plan item 2, item 3 window variants)", "",
         f"Specialist `{a.ckpt}`, main cache `{a.cache}`, motion cache `{a.motion}`. Every setting chosen on "
         f"`select` ({len(splits['select'][0])} sequences), frozen for `test` ({len(sds)} sequences). 15 px centre match. "
         f"Precision floor for the search: main + c5c's precision on `select`, {floor:.2%}.", "",
         f"Chosen: `{best_cfg.describe()}`; batch path start_pen {best_path_cfg.start_pen}, c0 {best_path_cfg.c0}.", "",
         "| test | recall | precision | outputs on unlabelled frames | longest miss mean / max |", "|---|---|---|---|---|",
         row("main detector alone", test_main), row(f"main + c5c, as wired (thr {a.system_thr})", test_c5c),
         row("tracker, measured only", test["measured"]), row("tracker, live (measured + coasted)", live),
         row("tracker only (no item 3 window change), live", test_tracker_only),
         row("batch path", test["batch"]), row("batch path + interpolation", test["batch_interp"]), "",
         f"Gate T (live recall above main + c5c, precision no lower, longest miss shorter): "
         f"**{'PASS' if gate_t else 'FAIL'}**.", "",
         "Live recall by frames since the last measurement (frames where the specialist ran):", "",
         "| gap | frames | recall |", "|---|---|---|"]
    for k, (r, n) in live["by_gap"].items():
        L.append(f"| {k} | {n} | {r:.2%} |")
    L += ["", f"Specialist calls on test: {calls} ({ms:.1f} ms per call, PyTorch, this machine).", "",
          f"Search log: `search_log.csv` ({len(log_rows)} select runs)."]
    text = "\n".join(L)
    print("\n" + text, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report_track.md").write_text(text + "\n", encoding="utf-8")
    (a.out / "tracker_config.json").write_text(json.dumps(_cfg_dict(best_cfg, best_path_cfg), indent=1))
    (a.out / "results_track.json").write_text(json.dumps(
        {"select_main": sel_main, "select_c5c": sel_c5c, "select_best": best_res, "test_main": test_main,
         "test_c5c": test_c5c, "test": test, "test_tracker_only_live": test_tracker_only, "gate_t": gate_t,
         "config": _cfg_dict(best_cfg, best_path_cfg), "spec_calls_test": calls, "ms_per_call": ms},
        indent=1, default=str))
    with open(a.out / "search_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        w.writeheader()
        w.writerows(log_rows)
    for sds_, _ in splits.values():
        for sd in sds_:
            sd.reader.close()
    return gate_t
