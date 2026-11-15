"""
trajectory_pass.py -- Phase 3b offline trajectory pass over a teacher's per-frame candidates
(docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md, Phase 3b). CPU only, seconds per run.

Input: `train_teacher_yolo_motion.py dump` (top-k candidates per frame + GT, select.json / test.json).
Offline, so every frame sees both its past and its future:

  1. Temporal support. Each candidate's score = its own confidence + w * sum over frames t' within +-W of
     max_j conf_j' * exp(-d^2 / (2 * sigma(dt)^2)), sigma(dt) = sigma0 + v * |dt| (a ball moving up to v px
     per frame). A real ball has neighbours along its path; an isolated false positive has none. Per frame
     the candidate with the highest support is kept.
  2. Gap fill. A frame left without an output (support below the threshold) between two kept frames at most
     G frames apart, whose displacement per frame is <= vmax, gets the linear interpolation, scored with the
     lower of the two endpoint scores.

Scoring is train_teacher_yolo's: one ball per frame, 15 px, threshold chosen on select (highest recall at
precision >= 90%), frozen for test. Knobs (W, sigma0, v, w, G) are chosen on select by the same rule; test is
read once with them. The raw row (top-1, no pass) must reproduce the eval.

    python scripts/pipeline_improvement_1/phase3b/trajectory_pass.py --cands data/outputs/pipeline_improvement_1/phase3b/teacher_yolo/candidates/t2m_s1920_ft
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_teacher_yolo as base  # noqa: E402

THRS = np.r_[0.002, 0.005, np.round(np.arange(0.01, 0.96, 0.01), 2)]
EVAL_THRS = np.round(np.arange(0.01, 0.96, 0.01), 2)      # the eval's grid, for the parity row


def load(path: Path):
    doc = json.loads(path.read_text())
    cands, gt = {}, {}
    for s, d in doc["seqs"].items():
        frames = sorted(int(f) for f in d["gt"])
        cands[s] = (frames, [d["cands"].get(str(f), []) for f in frames])
        gt[s] = {int(f): [tuple(b) for b in g] for f, g in d["gt"].items()}
    return cands, gt


def raw(cands) -> dict:
    return {s: {f: (tuple(c[0]) if c else None) for f, c in zip(fr, cs)} for s, (fr, cs) in cands.items()}


def support_pass(frames, cs, W, sigma0, v, w):
    """(frames, xy (T, 2), score (T,)): per frame the candidate with the most temporal support."""
    T, K = len(frames), max((len(c) for c in cs), default=0)
    if K == 0:
        return frames, np.zeros((T, 2)), np.zeros(T)
    xy, cf = np.zeros((T, K, 2)), np.zeros((T, K))
    for t, c in enumerate(cs):
        for k, (x, y, p) in enumerate(c[:K]):
            xy[t, k], cf[t, k] = (x, y), p
    score = cf.copy()
    for dt in range(1, W + 1):
        sig2 = 2 * (sigma0 + v * dt) ** 2
        for sgn in (-1, 1):
            a = slice(max(0, -sgn * dt), T - max(0, sgn * dt))      # frames t with a neighbour at t + sgn*dt
            b = slice(max(0, sgn * dt), T - max(0, -sgn * dt))
            d2 = ((xy[a, :, None, :] - xy[b, None, :, :]) ** 2).sum(-1)    # (T', K, K)
            score[a] += w * (cf[b, None, :] * np.exp(-d2 / sig2)).max(-1) * (cf[a] > 0)
    rows = np.arange(T)
    best = score.argmax(1)
    out_s = score[rows, best] / (1 + 2 * W * w)                         # back to a [0, 1] scale
    return frames, xy[rows, best], out_s


def fill_gaps(frames, out_xy, out_s, thr, G, vmax):
    """Frames >= thr are kept; frames between two kept ones <= G apart (plausible speed) are interpolated."""
    keep = np.flatnonzero(out_s >= thr)
    preds = {f: None for f in frames}
    for t in keep:
        preds[frames[t]] = (out_xy[t, 0], out_xy[t, 1], float(out_s[t]))
    for i, j in zip(keep[:-1], keep[1:]):
        gap = frames[j] - frames[i]
        if gap <= 1 or gap > G + 1 or np.hypot(*(out_xy[j] - out_xy[i])) / gap > vmax:
            continue
        sc = float(min(out_s[i], out_s[j]))
        for t in range(i + 1, j):
            r = (frames[t] - frames[i]) / gap
            x, y = out_xy[i] + r * (out_xy[j] - out_xy[i])
            preds[frames[t]] = (x, y, sc)
    return preds


def passed(cands, cfg):
    W, sigma0, v, w, G, vmax = cfg
    return {s: support_pass(fr, cs, W, sigma0, v, w) for s, (fr, cs) in cands.items()}


def preds_at(parts, cfg, thr):
    return {s: fill_gaps(*p, thr, cfg[4], cfg[5]) for s, p in parts.items()}


def pick(curve, min_precision=0.90):
    ok = [s for s in curve if s["precision"] >= min_precision]
    return max(ok, key=lambda s: s["recall"]) if ok else max(curve, key=lambda s: s["f1"])


def choose(cands, gt, cfg, min_precision):
    parts = passed(cands, cfg)
    return pick([base.score(preds_at(parts, cfg, float(t)), gt, float(t)) for t in THRS], min_precision)


def frozen(cands, gt, cfg, thr, seqs=None):
    return base.score(preds_at(passed(cands, cfg), cfg, thr), gt, thr, seqs=seqs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline trajectory pass over teacher candidates")
    ap.add_argument("--cands", type=Path, required=True, help="folder with select.json and test.json")
    ap.add_argument("--min_precision", type=float, default=0.90)
    ap.add_argument("--W", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--sigma0", type=float, nargs="+", default=[8.0])
    ap.add_argument("--v", type=float, nargs="+", default=[10.0, 25.0])
    ap.add_argument("--w", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--G", type=int, nargs="+", default=[0, 5, 10, 25])
    ap.add_argument("--vmax", type=float, nargs="+", default=[40.0])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    sel_c, sel_g = load(a.cands / "select.json")
    tst_c, tst_g = load(a.cands / "test.json")
    leak = set(tst_g) - {base.LEAK_SEQ}

    no_leak = set(tst_g) - {base.LEAK_SEQ}
    rows = []
    r_sel = pick([base.score(raw(sel_c), sel_g, float(t)) for t in EVAL_THRS], a.min_precision)
    rows.append(("raw top-1 (must match eval)", None, r_sel, base.score(raw(tst_c), tst_g, r_sel["thr"]),
                 base.score(raw(tst_c), tst_g, r_sel["thr"], seqs=no_leak)))

    grid = list(itertools.product(a.W, a.sigma0, a.v, a.w, a.G, a.vmax))
    print(f"{len(grid)} configurations on select ...", flush=True)
    best = None
    for cfg in grid:
        s = choose(sel_c, sel_g, cfg, a.min_precision)
        if s["precision"] >= a.min_precision and (best is None or s["recall"] > best[1]["recall"]):
            best = (cfg, s)
    if best is None:
        raise SystemExit("no configuration reaches the precision floor on select")
    cfg, s_sel = best
    W, sigma0, v, w, G, vmax = cfg
    rows.append((f"trajectory pass W={W} sigma0={sigma0} v={v} w={w} G={G} vmax={vmax}", cfg, s_sel,
                 frozen(tst_c, tst_g, cfg, s_sel["thr"]), frozen(tst_c, tst_g, cfg, s_sel["thr"], no_leak)))
    if G > 0:                                     # same knobs without gap fill: what each step gives
        c0 = (W, sigma0, v, w, 0, vmax)
        s0 = choose(sel_c, sel_g, c0, a.min_precision)
        rows.append(("support only (G=0)", c0, s0, frozen(tst_c, tst_g, c0, s0["thr"]),
                     frozen(tst_c, tst_g, c0, s0["thr"], no_leak)))

    f = lambda s: f"{s['recall']:.2%} | {s['precision']:.2%} | {s['f1']:.2%} | {s['recall_5px']:.2%}"  # noqa: E731
    L = [f"| method | thr | select R / P / F1 / R@5 | test R / P / F1 / R@5 | test w/o {base.LEAK_SEQ} R / P |",
         "|---|---|---|---|---|"]
    for name, _, ss, tt, tw in rows:
        L.append(f"| {name} | {ss['thr']:.2f} | {f(ss)} | {f(tt)} | {tw['recall']:.2%} / {tw['precision']:.2%} |")
    L += ["", "Knobs and threshold chosen on select (highest recall at precision >= "
          f"{a.min_precision:.0%}), frozen for test. Live system test: 76.29% / 90.21%. Gate: test recall >= 88% "
          "at precision >= 90%."]
    print("\n".join(L))
    out = a.out or a.cands / "trajectory_pass.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(
        [{"name": n, "cfg": c, "select": ss, "test": tt, "test_without_leak": tw} for n, c, ss, tt, tw in rows],
        indent=1))


if __name__ == "__main__":
    main()
