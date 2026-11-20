"""
decode_trajectory.py -- Phase 3b.2 step 1: global trajectory decoding over a teacher's per-frame candidates
(docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md, Phase 3b.2). CPU only. Same input and scoring as trajectory_pass.py.

trajectory_pass.py scores each candidate by its neighbours alone, so a motion ghost (the ball's position at t+-1,
which the motion channels fire on) gets as much support as the ball. Here each clip is decoded as ONE path:

  DP over nodes (frame t, candidate i), at most one node per frame, frames increasing.
    gain(t, i)       = conf - b                              (b: a node must beat this to be worth taking)
    edge (t,i)->(t',j), 1 <= t'-t <= M frames, speed <= vmax:
                     = -lam * (d / (sigma0 + v * dt))^2 - gamma * (dt - 1)
    reset            : start a new segment after any earlier node at cost rho (ball out of view, cut, kick)
  best(t, i) = gain + max(0, max_edges(best(t', j) + edge), bestUpTo(t-1) - rho); traced back from the best end.

Per kept node the output score = mean conf of the path's nodes within +-S frames (missing frames count 0), so an
isolated node scores low; frames between two path nodes <= G apart are interpolated (score = lower endpoint).
Knobs and threshold are chosen on select (highest recall at precision >= 90%), frozen for test.

    python scripts/pipeline_improvement_1/phase3b/decode_trajectory.py --cands data/outputs/pipeline_improvement_1/phase3b/teacher_yolo/candidates/fuse_t2m_s1920_ft+t1_r34_a0.7
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_teacher_yolo as base  # noqa: E402
import trajectory_pass as tp  # noqa: E402


def arrays(frames, cs):
    T, K = len(frames), max((len(c) for c in cs), default=0)
    xy, cf = np.zeros((T, max(K, 1), 2)), np.zeros((T, max(K, 1)))
    for t, c in enumerate(cs):
        for k, (x, y, p) in enumerate(c):
            xy[t, k], cf[t, k] = (x, y), p
    return np.asarray(frames), xy, cf


def decode(frames, xy, cf, b, lam, sigma0, v, vmax, M, gamma, rho):
    """Path nodes [(t, k)] of the best single path (resets allowed)."""
    T, K = cf.shape
    valid = cf > 0
    gain = np.where(valid, cf - b, -np.inf)
    best = np.full((T, K), -np.inf)
    back = np.full((T, K, 2), -1, dtype=np.int64)          # predecessor (t, k); (-1, -1) = start / reset
    upto = np.full(T, -np.inf)                             # best value of any node at frames <= t
    upto_arg = np.full((T, 2), -1, dtype=np.int64)
    for t in range(T):
        cur = np.zeros(K)                                  # start fresh
        arg = np.full((K, 2), -1, dtype=np.int64)
        if t > 0 and upto[t - 1] - rho > 0:                # reset after an earlier path
            cur[:] = upto[t - 1] - rho
            arg[:] = upto_arg[t - 1]
        for s in range(max(0, t - M), t):
            dt = frames[t] - frames[s]
            if dt > M:
                continue
            d = np.linalg.norm(xy[t][:, None, :] - xy[s][None, :, :], axis=-1)       # (K_t, K_s)
            e = -lam * (d / (sigma0 + v * dt)) ** 2 - gamma * (dt - 1)
            e[d / dt > vmax] = -np.inf
            cand = best[s][None, :] + e
            j = cand.argmax(1)
            val = cand[np.arange(K), j]
            upd = val > cur
            cur[upd] = val[upd]
            arg[upd, 0], arg[upd, 1] = s, j[upd]
        best[t] = np.where(valid[t], gain[t] + cur, -np.inf)
        back[t] = arg
        k = int(best[t].argmax())
        if t > 0 and upto[t - 1] >= best[t, k]:
            upto[t], upto_arg[t] = upto[t - 1], upto_arg[t - 1]
        else:
            upto[t], upto_arg[t] = best[t, k], (t, k)
    path, node = [], tuple(upto_arg[T - 1]) if T else (-1, -1)
    while node[0] >= 0 and np.isfinite(best[node]):
        path.append(node)
        node = tuple(back[node])
    return path[::-1]


def outputs(frames, xy, cf, path, S, G, thr):
    """{frame: (x, y, score) | None}: path nodes scored by local path support, gaps filled."""
    preds = {int(f): None for f in frames}
    if not path:
        return preds
    ts = np.array([t for t, _ in path])
    pf = frames[ts]
    pc = np.array([cf[t, k] for t, k in path])
    pxy = np.array([xy[t, k] for t, k in path])
    sc = np.array([pc[np.abs(pf - f) <= S].sum() / (2 * S + 1) for f in pf])
    keep = np.flatnonzero(sc >= thr)
    for i in keep:
        preds[int(pf[i])] = (pxy[i, 0], pxy[i, 1], float(sc[i]))
    fr_index = {int(f): t for t, f in enumerate(frames)}
    for i, j in zip(keep[:-1], keep[1:]):
        gap = pf[j] - pf[i]
        if gap <= 1 or gap > G + 1:
            continue
        s = float(min(sc[i], sc[j]))
        for f in range(int(pf[i]) + 1, int(pf[j])):
            if f in fr_index and preds[f] is None:
                r = (f - pf[i]) / gap
                x, y = pxy[i] + r * (pxy[j] - pxy[i])
                preds[f] = (x, y, s)
    return preds


def run(arrs, cfg, thr):
    b, lam, sigma0, v, vmax, M, gamma, rho, S, G = cfg
    return {s: outputs(fr, xy, cf, decode(fr, xy, cf, b, lam, sigma0, v, vmax, M, gamma, rho), S, G, thr)
            for s, (fr, xy, cf) in arrs.items()}


def choose(arrs, gt, cfg, min_precision):
    b, lam, sigma0, v, vmax, M, gamma, rho, S, G = cfg
    paths = {s: (fr, xy, cf, decode(fr, xy, cf, b, lam, sigma0, v, vmax, M, gamma, rho)) for s, (fr, xy, cf) in arrs.items()}
    curve = [base.score({s: outputs(fr, xy, cf, p, S, G, float(t)) for s, (fr, xy, cf, p) in paths.items()}, gt, float(t))
             for t in tp.THRS]
    return tp.pick(curve, min_precision)


def main() -> None:
    ap = argparse.ArgumentParser(description="Global trajectory decoding over teacher candidates")
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--min_precision", type=float, default=0.90)
    ap.add_argument("--b", type=float, nargs="+", default=[0.0, 0.02])
    ap.add_argument("--lam", type=float, nargs="+", default=[0.05, 0.2])
    ap.add_argument("--sigma0", type=float, nargs="+", default=[8.0])
    ap.add_argument("--v", type=float, nargs="+", default=[25.0])
    ap.add_argument("--vmax", type=float, nargs="+", default=[60.0])
    ap.add_argument("--M", type=int, nargs="+", default=[10])
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.0, 0.01])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.1, 0.5])
    ap.add_argument("--S", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--G", type=int, nargs="+", default=[10])
    ap.add_argument("--select_only", action="store_true", help="do not read test (knob search only)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    sel_c, sel_g = tp.load(a.cands / "select.json")
    sel = {s: arrays(fr, cs) for s, (fr, cs) in sel_c.items()}
    grid = list(itertools.product(a.b, a.lam, a.sigma0, a.v, a.vmax, a.M, a.gamma, a.rho, a.S, a.G))
    print(f"{len(grid)} configurations on select ...", flush=True)
    best, t0 = None, time.time()
    for cfg in grid:
        s = choose(sel, sel_g, cfg, a.min_precision)
        if s["precision"] >= a.min_precision and (best is None or s["recall"] > best[1]["recall"]):
            best = (cfg, s)
            print(f"  {cfg}: select {s['recall']:.2%} / {s['precision']:.2%} thr {s['thr']}", flush=True)
    print(f"search {time.time() - t0:.0f} s", flush=True)
    if best is None:
        raise SystemExit("no configuration reaches the precision floor on select")
    cfg, s_sel = best
    names = ("b", "lam", "sigma0", "v", "vmax", "M", "gamma", "rho", "S", "G")
    knobs = " ".join(f"{n}={c}" for n, c in zip(names, cfg))
    f = lambda s: f"{s['recall']:.2%} | {s['precision']:.2%} | {s['f1']:.2%} | {s['recall_5px']:.2%}"  # noqa: E731
    L = [f"Global decode, knobs chosen on select: {knobs}, thr {s_sel['thr']:.3f}",
         "", "| split | recall | precision | F1 | recall @5 px |", "|---|---|---|---|---|", f"| select | {f(s_sel)} |"]
    res = {"cfg": dict(zip(names, cfg)), "select": s_sel}
    if not a.select_only:
        tst_c, tst_g = tp.load(a.cands / "test.json")
        tst = {s: arrays(fr, cs) for s, (fr, cs) in tst_c.items()}
        preds = run(tst, cfg, s_sel["thr"])
        no_leak = set(tst_g) - {base.LEAK_SEQ}
        t_all, t_nl = base.score(preds, tst_g, s_sel["thr"]), base.score(preds, tst_g, s_sel["thr"], seqs=no_leak)
        L += [f"| test | {f(t_all)} |", f"| test w/o {base.LEAK_SEQ} | {f(t_nl)} |"]
        res.update(test=t_all, test_without_leak=t_nl)
    print("\n".join(L))
    out = a.out or a.cands / "decode_trajectory.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
