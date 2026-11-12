"""
phase3a_mine.py -- Phase 3a of docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md: replay the live system
(Phase 1: plausibility gate + c5c threshold 0.45) over every split, keep the specialist's top heatmap
peaks per call, mine the specialist's own failures for training, and test multi-peak selection.

Stages:
  replay     (GPU, shardable) per sequence: the Phase 1 replay, identical to phase1_sweep.simulate,
             plus per specialist call the top-K local peaks (max-pool NMS) and per frame the main
             detector's candidates. -> <out>/frames/<split>/<seq>.jsonl
  multipeak  (GPU) on select, then frozen on test: when c5c runs, accept the highest peak
             >= thr_lo inside the Phase 1 gate radius (r0 + speed * gap of the last sighting), else
             the top peak if >= hi. Grid over (thr_lo, hi) on select, objective = recall with
             precision >= Phase 1 - 1 pp; then test against Phase 1 (kill criterion +0.5 pp / -1 pp).
  report     (CPU) per-split Phase 1 numbers and miss buckets (phase0_report.analyse_seq), the mined
             set from train + pool, and the multi-peak result.
             -> report_phase3a.md, phase3a.json, mined.csv, frames.jsonl (select + test, diagnose
             format for phase0_report.py's split-integrity check)

Mined rows (train + pool only; select/test are never mined), one per example:
  hard_pos   ball inside the runtime window but c5c missed it: bucket B5 (peak < thr) or B6 (output
             > 15 px off), and B3 frames whose window (placed by a wrong sighting) still held the ball
  neg_spec   a c5c peak >= --neg_min more than --neg_dist px from the labelled ball (frames with a
             label only: there the ball is known to be elsewhere)
  neg_main   a main-detector ball >= 0.15 more than --neg_dist px from the labelled ball
Fields: split, seq, src (gsr | sntracking), t, kind, bucket, x, y (example position, full frame),
x0, y0 (runtime window, or -1), conf, gt_x, gt_y, gt_size, gap. Capped per sequence and kind
(--cap, strongest first) so one bad clip cannot dominate.

    python scripts/pipeline_improvement_1/phase3/phase3a_mine.py replay --src <roots> --cache <main cache> \
        --ckpt c5c_best.pt --out <dir> [--shard i --nshards n --device cuda:i]
    python scripts/pipeline_improvement_1/phase3/phase3a_mine.py multipeak --src ... --cache ... --ckpt ... --out <dir>
    python scripts/pipeline_improvement_1/phase3/phase3a_mine.py report --src ... --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
for p in (ROOT / "gsr_pipeline", ROOT / "scripts" / "ball_temporal", ROOT / "scripts" / "ball_eval",
          ROOT / "scripts" / "pipeline_improvement_1" / "phase1", ROOT / "scripts" / "pipeline_improvement_1" / "phase0"):
    sys.path.insert(0, str(p))
from ball_gate import BallGate, GateParams  # noqa: E402
from gsr_io import SeqReader, discover, parse_labels  # noqa: E402

GATE = GateParams(enabled=True, r0=30.0, speed=50.0, conf_hi=0.6, max_rejects=15)    # Phase 1, live
THR = 0.45
MATCH = 15.0
CROP = 640
SPLITS = ("train", "pool", "select", "test")
MP_THR_LO = (0.20, 0.25, 0.30, 0.35, 0.45)
MP_HI = (0.45, 0.55, 0.70, 1.01)


def _src_of(ref) -> str:
    return "sntracking" if ref.name.startswith("SNMOT") else "gsr"


def _d(p, gts):
    return min((float(np.hypot(p[0] - g[0], p[1] - g[1])) for g in gts), default=np.inf) if p else np.inf


# ---------------------------------------------------------------- GPU side

def make_peak_specialist(ckpt, dev, topk, peak_min, nms):
    import torch
    import torch.nn.functional as F
    from eval_ball_temporal import Specialist, build_input, decode_hm, load_model
    from heatmap_model import STRIDE

    class PeakSpecialist(Specialist):
        """Specialist whose memo also keeps the top-K local heatmap peaks of each window."""

        def __init__(self, model, channels, dev):
            super().__init__(model, channels, dev)
            self.peaks: dict = {}

        @torch.no_grad()
        def __call__(self, key, frames, x0, y0):
            if key in self.memo:
                return self.memo[key]
            x = np.concatenate([f[y0:y0 + CROP, x0:x0 + CROP, ::-1] for f in frames], axis=2)
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.dev)
            t0 = time.perf_counter()
            with torch.autocast("cuda", enabled=self.dev.type == "cuda"):
                logits, off = self.model(build_input(x, self.channels))
            p = decode_hm(logits, off)[0].tolist()
            hm = logits.float().sigmoid()[0, 0]
            S = hm.shape[-1]
            keep = (hm == F.max_pool2d(hm[None, None], nms, 1, nms // 2)[0, 0]) & (hm >= peak_min)
            conf, idx = hm[keep].flatten(), keep.flatten().nonzero()[:, 0]
            order = conf.argsort(descending=True)[:topk]
            o = off.float()[0].view(2, -1)
            pk = []
            for i in idx[order].tolist():
                iy, ix = divmod(i, S)
                pk.append((round((ix + float(o[0, i])) * STRIDE + x0, 1), round((iy + float(o[1, i])) * STRIDE + y0, 1),
                           round(float(hm.flatten()[i]), 4)))
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            self.ms += (time.perf_counter() - t0) * 1000
            self.calls += 1
            self.memo[key] = (p[0] + x0, p[1] + y0, p[2])
            self.peaks[key] = pk
            return self.memo[key]

    model, channels, epoch = load_model(ckpt, dev)
    return PeakSpecialist(model, channels, dev), epoch


def replay(sd, spec, a, mp=None) -> list[dict]:
    """phase1_sweep.simulate (Phase 1 as wired) + peaks and main candidates per frame.
    mp = (thr_lo, hi): multi-peak selection instead of the top peak."""
    from eval_ball_temporal import clamp_window
    gate = BallGate(GATE, a.max_miss)
    out, last = [], None
    for t in sorted(sd.files)[: a.max_frames or None]:
        cands = sorted((b for b in sd.dets.get(t, []) if b[4] >= a.main_conf), key=lambda b: -b[4])
        rec = dict(t=t, out=None, src="", win=None, sconf=None, gap=None, rej=None, peaks=None,
                   main=[[round(b[0], 1), round(b[1], 1), round(b[4], 3)] for b in cands])
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
            gap = t - last[2]
            rec.update(gap=gap, win=(int(x0), int(y0)))
            key = (sd.ref.name, t, x0, y0)
            px, py, conf = spec(key, None if key in spec.memo else sd.window_frames(t), x0, y0)
            rec["sconf"] = float(conf)
            rec["peaks"] = spec.peaks.get(key, [])
            pick = None
            if mp is None:
                if conf >= THR:
                    pick = (px, py)
            else:
                lo, hi = mp
                allowed = GATE.r0 + GATE.speed * max(gap, 1)
                pick = next(((x, y) for x, y, c in rec["peaks"]
                             if c >= lo and np.hypot(x - last[0], y - last[1]) <= allowed), None)
                if pick is None and conf >= hi:
                    pick = (px, py)
            if pick is not None:
                rec.update(out=pick, src="spec")
                last = (pick[0], pick[1], t)
        out.append(rec)
    return out


def _load(a, splits):
    from eval_ball_temporal import SeqData
    refs = [r for r in discover(a.src) if r.split in splits]
    if a.max_seqs:
        refs = [r for s in splits for r in [q for q in refs if q.split == s][: a.max_seqs]]
    return refs, SeqData


def stage_replay(a):
    import torch
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    spec, epoch = make_peak_specialist(a.ckpt, dev, a.topk, a.peak_min, a.nms)
    refs, SeqData = _load(a, a.splits)
    refs = [r for i, r in enumerate(refs) if i % a.nshards == a.shard]
    print(f"[shard {a.shard}] c5c epoch {epoch} on {dev}; {len(refs)} sequences", flush=True)
    t_all = time.time()
    for k, ref in enumerate(refs, 1):
        dst = a.out / "frames" / ref.split / f"{ref.name}.jsonl"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        t0, c0 = time.time(), spec.calls
        sd = SeqData(ref, a.cache)
        recs = replay(sd, spec, a)
        sd.reader.close()
        spec.memo.clear()
        spec.peaks.clear()
        tmp = dst.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for r in recs:
                r.update(split=ref.split, seq=ref.name, gt=[list(g) for g in sd.gt.get(r["t"], [])])
                f.write(json.dumps(r) + "\n")
        tmp.replace(dst)
        print(f"[shard {a.shard}] {k}/{len(refs)} {ref.split}/{ref.name}: {len(recs)} frames, "
              f"{spec.calls - c0} c5c calls, {time.time() - t0:.0f}s, elapsed {(time.time() - t_all) / 60:.1f} min",
              flush=True)


def stage_multipeak(a):
    import torch
    import phase1_sweep
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    spec, _ = make_peak_specialist(a.ckpt, dev, a.topk, a.peak_min, a.nms)
    res = {}
    for role, split in (("select", a.search_split), ("test", a.eval_split)):
        refs, SeqData = _load(a, (split,))
        sds = [SeqData(r, a.cache) for r in refs]
        base = phase1_sweep.score(sds, [replay(sd, spec, a) for sd in sds])
        floor = base["precision"] - a.prec_drop
        rows = [dict(thr_lo=None, hi=None, **base)]
        print(f"[{role}:{split}] phase1: recall {base['recall']:.2%} precision {base['precision']:.2%}", flush=True)
        grid = [(lo, hi) for lo in MP_THR_LO for hi in MP_HI if hi >= lo] if role == "select" else [res["chosen"]]
        for lo, hi in grid:
            s = phase1_sweep.score(sds, [replay(sd, spec, a, (lo, hi)) for sd in sds])
            rows.append(dict(thr_lo=lo, hi=hi, **s))
            print(f"[{role}:{split}] multipeak lo {lo:.2f} hi {hi:.2f}: recall {s['recall']:.2%} precision {s['precision']:.2%}",
                  flush=True)
        if role == "select":
            ok = [r for r in rows[1:] if r["precision"] >= floor]
            best = max(ok, key=lambda r: (r["recall"], r["precision"])) if ok else None
            res["chosen"] = (best["thr_lo"], best["hi"]) if best else (THR, THR)
        res[role] = rows
        for sd in sds:
            sd.reader.close()
    b, t = res["test"][0], res["test"][1]
    res["keep"] = t["recall"] - b["recall"] >= a.min_gain and b["precision"] - t["precision"] <= a.prec_drop
    (a.out / "multipeak.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"chosen on select {res['chosen']}; test {t['recall']:.2%} / {t['precision']:.2%} vs phase1 "
          f"{b['recall']:.2%} / {b['precision']:.2%}: {'KEEP' if res['keep'] else 'REVERT'}", flush=True)


# ---------------------------------------------------------------- CPU side

def mine_seq(rows, labels, ref, a):
    import phase0_report
    _im, _f, balls, *_ = labels
    _st, misses, _runs = phase0_report.analyse_seq(rows, labels, a.max_miss)
    bucket = {m["t"]: m["bucket"] for m in misses}
    by_t = {r["t"]: r for r in rows}
    out = defaultdict(list)
    for t, b in bucket.items():
        r = by_t[t]
        if b not in ("B3", "B5", "B6") or r["win"] is None:
            continue
        g = r["gt"][0]
        x0, y0 = r["win"]
        if not (x0 <= g[0] < x0 + CROP and y0 <= g[1] < y0 + CROP):
            continue
        size = max(balls[t][0][2], balls[t][0][3]) if balls.get(t) else 0.0
        near = max((c for x, y, c in r["peaks"] or [] if np.hypot(x - g[0], y - g[1]) <= MATCH), default=0.0)
        out["hard_pos"].append(dict(kind="hard_pos", bucket=b, t=t, x=g[0], y=g[1], x0=x0, y0=y0, conf=near,
                                    gt_x=g[0], gt_y=g[1], gt_size=size, gap=r["gap"]))
    for r in rows:
        if not r["gt"]:
            continue
        g = r["gt"][0]
        size = max(balls[r["t"]][0][2], balls[r["t"]][0][3]) if balls.get(r["t"]) else 0.0
        for x, y, c in r["peaks"] or []:
            if c >= a.neg_min and np.hypot(x - g[0], y - g[1]) > a.neg_dist:
                out["neg_spec"].append(dict(kind="neg_spec", bucket="", t=r["t"], x=x, y=y, x0=r["win"][0], y0=r["win"][1],
                                            conf=c, gt_x=g[0], gt_y=g[1], gt_size=size, gap=r["gap"]))
        for x, y, c in r["main"]:
            if np.hypot(x - g[0], y - g[1]) > a.neg_dist:
                out["neg_main"].append(dict(kind="neg_main", bucket="", t=r["t"], x=x, y=y, x0=-1, y0=-1, conf=c,
                                            gt_x=g[0], gt_y=g[1], gt_size=size, gap=""))
    rows_out = []
    for k, v in out.items():
        v.sort(key=lambda e: -e["conf"] if k != "hard_pos" else e["conf"])    # hard_pos: the worst misses first
        rows_out += [dict(split=ref.split, seq=ref.name, src=_src_of(ref), **e) for e in v[: a.cap]]
    return rows_out, misses


def stage_report(a):
    import phase0_report
    import phase1_sweep
    refs = {r.name: r for r in discover(a.src)}
    per_split, buckets, mined = defaultdict(list), defaultdict(Counter), []
    nball = Counter()
    fj = open(a.out / "frames.jsonl", "w")
    for split in SPLITS:
        for p in sorted((a.out / "frames" / split).glob("*.jsonl")):
            ref = refs[p.stem]
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            rd = SeqReader(ref)
            labels = parse_labels(rd.labels())
            rd.close()
            for r in rows:
                r["out"] = tuple(r["out"]) if r["out"] else None
            grp = f"{split}/{_src_of(ref)}" if split == "pool" else split
            per_split[grp].append((ref, rows))
            nball[grp] += sum(bool(r["gt"]) for r in rows)
            if split in a.mine_splits:
                m, misses = mine_seq(rows, labels, ref, a)
                mined += m
            else:
                _st, misses, _ = phase0_report.analyse_seq(rows, labels, a.max_miss)
                for r in rows:
                    fj.write(json.dumps(dict(split=split, seq=ref.name, t=r["t"], gt=r["gt"], out=r["out"], src=r["src"],
                                             win=r["win"], sconf=r["sconf"], gap=r["gap"],
                                             hit=float(bool(r["gt"]) and _d(r["out"], r["gt"]) <= MATCH))) + "\n")
            for m in misses:
                buckets[grp][m["bucket"]] += 1
    fj.close()

    class _SD:                       # phase1_sweep.score needs .ref.name and .gt only
        def __init__(self, ref, rows):
            self.ref, self.gt = ref, {r["t"]: [tuple(g) for g in r["gt"]] for r in rows if r["gt"]}

    summ = {}
    for grp, items in per_split.items():
        s = phase1_sweep.score([_SD(ref, rows) for ref, rows in items], [rows for _, rows in items])
        summ[grp] = dict(seqs=len(items), **s, buckets=dict(buckets[grp]))
    cols = ["split", "seq", "src", "t", "kind", "bucket", "x", "y", "x0", "y0", "conf", "gt_x", "gt_y", "gt_size", "gap"]
    with open(a.out / "mined.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(mined)
    mp = json.loads((a.out / "multipeak.json").read_text()) if (a.out / "multipeak.json").exists() else None
    mc = Counter((m["split"] + "/" + m["src"], m["kind"]) for m in mined)
    hb = Counter((m["split"] + "/" + m["src"], m["bucket"]) for m in mined if m["kind"] == "hard_pos")
    json.dump({"phase1": summ, "mined": {f"{k[0]}|{k[1]}": v for k, v in mc.items()}, "multipeak": mp},
              open(a.out / "phase3a.json", "w"), indent=1, default=float)

    pc = lambda x: f"{x:.2%}"  # noqa: E731
    L = ["# Phase 3a: Phase 1 replay over every split, mined failures, multi-peak selection", "",
         "## Phase 1 (live system) per split", "",
         "train and pool are c5c's training data, so their numbers are optimistic; select and test are held out.", "",
         "| split | seqs | ball frames | recall @15 | precision | B3 | B5 | B6 | B7 | B1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for grp in sorted(summ):
        s, bk = summ[grp], summ[grp]["buckets"]
        n = max(nball[grp], 1)
        L.append(f"| {grp} | {s['seqs']} | {nball[grp]} | {pc(s['recall'])} | {pc(s['precision'])} | "
                 + " | ".join(pc(bk.get(b, 0) / n) for b in ("B3", "B5", "B6", "B7", "B1")) + " |")
    L += ["", "Buckets as share of ball frames (phase0_report definitions).", "",
          f"## Mined set (`mined.csv`, cap {a.cap} per sequence and kind)", "",
          "| source | hard_pos | neg_spec | neg_main |", "|---|---|---|---|"]
    for grp in sorted({k[0] for k in mc}):
        L.append(f"| {grp} | " + " | ".join(str(mc.get((grp, k), 0)) for k in ("hard_pos", "neg_spec", "neg_main")) + " |")
    L += ["", "hard_pos by bucket: " + ", ".join(f"{g} {b}: {v}" for (g, b), v in sorted(hb.items())), ""]
    if mp:
        b, t = mp["test"][0], mp["test"][1]
        L += ["## Multi-peak selection", "",
              f"Chosen on select: thr_lo {mp['chosen'][0]}, hi {mp['chosen'][1]} (grid in `multipeak.json`).", "",
              "| test | recall @15 | precision | longest miss |", "|---|---|---|---|",
              f"| phase1 | {pc(b['recall'])} | {pc(b['precision'])} | {b['longest_miss_frames']} |",
              f"| multi-peak | {pc(t['recall'])} | {pc(t['precision'])} | {t['longest_miss_frames']} |", "",
              f"**Verdict: {'KEEP' if mp['keep'] else 'REVERT'}** (kill criterion +{a.min_gain:.1%} recall, "
              f"<= {a.prec_drop:.0%} precision loss)."]
    (a.out / "report_phase3a.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["replay", "multipeak", "report"])
    ap.add_argument("--src", required=True, help="GSR root(s), comma-separated (add the converted tracking clips)")
    ap.add_argument("--cache", type=Path, help="main-detector cache (replay, multipeak)")
    ap.add_argument("--ckpt", type=Path, help="c5c checkpoint (replay, multipeak)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_seqs", type=int, default=0, help="per split (smoke tests)")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--max_miss", type=int, default=15)
    ap.add_argument("--main_conf", type=float, default=0.15)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--peak_min", type=float, default=0.05)
    ap.add_argument("--nms", type=int, default=5, help="max-pool size for local peaks (heatmap cells)")
    ap.add_argument("--neg_min", type=float, default=0.2)
    ap.add_argument("--neg_dist", type=float, default=40.0)
    ap.add_argument("--mine_splits", nargs="+", default=["train", "pool"], help="never select/test except smoke tests")
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--search_split", default="select", help="multipeak: where (thr_lo, hi) is chosen")
    ap.add_argument("--eval_split", default="test", help="multipeak: frozen evaluation")
    ap.add_argument("--prec_drop", type=float, default=0.01)
    ap.add_argument("--min_gain", type=float, default=0.005)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"replay": stage_replay, "multipeak": stage_multipeak, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
