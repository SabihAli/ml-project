"""
phase0_report.py -- Phase 0 of docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md: the reproducible
baseline, the miss breakdown and the ceilings, from one replay of main + c5c as wired.

CPU only. Runs after scripts/ball_eval/diagnose_ball_misses.py (which replays
eval_ball_temporal's fused chain = BallFuser.update, on GPU) and reads its frames.jsonl, plus the
GSR labels for ball sizes, person boxes and split metadata.

    python scripts/pipeline_improvement_1/phase0/phase0_report.py \
        --src <GSR root> --diag <diagnose out> --out <phase0 out> [--frames_per_hash 25]

Outputs (<out>/):
  report_phase0.md     everything below, human-readable
  phase0.json          the same numbers, machine-readable
  buckets.csv          one row per miss: bucket, cross-flags, size, gap
  split_integrity.csv  one row per sequence: split, game_id, clip range, flags
  near_duplicates.csv  only if any near-duplicate frames were found

What it measures, per split (select, test):
  1. stages at 5/10/15 px: main alone, the specialist on the calls it is given (with crop
     containment), and the fused output. recall = hits / ball-labelled frames; precision = hits /
     outputs on ball-labelled frames (eval_ball_temporal's definitions).
  2. recall by ball size (max side, px): <10, 10-15, 15-20, 20+.
  3. temporal continuity: longest miss, miss-run p50/p95 (= recovery time), outputs on unlabelled frames.
  4. the miss breakdown (plan §5 Phase 0). Every 15 px miss gets exactly one bucket:
       B1 long_gap          no main ball, last sighting > 15 frames old: nothing searched
       B2 no_prior          no main ball and no sighting yet in the sequence
       B3 false_sighting    specialist ran, but the sighting that placed its window was wrong
       B4 outside_window    specialist ran from a correct sighting, ball outside the 640 window
       B5 below_threshold   ball inside the window, specialist peak < threshold
       B6 spec_wrong_place  specialist output >= threshold but > 15 px from the ball
       B7 main_false        main detector output on this frame, > 15 px from the ball
       B0 replay_check      recent sighting yet no specialist call: must be 0 unless --no_spec
     Cross-flags (not buckets): occluded_proxy (centre inside a person box +10%; the labels have
     no per-ball visibility, so this stands in for "hidden"), small (< 10 px).
  5. ceilings: recall if a bucket were fully fixed, and the occluded-proxy share.
  6. split integrity: sequence-name clashes, same game + overlapping clip time across splits, and
     near-duplicate frames (64-bit dHash, Hamming <= 4) between {select, test} and {train, pool}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "ball_temporal"))
from gsr_io import SeqReader, decode, discover, parse_labels  # noqa: E402

CROP = 640
FPS = 25
TOLS = (5.0, 10.0, 15.0)
MATCH = 15.0
SIZE_BUCKETS = ((0, 10, "<10"), (10, 15, "10-15"), (15, 20, "15-20"), (20, 1e9, "20+"))
BUCKETS = {
    "B0": ("replay_check", "should be 0 in a normal replay: investigate if not"),
    "B1": ("long_gap", "Phase 4 tracker / reacquisition"),
    "B2": ("no_prior", "Phase 2 detector"),
    "B3": ("false_sighting", "Phase 1 plausibility check (§3.1)"),
    "B4": ("outside_window", "Phase 1 window shift (§3.2)"),
    "B5": ("below_threshold", "Phase 3 specialist / threshold"),
    "B6": ("spec_wrong_place", "Phase 3 specialist localisation"),
    "B7": ("main_false", "Phase 1 plausibility check / Phase 2 detector"),
}
COUNT_KEYS = ("n_gt", "unlabelled_outputs", "spec_calls", "spec_calls_gt", "spec_contained", "spec_hits",
              "spec_hits_contained")


def new_agg():
    return {**{k: 0 for k in COUNT_KEYS},
            "main": {t: [0, 0] for t in TOLS}, "fused": {t: [0, 0] for t in TOLS},
            "size": defaultdict(lambda: [0, 0, 0])}   # size bucket -> [frames, main hits, fused hits]


def nearest(o, gts):
    if o is None:
        return np.inf
    return min((float(np.hypot(o[0] - g[0], o[1] - g[1])) for g in gts), default=np.inf)


def load_frames(diag: Path):
    by_seq = defaultdict(list)
    with open(diag / "frames.jsonl") as f:
        for line in f:
            r = json.loads(line)
            by_seq[(r["split"], r["seq"])].append(r)
    for v in by_seq.values():
        v.sort(key=lambda r: r["t"])
    return by_seq


def runs_of(mask):
    out, cur = [], 0
    for m in mask:
        if m:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def analyse_seq(rows, labels, max_miss):
    """One sequence of the fused replay -> counts, per-miss rows, miss-run lengths."""
    _im, _files, balls, persons, _lines, _wh = labels
    st = new_agg()
    misses, miss_mask = [], []
    last_t, last_ok = None, None   # the fused chain's last sighting, and whether it was on the ball
    for r in rows:
        t, o, src = r["t"], r["out"], r["src"]
        gts = [tuple(g) for g in r["gt"]]
        d = nearest(o, gts)
        hit = bool(gts) and d <= MATCH
        lab = balls[t][0] if balls.get(t) else None
        size = max(lab[2], lab[3]) if lab else 0.0

        if gts:
            st["n_gt"] += 1
            for tol in TOLS:
                if src == "main":
                    st["main"][tol][1] += 1
                    st["main"][tol][0] += d <= tol
                if o is not None:
                    st["fused"][tol][1] += 1
                    st["fused"][tol][0] += d <= tol
            sb = next(n for lo, hi, n in SIZE_BUCKETS if lo <= size < hi)
            s = st["size"][sb]
            s[0] += 1
            s[1] += src == "main" and hit
            s[2] += hit
        elif o is not None:
            st["unlabelled_outputs"] += 1

        ran = r["win"] is not None
        contained = None
        if ran:
            st["spec_calls"] += 1
            if gts:
                x0, y0 = r["win"]
                contained = x0 <= gts[0][0] < x0 + CROP and y0 <= gts[0][1] < y0 + CROP
                spec_hit = src == "spec" and hit
                st["spec_calls_gt"] += 1
                st["spec_contained"] += contained
                st["spec_hits"] += spec_hit
                st["spec_hits_contained"] += spec_hit and contained

        if gts:
            miss_mask.append(not hit)
        if gts and not hit:
            g = gts[0]
            if src == "main":
                b = "B7"
            elif ran:
                b = "B3" if last_ok is False else "B4" if not contained else "B6" if src == "spec" else "B5"
            elif last_t is None:
                b = "B2"
            elif t - last_t > max_miss:
                b = "B1"
            else:
                b = "B0"   # recent sighting but no specialist call: --no_spec, or replay and report disagree
            occ = any(abs(g[0] - p[0]) <= 0.55 * p[2] and abs(g[1] - p[1]) <= 0.55 * p[3] for p in persons.get(t, []))
            misses.append(dict(seq=r["seq"], t=t, bucket=b, name=BUCKETS[b][0], occluded_proxy=int(occ),
                               small=int(0 < size < 10), size=round(size, 1),
                               gap=(t - last_t) if last_t is not None else "",
                               sconf=round(r["sconf"], 3) if r["sconf"] is not None else ""))
        if o is not None:
            last_t, last_ok = t, hit
    return st, misses, runs_of(miss_mask)


def pct(a, b):
    return a / b if b else 0.0


def split_block(split, agg, misses, runs):
    n = agg["n_gt"]
    L = [f"## `{split}`: {agg['seqs']} sequences, {n} ball-labelled frames", "",
         "### Stages", "",
         "| stage | recall @5 px | @10 px | @15 px | precision @15 px |", "|---|---|---|---|---|"]
    out = {"sequences": agg["seqs"], "ball_frames": n}
    for name, label in (("main", "main detector alone"), ("fused", "main + c5c (fused, as wired)")):
        rec = [pct(agg[name][t][0], n) for t in TOLS]
        prec = pct(agg[name][MATCH][0], agg[name][MATCH][1])
        out[name] = {"recall": {f"{t:g}": v for t, v in zip(TOLS, rec)}, "precision_15": prec}
        L.append(f"| {label} | {rec[0]:.1%} | {rec[1]:.1%} | {rec[2]:.1%} | {prec:.1%} |")
    sc = agg["spec_calls_gt"]
    spec = {"calls": agg["spec_calls"], "calls_on_ball_frames": sc,
            "containment": pct(agg["spec_contained"], sc), "recall_on_calls": pct(agg["spec_hits"], sc),
            "recall_when_contained": pct(agg["spec_hits_contained"], agg["spec_contained"])}
    out["specialist"] = spec
    out["unlabelled_outputs"] = agg["unlabelled_outputs"]
    L += ["", f"Specialist: {spec['calls']} calls, {sc} on ball-labelled frames. Ball inside the window "
              f"(crop containment): {spec['containment']:.1%}. Recall on its calls: {spec['recall_on_calls']:.1%}; "
              f"when the ball was inside the window: {spec['recall_when_contained']:.1%}. "
              f"Outputs on frames with no ball label: {agg['unlabelled_outputs']} (may be unlabelled balls).", "",
          "### Recall by ball size (max side, px; 15 px match)", "",
          "| size | frames | main | fused |", "|---|---|---|---|"]
    out["by_size"] = {}
    for _lo, _hi, nm in SIZE_BUCKETS:
        s = agg["size"].get(nm, [0, 0, 0])
        out["by_size"][nm] = {"frames": s[0], "main": pct(s[1], s[0]), "fused": pct(s[2], s[0])}
        L.append(f"| {nm} | {s[0]} | {pct(s[1], s[0]):.1%} | {pct(s[2], s[0]):.1%} |")
    r = np.array(runs) if runs else np.array([0])
    cont = {"miss_runs": len(runs), "longest_frames": int(r.max()), "longest_s": float(r.max()) / FPS,
            "run_p50": float(np.percentile(r, 50)), "run_p95": float(np.percentile(r, 95))}
    out["continuity"] = cont
    L += ["", "### Temporal continuity", "",
          f"{cont['miss_runs']} miss runs. Longest: {cont['longest_frames']} frames ({cont['longest_s']:.1f} s). "
          f"Run length (frames to recover): p50 {cont['run_p50']:.0f}, p95 {cont['run_p95']:.0f}.", "",
          "### Miss breakdown (15 px)", "",
          "| bucket | misses | share of misses | share of ball frames | occluded-proxy | < 10 px | addressed by |",
          "|---|---|---|---|---|---|---|"]
    c = Counter(m["bucket"] for m in misses)
    occ = Counter(m["bucket"] for m in misses if m["occluded_proxy"])
    sml = Counter(m["bucket"] for m in misses if m["small"])
    nm_ = max(len(misses), 1)
    out["buckets"] = {}
    for b, (name, by) in BUCKETS.items():
        out["buckets"][b] = {"name": name, "misses": c[b], "share_of_misses": c[b] / nm_,
                             "share_of_ball_frames": pct(c[b], n), "occluded_proxy": occ[b], "small": sml[b]}
        L.append(f"| {b} {name} | {c[b]} | {c[b] / nm_:.1%} | {pct(c[b], n):.1%} | {occ[b]} | {sml[b]} | {by} |")
    occ_all = sum(m["occluded_proxy"] for m in misses)
    L.append(f"| **total** | {len(misses)} | 100% | {pct(len(misses), n):.1%} | {occ_all} | "
             f"{sum(m['small'] for m in misses)} | |")
    fused = out["fused"]["recall"]["15"]
    ceil = {b: fused + pct(c[b], n) for b in BUCKETS}
    ceil["phase1_B3_B4_B7"] = fused + pct(c["B3"] + c["B4"] + c["B7"], n)
    ceil["all_but_occluded_proxy"] = 1.0 - pct(occ_all, n)
    out["ceilings"] = ceil
    out["occluded_proxy_share"] = pct(occ_all, n)
    L += ["", "### Ceilings (recall @15 px if that bucket were fully fixed)", "",
          "| fix | recall |", "|---|---|", f"| now | {fused:.1%} |"]
    for b, (name, _) in BUCKETS.items():
        L.append(f"| {b} {name} | {ceil[b]:.1%} |")
    L += [f"| B3 + B4 + B7 (Phase 1, no new model) | {ceil['phase1_B3_B4_B7']:.1%} |",
          f"| every miss except the occluded-proxy ones | {ceil['all_but_occluded_proxy']:.1%} |", "",
          f"Occluded-proxy misses are {pct(occ_all, n):.1%} of ball frames. This is a proxy for \"hidden\", "
          "not a visibility label: some are visible balls in front of a player. The plan's ~10% rule (§6) "
          "should be applied only after the audit sheets confirm how many are truly hidden.", ""]
    return L, out


def dhash(img, size=8):
    import cv2
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size), interpolation=cv2.INTER_AREA)
    v = 0
    for bit in (g[:, 1:] > g[:, :-1]).flatten():
        v = (v << 1) | int(bit)
    return v


def split_integrity(refs, every):
    rows, hashes = [], defaultdict(list)
    for ref in refs:
        rd = SeqReader(ref)
        try:
            d = rd.labels()
            info = d.get("info", {})
            im_dir, files, *_ = parse_labels(d)
            rows.append(dict(seq=ref.name, src_split=ref.src_split, split=ref.split,
                             game_id=str(info.get("game_id", "")),
                             clip_start=int(info.get("clip_start") or -1), clip_stop=int(info.get("clip_stop") or -1),
                             game_time_start=info.get("game_time_start", ""), flags=""))
            if every:
                for f in sorted(files)[::every]:
                    try:
                        hashes[ref.split].append((ref.name, f, dhash(decode(rd.read(f"{im_dir}/{files[f]}")))))
                    except Exception:
                        pass
        finally:
            rd.close()
    findings = []
    for h in (r for r in rows if r["split"] in ("select", "test")):
        overl = [o for o in rows if o is not h and o["game_id"] and o["game_id"] == h["game_id"]
                 and o["src_split"] == h["src_split"] and o["split"] != h["split"]
                 and o["clip_start"] < h["clip_stop"] and h["clip_start"] < o["clip_stop"]]
        if overl:
            h["flags"] += "clip_overlap:" + "|".join(o["seq"] for o in overl) + ";"
            findings.append(f"{h['seq']} ({h['split']}) overlaps in time with "
                            + ", ".join(f"{o['seq']} ({o['split']})" for o in overl))
    dup = []
    ref_h = [x for s in ("train", "pool") for x in hashes.get(s, [])]
    if ref_h:
        ref_v = np.array([v for _, _, v in ref_h], dtype=np.uint64)
        for s in ("select", "test"):
            for n, f, v in hashes.get(s, []):
                x = np.bitwise_xor(ref_v, np.uint64(v))
                ham = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
                j = int(ham.argmin())
                if ham[j] <= 4:
                    dup.append(dict(split=s, seq=n, frame=f, match_seq=ref_h[j][0], match_frame=ref_h[j][1],
                                    hamming=int(ham[j])))
    names = {s: sorted(r["seq"] for r in rows if r["split"] == s) for s in ("train", "pool", "select", "test")}
    clash = sorted(set(names["test"]) & (set(names["train"]) | set(names["pool"]) | set(names["select"])))
    return rows, findings, dup, names, clash


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--diag", type=Path, required=True, help="diagnose_ball_misses.py --out")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["select", "test"])
    ap.add_argument("--max_miss", type=int, default=15)
    ap.add_argument("--frames_per_hash", type=int, default=25, help="hash every Nth frame for near-duplicates (0 = off)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    refs = discover(a.src)
    by_name = {r.name: r for r in refs}
    frames = load_frames(a.diag)
    dpath = a.diag / "diagnosis.json"
    diag_meta = json.loads(dpath.read_text()) if dpath.exists() else {}
    report = ["# Phase 0 baseline (PIPELINE_IMPROVEMENT_PLAN_1.md)", "",
              f"Replay: `{a.diag}` (diagnose_ball_misses.py, fused chain = BallFuser.update). Main output "
              "conf >= 0.15, c5c threshold 0.70, specialist up to 15 frames after the last sighting.", ""]
    data, all_misses = {"splits": {}}, []
    for split in a.splits:
        keys = sorted(k for k in frames if k[0] == split)
        agg = new_agg()
        agg["seqs"] = len(keys)
        misses_s, runs_s = [], []
        for k in keys:
            rd = SeqReader(by_name[k[1]])
            labels = parse_labels(rd.labels())
            rd.close()
            st, ms, rl = analyse_seq(frames[k], labels, a.max_miss)
            for key in COUNT_KEYS:
                agg[key] += st[key]
            for nm in ("main", "fused"):
                for t in TOLS:
                    agg[nm][t][0] += st[nm][t][0]
                    agg[nm][t][1] += st[nm][t][1]
            for b, v in st["size"].items():
                for i in range(3):
                    agg["size"][b][i] += v[i]
            for m in ms:
                m["split"] = split
            misses_s += ms
            runs_s += rl
        L, out = split_block(split, agg, misses_s, runs_s)
        d_rec = diag_meta.get("splits", {}).get(split, {}).get("recall")
        if d_rec is not None and abs(d_rec - out["fused"]["recall"]["15"]) > 1e-6:
            L.append(f"**WARNING:** fused recall differs from diagnosis.json ({d_rec:.4f}).")
        report += L
        data["splits"][split] = out
        all_misses += misses_s
        print(f"[phase0] {split}: fused recall {out['fused']['recall']['15']:.2%}, {len(misses_s)} misses", flush=True)

    rows, findings, dup, names, clash = split_integrity(refs, a.frames_per_hash)
    pairs = sorted({(d["seq"], d["match_seq"]) for d in dup})
    data["split_integrity"] = {"counts": {s: len(v) for s, v in names.items()}, "test_name_clash": clash,
                               "clip_overlaps": findings, "near_duplicate_frames": len(dup),
                               "near_duplicate_pairs": pairs}
    report += ["## Split integrity", "",
               "Sequences per split: " + ", ".join(f"{s} {len(v)}" for s, v in names.items()) + ".", "",
               f"Test sequence names also in another split: {clash or 'none'}.", "",
               "Same game and overlapping clip time across splits (compared within one GSR source split, since "
               "`game_id` is not comparable across GSR train/valid): " + ("; ".join(findings) or "none") + ".", "",
               f"Near-duplicate frames (dHash Hamming <= 4, every {a.frames_per_hash}th frame) between select/test "
               f"and train/pool: {len(dup)}" + (" -- pairs: " + ", ".join(f"{s}~{m}" for s, m in pairs[:20]) if dup else "")
               + ".", "",
               "Static frames (black, scoreboard-only) can match by hash without any leak; inspect a pair before "
               "calling it one.", ""]

    (a.out / "report_phase0.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (a.out / "phase0.json").write_text(json.dumps(data, indent=1, default=float))
    if all_misses:
        with open(a.out / "buckets.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_misses[0]))
            w.writeheader()
            w.writerows(all_misses)
    with open(a.out / "split_integrity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["seq"])
        w.writeheader()
        w.writerows(rows)
    if dup:
        with open(a.out / "near_duplicates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dup[0]))
            w.writeheader()
            w.writerows(dup)
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
