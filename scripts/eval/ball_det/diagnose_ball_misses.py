"""
diagnose_ball_misses.py -- item 1 of docs/plans/BALL_IMPROVEMENT_PLAN.md (§2): where do the
balls that main detector + c5c miss go, and what could each later item recover?

Standalone apart from scripts/ball_temporal/ and eval_ball_temporal.py. Needs torch (the
specialist replay), numpy, opencv-python. Reads the main-detector cache and the camera-motion
cache (camera_motion_cache.py).

  run     replay main + c5c exactly as wired (eval_ball_temporal's fused chain, threshold
          --system_thr) on `select` and `test`, recording every frame's output, every specialist
          call (window, raw confidence), then tag misses, measure displacement, estimate the
          ceiling, and write the audit sheets. GPU recommended; --no_spec replays main only.

Outputs (<out>/):
  frames.jsonl            one row per frame: split, seq, t, gt, out, src, spec window/conf, gap
  misses.csv              one row per miss: every cause flag, the primary cause, ball size/speed
  displacement.csv        per gap 1-25: p50/p90/p99 ball displacement, raw and camera-compensated
  report_diagnosis.md     cause tables (select, test), displacement, ceiling per item, FP analysis
  diagnosis.json          the same numbers, machine-readable (the orchestrator reads it)
  audit/page_XX.jpg       200 sampled test misses, 20 per page, causes drawn on
  audit/audit.csv         the same 200, with an empty `human_cause` column for the manual pass

Cause rules (a miss = ball-labelled frame with no output within 15 px), primary = first match in:
  near_miss       an output within 15-30 px of a GT ball
  low_conf        a cached main ball within 15 px at conf in [cache floor, --main_conf)
  occluded        GT ball centre inside a GT person box expanded by 10%
  outside_window  the specialist ran but the GT ball was outside its 640x640 window
  not_attempted   the specialist didn't run (no sighting yet, or gap > max_miss)
  fast            GT ball moved > 25 px since the previous frame (image pixels: blur)
  small           GT ball box max side < 10 px
  static_restart  GT ball inside a run of >= 25 frames moving < 1 px/frame (kickoff, set pieces)
  detector_blind  none of the above
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT / "scripts" / "ball_temporal", HERE):
    sys.path.insert(0, str(p))
import camera_motion_cache as cmc  # noqa: E402
from build_ball_temporal_dataset import CROP, clamp_window  # noqa: E402
from gsr_io import discover, parse_labels  # noqa: E402

MATCH_PX, NEAR_PX = 15.0, 30.0
FAST_PX, SMALL_PX = 25.0, 10.0
STATIC_PX, STATIC_RUN = 1.0, 25
CAUSES = ("near_miss", "low_conf", "occluded", "outside_window", "not_attempted", "fast", "small",
          "static_restart", "detector_blind")
ITEM_OF = {"low_conf": "item 2 (tracker)", "not_attempted": "item 2 (tracker)",
           "outside_window": "item 3 (c5c windows)", "fast": "item 4 (detector)", "small": "item 4 (detector)",
           "static_restart": "item 4 (detector)", "detector_blind": "item 4 (detector)",
           "near_miss": "localisation", "occluded": "batch interpolation only"}


def replay(sd, spec, thr, a):
    """eval_ball_temporal.simulate (fused chain), recording the specialist's window and raw conf."""
    out, last = [], None
    for t in sorted(sd.files):
        m = [b for b in sd.dets.get(t, []) if b[4] >= a.main_conf]
        if m:
            last = (m[0][0], m[0][1], t)
            out.append(dict(t=t, x=m[0][0], y=m[0][1], src="main", gap=None, win=None, sconf=None))
            continue
        rec = dict(t=t, x=None, y=None, src="", gap=None, win=None, sconf=None)
        if spec is not None and last is not None and t - last[2] <= a.max_miss:
            x0, y0, _ = clamp_window(last[0], last[1], sd.W, sd.H)
            rec["gap"] = t - last[2]
            key = (sd.ref.name, t, x0, y0)
            frames = None if key in spec.memo else sd.window_frames(t)
            px, py, conf = spec(key, frames, x0, y0)
            rec.update(win=(x0, y0), sconf=float(conf))
            if conf >= thr:
                rec.update(x=px, y=py, src="spec")
                last = (px, py, t)
        out.append(rec)
    return out


def _static_frames(gt_pos: dict) -> set:
    ts = sorted(gt_pos)
    still = [t for t, u in zip(ts[1:], ts) if t == u + 1 and
             np.hypot(gt_pos[t][0] - gt_pos[u][0], gt_pos[t][1] - gt_pos[u][1]) < STATIC_PX]
    runs, cur = set(), []
    for t in still:
        if cur and t != cur[-1] + 1:
            if len(cur) >= STATIC_RUN:
                runs.update(cur)
            cur = []
        cur.append(t)
    if len(cur) >= STATIC_RUN:
        runs.update(cur)
    return runs


def _off_pitch(p, line_pts) -> bool:
    if len(line_pts) < 3:
        return False
    import cv2
    hull = cv2.convexHull(np.array(line_pts, np.float32))
    return cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) < 0


def _displacements(gt_pos, motion, max_gap=25):
    """{gap: ([raw], [camera-compensated])} over pairs of labelled frames."""
    out = defaultdict(lambda: ([], []))
    for t in sorted(gt_pos):
        m = np.eye(3)
        ok = True
        for g in range(1, max_gap + 1):
            u = t + g
            a = motion.get(u) if motion else None
            if a is None:
                ok = False
            elif ok:
                m = np.vstack([a, [0, 0, 1]]) @ m
            if u not in gt_pos:
                continue
            p, q = gt_pos[t], gt_pos[u]
            out[g][0].append(float(np.hypot(q[0] - p[0], q[1] - p[1])))
            if ok:
                px = m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2]
                py = m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2]
                out[g][1].append(float(np.hypot(q[0] - px, q[1] - py)))
    return out


def analyse_sequence(split, sd, out, labels, motion, cache_floor, main_conf):
    _im, _files, balls, persons, lines, _wh = labels
    gt_pos = {t: (bs[0][0], bs[0][1]) for t, bs in balls.items() if bs}
    static = _static_frames(gt_pos)
    frames, misses, fps_ = [], [], []
    for r in out:
        t = r["t"]
        gts = balls.get(t, [])
        o = (r["x"], r["y"]) if r["x"] is not None else None
        d_out = min((np.hypot(o[0] - g[0], o[1] - g[1]) for g in gts), default=np.inf) if o else np.inf
        hit = bool(gts) and d_out <= MATCH_PX
        frames.append(dict(split=split, seq=sd.ref.name, t=t, gt=[(g[0], g[1]) for g in gts], out=o, src=r["src"],
                           win=r["win"], sconf=r["sconf"], gap=r["gap"], hit=hit))
        if gts and o is not None and not hit:
            fps_.append(dict(dist=float(d_out), offpitch=_off_pitch(o, lines.get(t, [])), src=r["src"]))
        if not gts or hit:
            continue
        g = gts[0]
        flags = {c: False for c in CAUSES}
        flags["near_miss"] = d_out <= NEAR_PX
        flags["low_conf"] = any(cache_floor <= b[4] < main_conf and np.hypot(b[0] - g[0], b[1] - g[1]) <= MATCH_PX
                                for b in sd.dets.get(t, []))
        flags["occluded"] = any(abs(g[0] - p[0]) <= 0.55 * p[2] and abs(g[1] - p[1]) <= 0.55 * p[3]
                                for p in persons.get(t, []))
        if r["win"] is not None:
            x0, y0 = r["win"]
            flags["outside_window"] = not (x0 <= g[0] < x0 + CROP and y0 <= g[1] < y0 + CROP)
        else:
            flags["not_attempted"] = True
        prev = gt_pos.get(t - 1)
        speed = float(np.hypot(g[0] - prev[0], g[1] - prev[1])) if prev else 0.0
        flags["fast"] = speed > FAST_PX
        flags["small"] = max(g[2], g[3]) < SMALL_PX
        flags["static_restart"] = t in static
        primary = next((c for c in CAUSES[:-1] if flags[c]), "detector_blind")
        flags["detector_blind"] = primary == "detector_blind"
        misses.append(dict(split=split, seq=sd.ref.name, t=t, primary=primary, gx=round(g[0], 1), gy=round(g[1], 1),
                           size=round(max(g[2], g[3]), 1), speed=round(speed, 1), gap=r["gap"] if r["gap"] else "",
                           sconf=round(r["sconf"], 3) if r["sconf"] is not None else "",
                           **{c: int(v) for c, v in flags.items()}))
    return frames, misses, fps_, _displacements(gt_pos, motion)


def cause_table(misses, n_gt, recall):
    c = Counter(m["primary"] for m in misses)
    flag = {k: sum(m[k] for m in misses) for k in CAUSES}
    n = max(len(misses), 1)
    rows = ["| primary cause | misses | share of misses | share of ball frames | flagged (any) | addressed by |",
            "|---|---|---|---|---|---|"]
    for k in CAUSES:
        rows.append(f"| {k} | {c[k]} | {c[k] / n:.1%} | {c[k] / max(n_gt, 1):.1%} | {flag[k]} | {ITEM_OF[k]} |")
    rows.append(f"| **total** | {len(misses)} | 100% | {len(misses) / max(n_gt, 1):.1%} (recall {recall:.1%}) | | |")
    return rows, dict(c)


def ceiling(primary_counts, n_gt, recall):
    share = {k: v / max(n_gt, 1) for k, v in primary_counts.items()}
    it2 = share.get("low_conf", 0) + share.get("not_attempted", 0)
    it3 = share.get("outside_window", 0)
    it4 = sum(share.get(k, 0) for k in ("fast", "small", "static_restart", "detector_blind"))
    occ = share.get("occluded", 0)
    loc = share.get("near_miss", 0)
    return {"now": recall, "item2": it2, "item3": it3, "item4": it4, "occluded": occ, "near_miss": loc,
            "live_ceiling": recall + it2 + it3 + it4 + loc, "batch_ceiling": recall + it2 + it3 + it4 + loc + occ}


def audit_sheets(misses, sds_by_name, out_dir: Path, frames_by_key, n=200, seed=0, per_page=20):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = [m for m in misses if m["split"] == "test"]
    if not pool:
        return 0
    rng = np.random.default_rng(seed)
    pick = [pool[i] for i in sorted(rng.choice(len(pool), min(n, len(pool)), replace=False))]
    tiles = []
    with open(out_dir / "audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["page", "slot", "seq", "t", "primary", "flags", "human_cause", "notes"])
        w.writeheader()
        for i, m in enumerate(pick):
            sd = sds_by_name[m["seq"]]
            img = sd.frame(m["t"]).copy()
            x0, y0, _ = clamp_window(m["gx"], m["gy"], sd.W, sd.H)
            fr = frames_by_key[(m["seq"], m["t"])]
            if fr["win"] is not None:
                wx, wy = fr["win"]
                cv2.rectangle(img, (int(wx), int(wy)), (int(wx) + CROP, int(wy) + CROP), (0, 255, 255), 3)
            cv2.circle(img, (int(m["gx"]), int(m["gy"])), int(max(m["size"], 8)) + 6, (0, 255, 0), 2)
            if fr["out"] is not None:
                cv2.drawMarker(img, (int(fr["out"][0]), int(fr["out"][1])), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            crop = img[y0:y0 + CROP, x0:x0 + CROP].copy()
            flags = ",".join(k for k in CAUSES if m[k])
            for txt, yy in ((f"#{i} {m['seq']} f{m['t']} {m['primary']}", 30), (flags[:60], 62)):
                cv2.putText(crop, txt, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(crop, txt, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(cv2.resize(crop, (320, 320), interpolation=cv2.INTER_AREA))
            w.writerow(dict(page=i // per_page, slot=i % per_page, seq=m["seq"], t=m["t"], primary=m["primary"],
                            flags=flags, human_cause="", notes=""))
    for p in range(0, len(tiles), per_page):
        page = tiles[p:p + per_page]
        page += [np.zeros_like(tiles[0])] * (per_page - len(page))
        grid = np.concatenate([np.concatenate(page[r * 5:(r + 1) * 5], axis=1) for r in range(per_page // 5)], axis=0)
        cv2.imwrite(str(out_dir / f"page_{p // per_page:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return len(pick)


def stage_run(a):
    import torch
    from eval_ball_temporal import SeqData, Specialist, load_model
    spec = None
    if not a.no_spec:
        model, channels, epoch = load_model(a.ckpt, torch.device(a.device))
        spec = Specialist(model, channels, torch.device(a.device))
        print(f"[diagnose] specialist {a.ckpt} (epoch {epoch}), threshold {a.system_thr}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    report, data = [], {"splits": {}}
    all_frames, all_misses, sds_by_name, frames_by_key = [], [], {}, {}
    disp_all = defaultdict(lambda: ([], []))
    for split in a.splits:
        refs = [r for r in discover(a.src) if r.split == split]
        if a.max_seqs:
            refs = refs[:a.max_seqs]
        misses_s, fps_s, n_gt, hits = [], [], 0, 0
        t0 = time.time()
        for i, ref in enumerate(refs, 1):
            sd = SeqData(ref, a.cache)
            meta = json.loads((a.cache / ref.split / f"{ref.name}.json").read_text()).get("meta", {})
            cache_floor = float(meta.get("conf", 0.05))
            has_motion = a.motion and (a.motion / ref.split / f"{ref.name}.json").exists()
            motion = cmc.load(a.motion, ref) if has_motion else {}
            out = replay(sd, spec, a.system_thr, a)
            labels = parse_labels(sd.reader.labels())
            frames, misses, fps_, disp = analyse_sequence(split, sd, out, labels, motion, cache_floor, a.main_conf)
            for g, (raw, comp) in disp.items():
                disp_all[g][0].extend(raw)
                disp_all[g][1].extend(comp)
            all_frames += frames
            misses_s += misses
            fps_s += fps_
            sds_by_name[ref.name] = sd
            for fr in frames:
                frames_by_key[(fr["seq"], fr["t"])] = fr
            seq_gt = sum(bool(f["gt"]) for f in frames)
            seq_hit = sum(f["hit"] for f in frames)
            n_gt, hits = n_gt + seq_gt, hits + seq_hit
            print(f"[diagnose {split} {i}/{len(refs)}] {ref.name}: recall {seq_hit / max(seq_gt, 1):.1%}, "
                  f"{len(misses)} misses, top causes {Counter(m['primary'] for m in misses).most_common(3)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        recall = hits / max(n_gt, 1)
        rows, prim = cause_table(misses_s, n_gt, recall)
        ceil = ceiling(prim, n_gt, recall)
        offp = sum(f["offpitch"] for f in fps_s)
        fp_med = float(np.median([f["dist"] for f in fps_s])) if fps_s else None
        data["splits"][split] = {"sequences": len(refs), "ball_frames": n_gt, "recall": recall,
                                 "misses": len(misses_s), "primary": prim, "ceiling": ceil,
                                 "false_positives": len(fps_s), "fp_offpitch": offp, "fp_median_dist": fp_med}
        report += [f"## `{split}`: {len(refs)} sequences, {n_gt} ball frames, main + c5c recall {recall:.2%}", ""] + rows + [
            "", f"Ceiling if every miss of a kind were fixed (shares of ball frames): item 2 +{ceil['item2']:.1%}, "
                f"item 3 +{ceil['item3']:.1%}, item 4 +{ceil['item4']:.1%}, localisation +{ceil['near_miss']:.1%}; "
                f"hidden behind players {ceil['occluded']:.1%} (batch interpolation only). "
                f"**Live ceiling {ceil['live_ceiling']:.1%}, batch ceiling {ceil['batch_ceiling']:.1%}.**", "",
            f"False positives on ball-labelled frames: {len(fps_s)}, of which {offp} outside the pitch-line hull "
            f"(likely a spare ball); median distance to the labelled ball {fp_med or 0:.0f} px.", ""]
        all_misses += misses_s

    disp_rows = []
    for g in sorted(disp_all):
        raw = np.array(disp_all[g][0]) if disp_all[g][0] else np.array([np.nan])
        comp = np.array(disp_all[g][1]) if disp_all[g][1] else np.array([np.nan])
        disp_rows.append({"gap": g, "n": len(disp_all[g][0]),
                          "raw_p50": float(np.nanpercentile(raw, 50)), "raw_p90": float(np.nanpercentile(raw, 90)),
                          "raw_p99": float(np.nanpercentile(raw, 99)), "comp_n": len(disp_all[g][1]),
                          "comp_p50": float(np.nanpercentile(comp, 50)), "comp_p90": float(np.nanpercentile(comp, 90)),
                          "comp_p99": float(np.nanpercentile(comp, 99))})
    fit = None
    pts = [(r["gap"], r["comp_p99"]) for r in disp_rows if r["comp_n"] >= 20 and np.isfinite(r["comp_p99"])]
    if len(pts) >= 3:
        gg, vv = np.array(pts).T
        b, a0 = np.polyfit(gg, vv, 1)
        fit = {"gate_a": float(max(a0, 10.0)), "gate_b": float(max(b, 1.0))}
    data["displacement"] = disp_rows
    data["gate_fit"] = fit
    report += ["## Ball displacement between labelled frames (all diagnosed splits)", "",
               "| gap | pairs | raw p50 / p90 / p99 px | camera-compensated pairs | compensated p50 / p90 / p99 px |",
               "|---|---|---|---|---|"]
    for r in disp_rows:
        report.append(f"| {r['gap']} | {r['n']} | {r['raw_p50']:.0f} / {r['raw_p90']:.0f} / {r['raw_p99']:.0f} | "
                      f"{r['comp_n']} | {r['comp_p50']:.0f} / {r['comp_p90']:.0f} / {r['comp_p99']:.0f} |")
    report += ["", f"Gate fit to the compensated p99 (r(g) = a + b*g): {fit}" if fit else
               "Gate fit: not enough camera-compensated pairs (no motion cache?)", ""]

    n_audit = audit_sheets(all_misses, sds_by_name, a.out / "audit", frames_by_key, n=a.audit, seed=a.seed) if a.audit else 0
    report += [f"Audit: {n_audit} sampled `test` misses in `audit/page_*.jpg`; fill `audit/audit.csv`'s `human_cause` "
               "column on one pass by eye, then compare with `primary` (plan §2.4: rules below ~80% agreement get "
               "fixed on `select`).", ""]
    spec_txt = "off" if a.no_spec else f"`{a.ckpt}` at {a.system_thr}"
    text = "\n".join(["# Ball miss diagnosis (plan item 1)", "",
                      f"Main cache `{a.cache}` (output conf >= {a.main_conf}), specialist {spec_txt}, "
                      f"motion cache `{a.motion}`. 15 px centre match. A miss can carry several causes; the table "
                      "counts each under its primary (first-matching) cause.", ""] + report)
    print("\n" + text, flush=True)
    (a.out / "report_diagnosis.md").write_text(text + "\n", encoding="utf-8")
    (a.out / "diagnosis.json").write_text(json.dumps(data, indent=1, default=float))
    with open(a.out / "frames.jsonl", "w") as f:
        for fr in all_frames:
            f.write(json.dumps(fr, default=float) + "\n")
    if all_misses:
        with open(a.out / "misses.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_misses[0]))
            w.writeheader()
            w.writerows(all_misses)
    with open(a.out / "displacement.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(disp_rows[0]) if disp_rows else ["gap"])
        w.writeheader()
        w.writerows(disp_rows)
    for sd in sds_by_name.values():
        sd.reader.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="cache_main_detections.py output")
    ap.add_argument("--motion", type=Path, default=None, help="camera_motion_cache.py output")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs" / "ball_temporal" / "c5c" / "best.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["select", "test"])
    ap.add_argument("--main_conf", type=float, default=0.15)
    ap.add_argument("--max_miss", type=int, default=15)
    ap.add_argument("--system_thr", type=float, default=0.70)
    ap.add_argument("--no_spec", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--audit", type=int, default=200, help="sampled test misses for the audit sheets (0 = none)")
    ap.add_argument("--max_seqs", type=int, default=0, help="first N sequences per split (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.chain = "fused"
    stage_run(a)


if __name__ == "__main__":
    main()
