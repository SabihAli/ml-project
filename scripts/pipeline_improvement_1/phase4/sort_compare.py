"""
sort_compare.py -- Phase 4 follow-up of docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md: off-the-shelf
SORT-family trackers (boxmot: ByteTrack, OC-SORT, BoT-SORT without re-id) driving main + c5c,
against the Phase 1 system, on the same splits, scorer and kill criterion as phase4_sweep.py.

Per frame:
  1. detections = cached main balls at conf >= floor, as boxes centred on the ball with side
     max(w, h, box_px) (box_px > 0 makes IoU usable for a fast 5-15 px ball)
  2. c5c on a 640 window at the last measured output (gap <= max_miss, memoised per window); a
     peak >= thr not within 30 px of a main detection is added as one more detection
  3. tracker.update(detections, frame) -> tracks matched to a detection this frame
  4. one ball out: the sticky ball track if matched, else the matched track with the highest
     confidence (it becomes the ball track); the output is that detection's centre
BoT-SORT is searched without camera compensation (decoding full frames for every trial is too slow);
on test it runs both without (botsort) and with it (botsort_cmc, sparse optical flow on the real frame).

Search on select per tracker: coordinate descent (--passes) over SPACE; objective = recall with
precision >= Phase-1 precision - --prec_drop. Then frozen on test against Phase 1.

Outputs (--out): report_sort.md, sort.json, sort_search.csv, sort_log.txt.

    python scripts/pipeline_improvement_1/phase4/sort_compare.py --src data/datasets/SoccerNetGS \
        --cache <phase4 main_cache> --motion <phase4 motion_cache> --ckpt runs/ball_temporal/c5c/best.pt \
        --out data/outputs/pipeline_improvement_1/phase4/sort
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import phase4_sweep as p4  # noqa: E402  (also puts gsr_pipeline / ball_temporal / ball_eval on the path)
from eval_ball_temporal import Specialist, clamp_window, load_model  # noqa: E402
from gsr_io import discover  # noqa: E402

DEDUP_PX = 30.0
COMMON = {
    "thr": [0.35, 0.45, 0.55],
    "floor": [0.10, 0.15, 0.25],
    "box_px": [0, 24, 48],
    "asso_func": ["iou", "centroid"],
    "max_age": [15, 30, 60],
    "min_hits": [1, 3],
}
SPACE = {
    "bytetrack": {**COMMON, "track_thresh": [0.2, 0.4, 0.6], "match_thresh": [0.7, 0.8, 0.9]},
    "ocsort": {**COMMON, "det_thresh": [0.2, 0.4, 0.6], "iou_threshold": [0.1, 0.2, 0.3],
               "use_byte": [False, True]},
    "botsort": {**COMMON, "track_high_thresh": [0.2, 0.4, 0.6], "new_track_thresh": [0.3, 0.5, 0.7],
                "match_thresh": [0.7, 0.8, 0.9]},
}
DEFAULTS = {
    "bytetrack": {"thr": 0.45, "floor": 0.15, "box_px": 24, "asso_func": "iou", "max_age": 30, "min_hits": 1,
                  "track_thresh": 0.4, "match_thresh": 0.8},
    "ocsort": {"thr": 0.45, "floor": 0.15, "box_px": 24, "asso_func": "iou", "max_age": 30, "min_hits": 1,
               "det_thresh": 0.4, "iou_threshold": 0.2, "use_byte": True},
    "botsort": {"thr": 0.45, "floor": 0.15, "box_px": 24, "asso_func": "iou", "max_age": 30, "min_hits": 1,
                "use_cmc": False, "track_high_thresh": 0.4, "new_track_thresh": 0.5, "match_thresh": 0.8},
}
REPLAY = ("thr", "floor", "box_px")


def make_tracker(name: str, knobs: dict):
    import boxmot
    kw = {k: v for k, v in knobs.items() if k not in REPLAY}
    if name == "bytetrack":
        kw["max_lost_frames"] = kw["max_age"]
        return boxmot.ByteTrack(boxmot.ByteTrackConfig(**kw))
    if name == "ocsort":
        return boxmot.OcSort(boxmot.OcSortConfig(**kw))
    kw.update(max_lost_frames=kw["max_age"], use_embeddings=False)
    return boxmot.BotSort(boxmot.BotSortConfig(**kw), reid=None)


def run_sort(name: str, q: p4.Seq, spec: Specialist, knobs: dict, a) -> list[dict]:
    sd = q.sd
    tr = make_tracker(name, knobs)
    blank = np.zeros((sd.H, sd.W, 3), np.uint8)
    ball_id, last, out = None, None, []
    for t in sorted(sd.files)[: a.max_frames or None]:
        pts = [(b[0], b[1], max(b[2], b[3], knobs["box_px"]), b[4]) for b in sd.dets.get(t, []) if b[4] >= knobs["floor"]]
        rec = dict(t=t, out=None, pred=None, src="", win=None, sconf=None, gap=None)
        if last is not None and t - last[2] <= a.max_miss:
            x0, y0, _ = clamp_window(last[0], last[1], sd.W, sd.H)
            key = (sd.ref.name, t, x0, y0)
            px, py, conf = spec(key, None if key in spec.memo else sd.window_frames(t), x0, y0)
            rec.update(win=(int(x0), int(y0)), sconf=float(conf), gap=t - last[2])
            if conf >= knobs["thr"] and all(np.hypot(px - p[0], py - p[1]) > DEDUP_PX for p in pts):
                pts.append((px, py, max(10.0, knobs["box_px"]), float(conf), "spec"))
        dets = np.array([[x - s / 2, y - s / 2, x + s / 2, y + s / 2, c, 0] for x, y, s, c, *_ in pts], float)
        img = sd.frame(t) if knobs.get("use_cmc") else blank
        res = tr.update(dets.reshape(-1, 6), img)
        res = [r for r in np.asarray(res).reshape(-1, 8) if 0 <= int(r[7]) < len(pts)] if len(res) else []
        if res:
            pick = next((r for r in res if int(r[4]) == ball_id), None)
            if pick is None:
                pick = max(res, key=lambda r: r[5])
            ball_id = int(pick[4])
            p = pts[int(pick[7])]
            rec.update(out=(float(p[0]), float(p[1])), src="spec" if len(p) == 5 else "main")
            last = (p[0], p[1], t)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--motion", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trackers", nargs="+", default=list(SPACE))
    ap.add_argument("--search_split", default="select")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--prec_drop", type=float, default=0.01)
    ap.add_argument("--min_gain", type=float, default=0.005)
    ap.add_argument("--abrupt_px", type=float, default=6.0)
    ap.add_argument("--leak_seqs", default="SNGS-050")
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--max_miss", type=int, default=15)
    ap.add_argument("--main_conf", type=float, default=0.15, help="Phase 1 baseline only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("boxmot").setLevel(logging.WARNING)
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    logf = open(a.out / "sort_log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    import boxmot
    import torch
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model, channels, epoch = load_model(a.ckpt, dev)
    spec = Specialist(model, channels, dev)
    refs = discover(a.src)
    leak = tuple(s for s in a.leak_seqs.split(",") if s)

    def load(split):
        return [p4.Seq(r, a.cache, a.motion, a.abrupt_px) for r in [r for r in refs if r.split == split][: a.max_seqs or None]]

    sel = load(a.search_split)
    base = p4.score(sel, [p4.run_phase1(q, spec, a) for q in sel])
    floor = base["precision"] - a.prec_drop
    log(f"boxmot {boxmot.__version__}; c5c epoch {epoch} on {dev}; {a.search_split}: {len(sel)} sequences")
    log(f"[phase1 on {a.search_split}] recall {base['recall']:.2%} precision {base['precision']:.2%}; floor {floor:.2%}")

    def obj(s):
        return (s["recall"] if s["precision"] >= floor else -1.0, s["precision"])

    rows, chosen = [], {}
    for name in a.trackers:
        seen = {}

        def trial(kn, tag):
            key = json.dumps(kn, sort_keys=True)
            if key not in seen:
                t0 = time.time()
                s = p4.score(sel, [run_sort(name, q, spec, kn, a) for q in sel])
                s["secs"] = round(time.time() - t0, 1)
                seen[key] = s
                rows.append(dict(tracker=name, tag=tag, knobs=key, **{k: v for k, v in s.items() if k != "cond"}))
                log(f"  {name:>9} {tag:>24}: recall {s['recall']:.2%} prec {s['precision']:.2%} | {s['secs']}s")
            return seen[key]

        best = dict(DEFAULTS[name])
        best_s = trial(best, "defaults")
        for p in range(a.passes):
            improved = False
            for k, vals in SPACE[name].items():
                for v in vals:
                    if v != best[k]:
                        s = trial({**best, k: v}, f"{k}={v}")
                        if obj(s) > obj(best_s):
                            best, best_s, improved = {**best, k: v}, s, True
            if not improved:
                break
        chosen[name] = best
        log(f"[{name}] chosen on {a.search_split}: {best} -> recall {best_s['recall']:.2%} prec {best_s['precision']:.2%}")
    for q in sel:
        q.sd.reader.close()

    test = load(a.eval_split)
    res = {}
    p1 = [p4.run_phase1(q, spec, a) for q in test]
    res["phase1"] = p4.score(test, p1)
    res["phase1"]["recall_ex"] = p4.score(test, p1, leak)["recall"]
    runs = dict(chosen)
    if "botsort" in chosen:
        runs["botsort_cmc"] = {**chosen["botsort"], "use_cmc": True}
    for name, kn in runs.items():
        outs = [run_sort(name.replace("_cmc", ""), q, spec, kn, a) for q in test]
        res[name] = p4.score(test, outs)
        res[name]["recall_ex"] = p4.score(test, outs, leak)["recall"]
        log(f"[{a.eval_split}] {name:>9}: recall {res[name]['recall']:.2%} prec {res[name]['precision']:.2%} | "
            f"longest {res[name]['longest_miss']}")
    for q in test:
        q.sd.reader.close()
    b = res["phase1"]
    keep = {n: res[n]["recall"] - b["recall"] >= a.min_gain and b["precision"] - res[n]["precision"] <= a.prec_drop
            for n in runs}

    with open(a.out / "sort_search.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump({"boxmot": boxmot.__version__, "chosen": chosen, "floor": floor, "phase1_search": base, "eval": res,
               "keep": keep}, open(a.out / "sort.json", "w"), indent=1, default=str)

    pc = lambda x: f"{x:.2%}"  # noqa: E731
    L = ["# Phase 4 follow-up: SORT-family trackers (boxmot " + boxmot.__version__ + ")", "",
         f"Knobs chosen per tracker on `{a.search_split}` (coordinate descent, {len(rows)} runs in all; precision floor "
         f"{pc(floor)}), frozen for `{a.eval_split}`. Same scorer as `report_phase4.md` (measured outputs).", "",
         f"## `{a.eval_split}`", "",
         f"| system | recall @15 | @10 | @5 | precision | w/o {','.join(leak)} | longest miss | miss run p95 | verdict |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, s in res.items():
        v = "baseline" if n == "phase1" else ("KEEP" if keep[n] else "REVERT")
        L.append(f"| {n} | {pc(s['recall'])} | {pc(s['recall10'])} | {pc(s['recall5'])} | {pc(s['precision'])} | "
                 f"{pc(s['recall_ex'])} | {s['longest_miss']} | {s['miss_run_p95']:.0f} | {v} |")
    L += ["", "## Recall by condition", "", "| system | " + " | ".join(f"{c} (n={b['cond'][c]['n']})" for c in p4.CONDS) + " |",
          "|---|" + "---|" * len(p4.CONDS)]
    for n, s in res.items():
        L.append(f"| {n} | " + " | ".join(pc(s["cond"][c]["recall"]) for c in p4.CONDS) + " |")
    L += ["", "## Chosen knobs", "", "```", json.dumps(chosen, indent=1), "```", "",
          f"Search log: `sort_search.csv`. Specialist calls: {spec.calls}."]
    (a.out / "report_sort.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"wrote {a.out / 'report_sort.md'}")


if __name__ == "__main__":
    main()
