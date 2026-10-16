"""
eval_ball_temporal.py -- step 4 of docs/path_b/plans/BALL_TEMPORAL_PLAN.md §9: gates 1-3 (plan §8)
for a temporal ball specialist checkpoint (scripts/ball_temporal/train_ball_temporal.py).

Standalone apart from scripts/ball_temporal/ (gsr_io, heatmap_model, build_ball_temporal_dataset,
train_ball_temporal). Needs torch, numpy, opencv-python. Reads the main-detector cache (step 1) and
the crop dataset (step 2).

  crops   gates 1-2: the step 2 crops of a split. Specialist vs main detector on the same crops,
          all positives and hard positives, plus false positives on negative crops.
  system  gate 3: main + specialist on every frame of a split, simulated as the runtime runs it.
  all     both, threshold chosen on `select`, applied unchanged to `test`; writes report.md.

System simulation, per sequence, frame by frame (implementation plan §5.1, temporal plan §6):
  * the main detector's top ball at conf >= --main_conf is the output, and its position is the
    last sighting;
  * otherwise, if the last sighting is at most --max_miss frames old, the specialist runs on
    frames t-2, t-1, t cut by a 640x640 window centred on it (shift-clamped, no camera shift);
    its peak above the threshold is the output;
  * --chain fused (the runtime): a specialist detection becomes the new sighting (window
    re-centres, gap resets). --chain main (as the training crops were built): only main-detector
    sightings place the window.
Scoring: 15 px centre match. recall = hit frames / ball-labelled frames. precision = hits / outputs
on ball-labelled frames (outputs on frames without a ball label are counted separately: those
frames may show an unlabelled ball). Longest miss = longest run of ball-labelled frames not hit.
Threshold (--rule, chosen on `select`, frozen for `test`):
  spec_precision (default, 2026-10-15): the lowest threshold at which >= --spec_precision of the
    specialist's own outputs on ball-labelled frames are on the labelled ball. Independent of the
    main detector's precision, which differs between splits (88.3% select, 85.3% test).
  main_precision (the first rule): the highest recall whose system precision is >= the main
    detector's own on `select`.

    python scripts/eval/ball_det/eval_ball_temporal.py all --ckpt runs/ball_temporal/c5b/best.pt \
        --src data/datasets/SoccerNetGS --out data/outputs/ball_temporal/eval/c5b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "ball_temporal"))
from build_ball_temporal_dataset import CROP, clamp_window  # noqa: E402
from gsr_io import SeqReader, decode, discover, parse_labels  # noqa: E402
from heatmap_model import BallHeatmapNet, build_input, decode as decode_hm  # noqa: E402

MATCH_PX = 15.0
FPS = 25
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
GAP_BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 15), (16, 25))


def load_model(ckpt: Path, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    m = BallHeatmapNet(ck["channels"], pretrained=False).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, ck["channels"], ck.get("epoch")


def _near(p, gts) -> bool:
    return any(np.hypot(p[0] - g[0], p[1] - g[1]) <= MATCH_PX for g in gts)


# --------------------------------------------------------------------------- gates 1-2: crops

@torch.no_grad()
def crop_scores(model, channels, data: Path, split: str, dev, workers: int):
    """Specialist peak per crop of a split -> rows with conf, hit (and the manifest's main_conf)."""
    from torch.utils.data import DataLoader
    from train_ball_temporal import TemporalCrops, read_manifest
    rows = read_manifest(data, (split,))
    dl = DataLoader(TemporalCrops(data, rows, False), 32, num_workers=workers)
    for x, pts, _, idx in dl:
        x, pts = x.to(dev), pts.to(dev)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            logits, off = model(build_input(x, channels))
        p = decode_hm(logits, off)
        d = torch.cdist(p[:, None, :2], pts[..., :2]).squeeze(1)
        d[pts[..., 2] <= 0] = float("inf")
        hit = (d.min(1).values <= MATCH_PX).cpu().numpy()
        for j, i in enumerate(idx.tolist()):
            rows[i]["spec_conf"] = float(p[j, 2])
            rows[i]["spec_hit"] = bool(hit[j])
    return rows


def crop_report(rows, spec_thr: float, main_conf: float) -> dict:
    """Gates 1-2. Main detector on a crop: a main ball >= main_conf within 15 px of the GT ball
    (positives), or any main ball >= main_conf inside the window (negatives); from the manifest."""
    out = {}
    groups = {"all positives": lambda r: r["kind"] in ("hard", "easy"),
              "hard": lambda r: r["kind"] == "hard", "easy": lambda r: r["kind"] == "easy"}
    for name, f in groups.items():
        rs = [r for r in rows if f(r)]
        out[name] = {"n": len(rs),
                     "main": float(np.mean([float(r["main_conf"]) >= main_conf for r in rs])) if rs else 0.0,
                     "spec": float(np.mean([r["spec_hit"] and r["spec_conf"] >= spec_thr for r in rs])) if rs else 0.0}
    neg = [r for r in rows if r["kind"].startswith("neg")]
    out["negatives fire"] = {"n": len(neg),
                             "main": float(np.mean([float(r["main_conf"]) >= main_conf for r in neg])),
                             "spec": float(np.mean([r["spec_conf"] >= spec_thr for r in neg]))}
    outside = [r for r in rows if r["kind"] == "outside"]
    out["outside fire"] = {"n": len(outside),
                           "main": float(np.mean([float(r["main_conf"]) >= main_conf for r in outside])) if outside else 0.0,
                           "spec": float(np.mean([r["spec_conf"] >= spec_thr for r in outside])) if outside else 0.0}
    return out


def neg_fpr_threshold(rows, fpr: float) -> float:
    neg = np.array([r["spec_conf"] for r in rows if r["kind"].startswith("neg")])
    return float(np.quantile(neg, 1 - fpr))


# --------------------------------------------------------------------------- gate 3: system

class Specialist:
    """Runs the model on one window of frames t-2, t-1, t; memoised by (frame, x0, y0)."""

    def __init__(self, model, channels, dev):
        self.model, self.channels, self.dev = model, channels, dev
        self.memo: dict = {}
        self.calls = 0
        self.ms = 0.0

    @torch.no_grad()
    def __call__(self, key, frames, x0, y0):
        if key in self.memo:
            return self.memo[key]
        x = np.concatenate([f[y0:y0 + CROP, x0:x0 + CROP, ::-1] for f in frames], axis=2)   # BGR->RGB, HxWx9
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.dev)
        t0 = time.perf_counter()
        with torch.autocast("cuda", enabled=self.dev.type == "cuda"):
            logits, off = self.model(build_input(x, self.channels))
        p = decode_hm(logits, off)[0].tolist()
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        self.ms += (time.perf_counter() - t0) * 1000
        self.calls += 1
        self.memo[key] = (p[0] + x0, p[1] + y0, p[2])
        return self.memo[key]


class SeqData:
    """One sequence: labels, cached main detections, and a small decoded-frame cache."""

    def __init__(self, ref, cache_dir: Path):
        self.ref = ref
        self.reader = SeqReader(ref)
        self.im_dir, self.files, balls, *_rest, (self.W, self.H) = parse_labels(self.reader.labels())
        self.gt = {f: [(b[0], b[1]) for b in bs] for f, bs in balls.items()}
        cache = json.loads((cache_dir / ref.split / f"{ref.name}.json").read_text())["frames"]
        self.dets = {int(k): v["balls"] for k, v in cache.items()}
        self._frames: dict = {}

    def frame(self, f):
        if f not in self._frames:
            if len(self._frames) > 8:
                self._frames.pop(min(self._frames))
            self._frames[f] = decode(self.reader.read(f"{self.im_dir}/{self.files[f]}"))
        return self._frames[f]

    def window_frames(self, t):
        return [self.frame(f if f in self.files else t) for f in (t - 2, t - 1, t)]


def simulate(sd: SeqData, spec: Specialist | None, thr: float, a) -> list[dict]:
    """-> per frame {t, x, y, src, gap} (x None when no output)."""
    out, last = [], None                        # last sighting: (x, y, frame)
    for t in sorted(sd.files):
        m = [b for b in sd.dets.get(t, []) if b[4] >= a.main_conf]
        if m:
            last = (m[0][0], m[0][1], t)
            out.append(dict(t=t, x=m[0][0], y=m[0][1], src="main", gap=None))
            continue
        rec = dict(t=t, x=None, y=None, src="", gap=None)
        if spec is not None and last is not None and t - last[2] <= a.max_miss:
            x0, y0, _ = clamp_window(last[0], last[1], sd.W, sd.H)
            rec["gap"] = t - last[2]
            key = (sd.ref.name, t, x0, y0)
            frames = None if key in spec.memo else sd.window_frames(t)
            px, py, conf = spec(key, frames, x0, y0)
            if conf >= thr:
                rec.update(x=px, y=py, src="spec")
                if a.chain == "fused":
                    last = (px, py, t)
        out.append(rec)
    return out


def score(sd_list, outs) -> dict:
    n_gt = hit = pred_gt = pred_nogt = spec_out = spec_hit = 0
    misses, gap_n, gap_hit = [], defaultdict(int), defaultdict(int)
    for sd, out in zip(sd_list, outs):
        run, longest = 0, 0
        for r in out:
            gts = sd.gt.get(r["t"])
            ok = bool(gts) and r["x"] is not None and _near((r["x"], r["y"]), gts)
            if gts:
                n_gt += 1
                hit += ok
                pred_gt += r["x"] is not None
                if r["src"] == "spec":
                    spec_out += 1
                    spec_hit += ok
                if r["gap"] is not None:
                    b = next((f"{lo}-{hi}" if lo != hi else str(lo) for lo, hi in GAP_BUCKETS if lo <= r["gap"] <= hi), ">25")
                    gap_n[b] += 1
                    gap_hit[b] += ok
                run = 0 if ok else run + 1
                longest = max(longest, run)
            elif r["x"] is not None:
                pred_nogt += 1
        misses.append(longest / FPS)
    return {"gt_frames": n_gt, "recall": hit / max(n_gt, 1), "precision": hit / max(pred_gt, 1),
            "outputs_unlabelled": pred_nogt, "spec_outputs": spec_out,
            "spec_precision": spec_hit / max(spec_out, 1), "spec_added": spec_hit / max(n_gt, 1),
            "longest_miss_mean_s": float(np.mean(misses)), "longest_miss_max_s": float(np.max(misses)),
            "by_gap": {k: (gap_hit[k] / gap_n[k], gap_n[k]) for k in sorted(gap_n, key=lambda s: int(s.split("-")[0].lstrip(">")))}}


def run_system(split, spec, a, thresholds):
    refs = [r for r in discover(a.src) if r.split == split]
    sds = [SeqData(r, a.cache) for r in refs]
    base = score(sds, [simulate(sd, None, 1.0, a) for sd in sds])
    res = {}
    for thr in thresholds:
        t0 = time.time()
        res[float(thr)] = score(sds, [simulate(sd, spec, float(thr), a) for sd in sds])
        print(f"  {split} thr {thr:.2f}: recall {res[float(thr)]['recall']:.2%} precision "
              f"{res[float(thr)]['precision']:.2%} ({time.time() - t0:.0f}s, {spec.calls} specialist calls)", flush=True)
    for sd in sds:
        sd.reader.close()
    return base, res


# --------------------------------------------------------------------------- report

def pct(v):
    return f"{v:.2%}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["all"])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--src", type=Path, default=Path("data/datasets/SoccerNetGS"))
    ap.add_argument("--cache", type=Path, default=Path("data/outputs/ball_temporal/main_cache"))
    ap.add_argument("--data", type=Path, default=Path("data/outputs/ball_temporal/dataset"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--main_conf", type=float, default=0.15, help="main-detector output threshold")
    ap.add_argument("--max_miss", type=int, default=15, help="frames since the last sighting the specialist still runs")
    ap.add_argument("--chain", choices=["fused", "main"], default="fused")
    ap.add_argument("--rule", choices=["spec_precision", "main_precision"], default="spec_precision")
    ap.add_argument("--spec_precision", type=float, default=0.90,
                    help="spec_precision rule: required share of specialist outputs on the labelled ball")
    ap.add_argument("--crop_fpr", type=float, default=0.05, help="gates 1-2 threshold: this share of select negatives fire")
    ap.add_argument("--test_sweep", action="store_true",
                    help="also report the test curve at every threshold: a diagnostic, never for choosing")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cudnn.benchmark = True
    a.out.mkdir(parents=True, exist_ok=True)
    model, channels, epoch = load_model(a.ckpt, dev)
    print(f"{a.ckpt} (epoch {epoch}, {channels} channels), chain {a.chain}, max_miss {a.max_miss}", flush=True)

    # gates 1-2
    sel_rows = crop_scores(model, channels, a.data, "select", dev, a.workers)
    crop_thr = neg_fpr_threshold(sel_rows, a.crop_fpr)
    test_rows = crop_scores(model, channels, a.data, "test", dev, a.workers)
    crops = crop_report(test_rows, crop_thr, a.main_conf)
    print(f"crop threshold {crop_thr:.3f} ({a.crop_fpr:.0%} of select negatives fire)", flush=True)

    # gate 3
    spec = Specialist(model, channels, dev)
    sel_base, sel = run_system("select", spec, a, THRESHOLDS)
    if a.rule == "spec_precision":
        ok = [t for t, r in sel.items() if r["spec_precision"] >= a.spec_precision]
        thr = min(ok) if ok else max(sel)
        rule_text = (f"the lowest `select` threshold at which >= {a.spec_precision:.0%} of the specialist's "
                     f"outputs are on the labelled ball")
    else:
        ok = [t for t, r in sel.items() if r["precision"] >= sel_base["precision"]]
        thr = max(ok, key=lambda t: sel[t]["recall"]) if ok else max(sel, key=lambda t: sel[t]["precision"])
        rule_text = f"the highest `select` recall with precision >= the main detector's on `select` ({sel_base['precision']:.2%})"
    print(f"system threshold {thr:.2f} (select: recall {sel[thr]['recall']:.2%}, precision "
          f"{sel[thr]['precision']:.2%}; main alone {sel_base['recall']:.2%} / {sel_base['precision']:.2%})", flush=True)
    spec.calls, spec.ms = 0, 0.0
    test_base, test = run_system("test", spec, a, THRESHOLDS if a.test_sweep else [thr])
    t = test[thr]
    ms = spec.ms / max(spec.calls, 1)      # with --test_sweep, averaged over every pass

    g1 = crops["all positives"]["spec"] > crops["all positives"]["main"]
    g3 = t["recall"] >= test_base["recall"] + 0.10 and t["precision"] >= test_base["precision"]
    L = [f"# Temporal ball specialist: test gates", "",
         f"Checkpoint `{a.ckpt}` (epoch {epoch}, {channels} channels). Main detector cache "
         f"`{a.cache}` (imgsz 1280, output conf >= {a.main_conf}). 15 px centre match. Thresholds "
         f"chosen on `select`, frozen for `test` (15 sequences).", "",
         "## Gates 1-2: crops (step 2 test crops)", "",
         f"Specialist threshold {crop_thr:.3f}: {a.crop_fpr:.0%} of `select` negative crops fire. "
         f"Main detector on a crop: a main ball >= {a.main_conf} within 15 px (positives) or inside the window (negatives).", "",
         "| crops | n | main detector | specialist |", "|---|---|---|---|"]
    for k, v in crops.items():
        L.append(f"| {k} | {v['n']} | {pct(v['main'])} | {pct(v['spec'])} |")
    L += ["", f"Gate 1 (specialist beats main on all test crops): **{'PASS' if g1 else 'FAIL'}**. "
              f"Gate 2 (hard crops): reported above.", "",
          f"## Gate 3: system, every test frame (chain `{a.chain}`, max_miss {a.max_miss})", "",
          f"System threshold {thr:.2f} (rule `{a.rule}`): {rule_text}.", "",
          "| | recall | precision | outputs on unlabelled frames | longest miss mean / max |", "|---|---|---|---|---|",
          f"| main detector | {pct(test_base['recall'])} | {pct(test_base['precision'])} | {test_base['outputs_unlabelled']} | "
          f"{test_base['longest_miss_mean_s']:.1f} s / {test_base['longest_miss_max_s']:.1f} s |",
          f"| main + specialist | {pct(t['recall'])} | {pct(t['precision'])} | {t['outputs_unlabelled']} | "
          f"{t['longest_miss_mean_s']:.1f} s / {t['longest_miss_max_s']:.1f} s |", "",
          f"Recall {(t['recall'] - test_base['recall']) * 100:+.1f} points, precision "
          f"{(t['precision'] - test_base['precision']) * 100:+.1f} points. Specialist outputs on labelled frames: "
          f"{t['spec_outputs']}, {pct(t['spec_precision'])} of them on the labelled ball.", "",
          f"Gate 3 (recall >= main + 10 points, precision >= main): **{'PASS' if g3 else 'FAIL'}**.", "",
          "Specialist recall on labelled frames the main detector missed, by frames since the last sighting:", "",
          "| gap | frames | recall |", "|---|---|---|"]
    for k, (r, n) in t["by_gap"].items():
        L.append(f"| {k} | {n} | {pct(r)} |")
    L += ["", f"`select` sweep (main alone: recall {pct(sel_base['recall'])}, precision {pct(sel_base['precision'])}):", "",
          "| threshold | recall | precision | specialist outputs on the ball |", "|---|---|---|---|"]
    for k, r in sel.items():
        L.append(f"| {k:.2f} | {pct(r['recall'])} | {pct(r['precision'])} | {pct(r['spec_precision'])} |")
    if a.test_sweep:
        L += ["", "`test` sweep, **diagnostic only** (the gate uses the `select` threshold above):", "",
              "| threshold | recall | precision |", "|---|---|---|"]
        for k, r in test.items():
            L.append(f"| {k:.2f} | {pct(r['recall'])} | {pct(r['precision'])} |")
    L += ["", f"Specialist PyTorch FP16 forward + decode, batch 1, this machine: {ms:.1f} ms per call "
              f"({spec.calls} calls on test). Not the gate 4 figure (that is TensorRT, step 5)."]
    text = "\n".join(L)
    print("\n" + text)
    (a.out / "report.md").write_text(text + "\n", encoding="utf-8")
    (a.out / "results.json").write_text(json.dumps(
        {"ckpt": str(a.ckpt), "epoch": epoch, "channels": channels, "chain": a.chain, "max_miss": a.max_miss,
         "rule": a.rule, "spec_precision_target": a.spec_precision,
         "crop_thr": crop_thr, "crops": crops, "system_thr": thr, "select_main": sel_base, "select": sel,
         "test_main": test_base, "test": t, "ms_per_call": ms, "gate1": g1, "gate3": g3}, indent=1, default=str))
    with open(a.out / "test_crops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(test_rows[0]))
        w.writeheader()
        w.writerows(test_rows)


if __name__ == "__main__":
    main()
