"""
evaluate_calib_propagation.py -- choose `calibration.calib_every_k` from data
(docs/path_b/plans/IMPLEMENTATION_PLAN.md section 7 item 3).

Runs full NBJW on every frame once (the accuracy reference), then replays the sequence
through LiveCalibrator for each K, serving its NBJW calls from that cache so only the
camera-motion + propagation path differs between runs. For every ground-truth athlete it
projects the bottom-centre of the GT box with each frame's homography and reports, per K:

  vs_nbjw  -- distance (m) to the same point projected with per-frame NBJW
  vs_gt    -- distance (m) to the GT pitch position (bbox_pitch)

as mean and p95, plus how many frames had no homography. Recommended K = the largest whose
vs_gt mean and p95 both stay within --tolerance_m (default 0.5) of K=1.

GT boxes stand in for detector boxes in the camera-motion mask (the athlete detector's
recall is ~98%, docs/results/PATH_B_EVAL_T4_2026-10-05.md). Inline mode only: async adds result latency that depends on the
NBJW backend's speed, so gate it separately on the production GPU.

Usage:
    python -m gsr_pipeline.evaluate_calib_propagation \\
        --sequence_dir data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021 --ks 1 3 5 10 25
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

from .calibrate import NBJWHomography, NBJWKeypointDetector
from .calibration_live import H_NONE, LiveCalibrator

PERSON_ROLES = ("player", "goalkeeper", "referee", "other")


def load_gt(sequence_dir: Path):
    """([(image_id, file_name)] in frame order, {image_id: [(ltrb, foot_px, pitch_m)]})."""
    with open(sequence_dir / "Labels-GameState.json", "r") as f:
        data = json.load(f)
    images = sorted(((img["image_id"], img["file_name"]) for img in data["images"] if img.get("is_labeled", True)),
                    key=lambda pair: pair[1])
    by_image: Dict[str, list] = defaultdict(list)
    for ann in data["annotations"]:
        box, pitch = ann.get("bbox_image"), ann.get("bbox_pitch")
        if (ann.get("attributes") or {}).get("role") in PERSON_ROLES and box and pitch:
            by_image[ann["image_id"]].append((
                (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]),
                (box["x"] + box["w"] / 2, box["y"] + box["h"]),
                (pitch["x_bottom_middle"], pitch["y_bottom_middle"]),
            ))
    return images, by_image


def _project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ h.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def _summary(d: List[float]) -> Dict[str, float]:
    if not d:
        return {"n": 0, "mean": float("nan"), "p95": float("nan")}
    a = np.asarray(d)
    return {"n": int(a.size), "mean": float(a.mean()), "p95": float(np.percentile(a, 95))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick calib_every_k: propagated vs per-frame NBJW, in metres")
    ap.add_argument("--sequence_dir", required=True)
    ap.add_argument("--config", default="gsr_pipeline/configs/default.yaml")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 15, 25])
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--device", default=None, help="NBJW device (default: config)")
    ap.add_argument("--backend", default=None, help="NBJW backend (default: config)")
    ap.add_argument("--tolerance_m", type=float, default=0.5)
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    with open(args.config) as f:
        c = yaml.safe_load(f)["calibration"]
    seq = Path(args.sequence_dir)
    images, gt = load_gt(seq)
    images = images[args.start_frame:]
    if args.max_frames:
        images = images[:args.max_frames]
    img_dir = seq / "img1" if (seq / "img1").is_dir() else seq / "images"

    def read(fn):
        return cv2.imread(str(img_dir / fn))

    kp = NBJWKeypointDetector(
        checkpoint_kp=c["checkpoint_kp"], checkpoint_l=c["checkpoint_l"],
        device=args.device or c.get("device", "cpu"), backend=args.backend or c.get("backend", "torch"),
        onnx_kp=c.get("onnx_kp"), onnx_l=c.get("onnx_l"), fp16=c.get("fp16", False),
        fast_decode=c.get("fast_decode", True))
    fitter = NBJWHomography(c.get("image_width", 1920), c.get("image_height", 1080), use_prev_homography=False)

    print(f"{seq.name}: per-frame NBJW reference over {len(images)} frames ({kp.backend_name}) ...", flush=True)
    t0 = time.perf_counter()
    ref = []                                  # per frame: (H, params, kps, lines)
    for _, fn in images:
        kps, lines = kp.detect(read(fn))
        H, params = fitter.compute(kps)
        ref.append((H, params, kps, lines))
    print(f"  {1000 * (time.perf_counter() - t0) / max(1, len(images)):.0f} ms/frame for NBJW + homography",
          flush=True)

    results = {}
    for k in args.ks:
        calib = LiveCalibrator(
            kp_detector=None, image_width=c.get("image_width", 1920), image_height=c.get("image_height", 1080),
            use_prev_homography=c.get("use_prev_homography", True), calib_every_k=k, mode="inline",
            max_h_age_frames=c.get("max_h_age_frames", 50), min_inlier_ratio=c.get("min_inlier_ratio", 0.5),
            max_zoom_change=c.get("max_zoom_change", 0.15),
            min_trigger_interval_frames=c.get("min_trigger_interval_frames", 3), motion=c.get("motion"))
        cur = {"t": 0}
        calib._run_nbjw = lambda frame, cur=cur: ref[cur["t"]]
        vs_nbjw, vs_gt, sources, nbjw_runs, overhead_ms = [], [], defaultdict(int), 0, []
        last_ref = None                       # per-frame NBJW with reuse -- what K=1 emits
        for t, (image_id, fn) in enumerate(images):
            cur["t"] = t
            anns = gt.get(image_id, [])
            boxes = np.array([a[0] for a in anns], float) if anns else None
            frame = read(fn)
            s = time.perf_counter()
            res = calib.update(t, frame, boxes)
            if not res.nbjw_ran:
                overhead_ms.append(1000 * (time.perf_counter() - s))
            nbjw_runs += int(res.nbjw_ran)
            sources[res.h_source] += 1
            if ref[t][0] is not None:
                last_ref = ref[t][0]
            if not anns or res.H is None:
                continue
            foot = np.array([a[1] for a in anns], float)
            gt_pitch = np.array([a[2] for a in anns], float)
            pk = _project(res.H, foot)
            ok = np.isfinite(pk).all(1)
            vs_gt += np.linalg.norm(pk[ok] - gt_pitch[ok], axis=1).tolist()
            if last_ref is not None:
                pr = _project(last_ref, foot)
                ok2 = ok & np.isfinite(pr).all(1)
                vs_nbjw += np.linalg.norm(pk[ok2] - pr[ok2], axis=1).tolist()
        results[k] = {"vs_nbjw": _summary(vs_nbjw), "vs_gt": _summary(vs_gt),
                      "nbjw_runs": nbjw_runs, "frames": len(images),
                      "no_h_frames": sources.get(H_NONE, 0), "h_sources": dict(sources),
                      "propagation_ms_mean": float(np.mean(overhead_ms)) if overhead_ms else 0.0}

    base = results.get(1) or results[min(results)]
    print("\n| K | NBJW runs | vs NBJW mean / p95 (m) | vs GT mean / p95 (m) | no-H frames "
          "| propagate ms/frame | sources |")
    print("|---|---|---|---|---|---|---|")
    passing = []
    for k, r in results.items():
        a, g = r["vs_nbjw"], r["vs_gt"]
        ok = (g["mean"] - base["vs_gt"]["mean"] <= args.tolerance_m
              and g["p95"] - base["vs_gt"]["p95"] <= args.tolerance_m)
        if ok:
            passing.append(k)
        print(f"| {k} | {r['nbjw_runs']}/{r['frames']} | {a['mean']:.2f} / {a['p95']:.2f} | "
              f"{g['mean']:.2f} / {g['p95']:.2f} | {r['no_h_frames']} | {r['propagation_ms_mean']:.1f} | "
              + ", ".join(f"{s} {n}" for s, n in sorted(r["h_sources"].items()))
              + f" |{'' if ok else ' FAIL'}")
    rec = max(passing) if passing else 1
    print(f"\nRecommended calib_every_k (vs-GT mean and p95 within {args.tolerance_m} m of K=1): {rec}")
    print("One sequence is not enough to set it; repeat over several GSR valid sequences.")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"sequence": seq.name, "results": results, "recommended_k": rec,
                       "tolerance_m": args.tolerance_m}, f, indent=2)


if __name__ == "__main__":
    main()
