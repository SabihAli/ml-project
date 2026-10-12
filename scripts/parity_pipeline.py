"""
parity_pipeline.py -- prove an optimised code path gives the same outputs as the reference
on real frames (docs/path_b/plans/OPTIMIZATION_PLAN.md, 0.4).

Every "exact" optimisation in the plan lands behind a switch and must pass this before it
ships. Checks implemented so far:

  nbjw_decode   one NBJW forward per frame, decoded by the vendored reference and by
                gsr_pipeline/nbjw_fast_decode.py; keypoints, lines and the fitted homography
                must be identical. Also times both decodes on the same heatmaps.

Exit code 0 = identical on every frame, 1 = any difference (details printed).

    python scripts/parity_pipeline.py --data_root data/datasets/SoccerNetGS/gamestate-2024/valid \\
        --sequences SNGS-021 SNGS-022 SNGS-023 --frames 100 --backend tensorrt --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _frame_paths(seq: Path, frames: int, step: int):
    img_dir = seq / "img1" if (seq / "img1").is_dir() else seq / "images"
    paths = sorted(img_dir.glob("*.jpg"))[::step]
    return paths[:frames] if frames else paths


def _summary(ms):
    a = np.asarray(ms)
    return f"mean {a.mean():.2f}  p50 {np.percentile(a, 50):.2f}  p95 {np.percentile(a, 95):.2f}  max {a.max():.2f}"


def check_nbjw_decode(args, cfg) -> int:
    import cv2
    import torch
    from gsr_pipeline.calibrate import NBJWHomography, NBJWKeypointDetector, _kp_to_lines

    c = cfg["calibration"]
    kp = NBJWKeypointDetector(
        checkpoint_kp=c["checkpoint_kp"], checkpoint_l=c["checkpoint_l"],
        image_width=c.get("image_width", 1920), image_height=c.get("image_height", 1080),
        device=args.device, backend=args.backend, onnx_kp=c.get("onnx_kp"), onnx_l=c.get("onnx_l"))
    fit_ref, fit_fast = (NBJWHomography(c.get("image_width", 1920), c.get("image_height", 1080),
                                        use_prev_homography=False) for _ in range(2))
    cuda = str(kp.input_device).startswith("cuda")

    def tick():
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    print(f"nbjw_decode: backend {kp.backend_name}", flush=True)
    n = bad = 0
    t_ref, t_fast, worst_h = [], [], 0.0
    for seq_name in args.sequences:
        seq = Path(args.data_root) / seq_name
        for path in _frame_paths(seq, args.frames, args.step):
            frame = cv2.imread(str(path))
            hm_kp, hm_l = kp.forward(kp.preprocess(frame))
            # Alternate the order so neither decode always runs on a warm cache.
            order = (True, False) if n % 2 else (False, True)
            out = {}
            for fast in order:
                t0 = tick()
                out[fast] = kp.decode(hm_kp, hm_l, fast=fast)
                (t_fast if fast else t_ref).append((tick() - t0) * 1000)
            ref_kp, fast_kp = out[False], out[True]
            h_ref, _ = fit_ref.compute(ref_kp)
            h_fast, _ = fit_fast.compute(fast_kp)

            problems = []
            if fast_kp != ref_kp or list(fast_kp) != list(ref_kp):
                keys = sorted(set(ref_kp) ^ set(fast_kp))
                diff = [k for k in set(ref_kp) & set(fast_kp) if ref_kp[k] != fast_kp[k]]
                problems.append(f"keypoints differ (only-one-side {keys}, value diffs {sorted(diff)})")
            if _kp_to_lines(fast_kp) != _kp_to_lines(ref_kp):
                problems.append("lines differ")
            if (h_ref is None) != (h_fast is None):
                problems.append(f"homography found on one side only (ref {h_ref is not None})")
            elif h_ref is not None:
                d = float(np.abs(h_ref / h_ref[2, 2] - h_fast / h_fast[2, 2]).max())
                worst_h = max(worst_h, d)
                if d > args.h_tol:
                    problems.append(f"H differs by {d:.3g}")
            if problems:
                bad += 1
                print(f"  {seq_name}/{path.name}: " + "; ".join(problems), flush=True)
            n += 1
        print(f"  {seq_name}: done ({n} frames so far, {bad} differing)", flush=True)

    warm = slice(min(3, max(0, n - 1)), None)       # first decodes include lazy imports / allocator warm-up
    print(f"\nframes {n}, differing {bad}, worst |dH| {worst_h:.3g}")
    print(f"decode ms, reference: {_summary(t_ref[warm])}")
    print(f"decode ms, fast:      {_summary(t_fast[warm])}")
    print("PARITY: " + ("IDENTICAL" if bad == 0 else "DIFFERENT"))
    if args.json:
        Path(args.json).write_text(json.dumps({
            "check": "nbjw_decode", "backend": kp.backend_name, "frames": n, "differing": bad,
            "worst_h_diff": worst_h, "decode_ms_reference": t_ref, "decode_ms_fast": t_fast}, indent=2))
    return 0 if bad == 0 else 1


CHECKS = {"nbjw_decode": check_nbjw_decode}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--check", default="nbjw_decode", choices=sorted(CHECKS))
    p.add_argument("--data_root", default="data/datasets/SoccerNetGS/gamestate-2024/valid")
    p.add_argument("--sequences", nargs="+", default=["SNGS-021"])
    p.add_argument("--frames", type=int, default=100, help="frames per sequence (0 = all)")
    p.add_argument("--step", type=int, default=1, help="use every Nth frame")
    p.add_argument("--config", default="gsr_pipeline/configs/default.yaml")
    p.add_argument("--backend", default="torch", choices=["torch", "auto", "tensorrt", "ort_gpu", "ort_cpu"],
                   help="NBJW forward backend; the decode is compared on whatever heatmaps it produces")
    p.add_argument("--device", default="cpu")
    p.add_argument("--h_tol", type=float, default=1e-9, help="max |H_ref - H_fast| (normalised) counted as equal")
    p.add_argument("--json", default=None, help="also write the raw result here")
    args = p.parse_args()
    args.data_root = str(Path(args.data_root).resolve())

    import os
    import yaml
    os.chdir(ROOT)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sys.exit(CHECKS[args.check](args, cfg))


if __name__ == "__main__":
    main()
