"""
export_nbjw.py -- export the two NBJW calibration networks to ONNX, optionally build their
TensorRT FP16 engines, and parity-check the exported graphs against torch.

docs/path_b/plans/IMPLEMENTATION_PLAN.md section 7 item 1. The vendored
nbjw_calib package is not edited: only the forward pass moves to ONNX/TensorRT, and its
heatmap -> keypoint -> homography post-processing is reused unchanged
(calibrate.NBJWKeypointDetector picks the backend from configs/default.yaml).

Export runs anywhere (CPU is fine). --build_engine needs the production GPU: TensorRT
engines are specific to the GPU, driver and TRT version, so they are built per host and
written next to the ONNX as <stem>_fp16.engine with an environment sidecar, which the
runtime checks before loading.

--check compares, on real frames, the torch FP32 model against a chosen backend: max
heatmap difference, keypoints found by one side only, and -- the gate that matters -- the
pitch-position error in metres between the resulting homographies over a grid of points
on the visible pitch.

Usage:
    python scripts/export_nbjw.py                                    # export both ONNX
    python scripts/export_nbjw.py --build_engine                     # + TRT FP16 (GPU host)
    python scripts/export_nbjw.py --skip_export --check ort_cpu --frames 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OPSET = 17


def export(model, path: Path) -> None:
    import torch
    from gsr_pipeline.calibrate import NBJW_INPUT_HW

    dummy = torch.zeros(1, 3, *NBJW_INPUT_HW)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(opset_version=OPSET, input_names=["images"], output_names=["heatmaps"],
                  do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:       # torch < 2.5 has no `dynamo` argument; TorchScript export is the default
        torch.onnx.export(model, dummy, str(path), **kwargs)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def pitch_error_m(h_ref, h_test, width=1920, height=1080, step=40) -> tuple[float, float]:
    """Mean / max distance in metres between two image->pitch homographies, over a pixel
    grid restricted to points the reference maps inside the pitch (+5 m)."""
    xs, ys = np.meshgrid(np.arange(0, width, step), np.arange(height // 3, height, step))
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 0)

    def proj(h):
        p = h @ pts
        return (p[:2] / p[2]).T

    a, b = proj(h_ref), proj(h_test)
    inside = (np.abs(a[:, 0]) < 57.5) & (np.abs(a[:, 1]) < 39)
    if not inside.any():
        return float("nan"), float("nan")
    d = np.linalg.norm(a[inside] - b[inside], axis=1)
    return float(d.mean()), float(d.max())


def check(args, cfg) -> None:
    import cv2
    from gsr_pipeline.calibrate import NBJWHomography, NBJWKeypointDetector
    from gsr_pipeline.evaluate_detection import load_gt_boxes

    seq = Path(args.sequence_dir)
    images, _ = load_gt_boxes(seq / "Labels-GameState.json")
    img_dir = seq / "img1" if (seq / "img1").is_dir() else seq / "images"
    step = max(1, len(images) // args.frames)
    frames = [cv2.imread(str(img_dir / fn)) for _, fn in images[::step][:args.frames]]

    common = dict(checkpoint_kp=cfg["checkpoint_kp"], checkpoint_l=cfg["checkpoint_l"],
                  device=args.device, onnx_kp=args.onnx_kp, onnx_l=args.onnx_l)
    ref = NBJWKeypointDetector(**common, backend="torch")
    test = NBJWKeypointDetector(**common, backend=args.check, fp16=args.check == "torch")
    print(f"reference {ref.backend_name}  vs  test {test.backend_name}")

    errs, kp_diff, hm_diff, one_sided = [], [], [], 0
    for frame in frames:
        hk_r, hl_r = (t.float().cpu().numpy() for t in ref.forward(ref.preprocess(frame)))
        hk_t, hl_t = (t.float().cpu().numpy() for t in test.forward(test.preprocess(frame)))
        hm_diff.append(max(np.abs(hk_r - hk_t).max(), np.abs(hl_r - hl_t).max()))

        kp_r, _ = ref.detect(frame)
        kp_t, _ = test.detect(frame)
        kp_diff.append(len(set(kp_r) ^ set(kp_t)))
        h_r = NBJWHomography(use_prev_homography=False).compute(kp_r)[0]
        h_t = NBJWHomography(use_prev_homography=False).compute(kp_t)[0]
        if h_r is None or h_t is None:
            one_sided += int((h_r is None) != (h_t is None))
            continue
        errs.append(pitch_error_m(h_r, h_t))

    mean = [e[0] for e in errs if np.isfinite(e[0])]
    worst = [e[1] for e in errs if np.isfinite(e[1])]
    print(f"frames {len(frames)}   max |heatmap diff| {max(hm_diff):.4f}   "
          f"keypoint-set differences per frame (mean) {np.mean(kp_diff):.2f}   "
          f"H found by only one side: {one_sided}")
    if mean:
        print(f"pitch error vs torch FP32: mean {np.mean(mean):.3f} m   "
              f"p95 of per-frame mean {np.percentile(mean, 95):.3f} m   worst point {max(worst):.3f} m")


def main() -> None:
    p = argparse.ArgumentParser(description="Export NBJW to ONNX / TensorRT and parity-check it")
    p.add_argument("--config", default=str(REPO_ROOT / "gsr_pipeline/configs/default.yaml"))
    p.add_argument("--onnx_kp", default=None, help="default: calibration.onnx_kp from the config")
    p.add_argument("--onnx_l", default=None, help="default: calibration.onnx_l from the config")
    p.add_argument("--skip_export", action="store_true", help="reuse existing ONNX files")
    p.add_argument("--build_engine", action="store_true", help="build TensorRT FP16 engines (GPU host)")
    p.add_argument("--workspace_gb", type=float, default=4.0)
    p.add_argument("--force", action="store_true", help="overwrite existing engines")
    p.add_argument("--check", default=None, choices=["tensorrt", "ort_gpu", "ort_cpu", "torch"],
                   help="parity-check this backend against torch FP32 (torch = torch FP16 autocast)")
    p.add_argument("--sequence_dir",
                   default=str(REPO_ROOT / "data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021"))
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--device", default=None, help="default cuda if available else cpu")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import torch
    import yaml
    from gsr_pipeline.calibrate import load_nbjw_models

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["calibration"]
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.onnx_kp = str(REPO_ROOT / (args.onnx_kp or cfg["onnx_kp"]))
    args.onnx_l = str(REPO_ROOT / (args.onnx_l or cfg["onnx_l"]))
    cfg = {**cfg, "checkpoint_kp": str(REPO_ROOT / cfg["checkpoint_kp"]),
           "checkpoint_l": str(REPO_ROOT / cfg["checkpoint_l"])}

    if not args.skip_export:
        model_kp, model_l = load_nbjw_models(cfg["checkpoint_kp"], cfg["checkpoint_l"], "cpu")
        export(model_kp, Path(args.onnx_kp))
        export(model_l, Path(args.onnx_l))

    if args.build_engine:
        if not torch.cuda.is_available():
            print("No CUDA device visible; a TensorRT engine can only be built on the GPU it "
                  "will run on. backend: auto falls back to ORT GPU / torch meanwhile.", file=sys.stderr)
            raise SystemExit(2)
        from gsr_pipeline.runtime.backend import sha256_file
        from gsr_pipeline.runtime.trt_runner import build_engine
        for onnx in (Path(args.onnx_kp), Path(args.onnx_l)):
            engine = onnx.with_name(f"{onnx.stem}_fp16.engine")
            if engine.exists() and not args.force:
                print(f"{engine.name} exists; pass --force to rebuild")
                continue
            build_engine(onnx, engine, sha256_file(onnx), fp16=True, workspace_gb=args.workspace_gb)

    if args.check:
        check(args, cfg)


if __name__ == "__main__":
    main()
