"""
build_trt_engine.py -- first-install step: build a TensorRT engine from the shipped ONNX.

IMPLEMENTATION_PLAN.md section 3.2. TensorRT engines are specific to the GPU model, driver
and TRT version, so they cannot be shipped prebuilt. The ONNX is the portable artifact;
this runs once per production host (and again whenever the model, GPU or TRT version
changes). Expect minutes, not seconds -- the builder runs at its maximum optimisation
level deliberately, trading build time for inference speed.

The engine is written with a sidecar recording the environment it was built for, which
gsr_pipeline/runtime/backend.py checks at every startup and refuses on mismatch.

NOTE (2026-09-26): unverified. The dev machine has no NVIDIA GPU (plan section 16 item 2),
so this is Phase 1b work, to be run and confirmed on a CUDA host.

Usage:
    python scripts/build_trt_engine.py
    python scripts/build_trt_engine.py --onnx model_testing/.../best_1280x1280.onnx --fp32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ONNX = REPO_ROOT / "model_testing" / "yolo26s_athlete_ball" / "best_736x1280.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from the exported ONNX")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--engine", default=None, help="Defaults to <onnx stem>_fp16.engine")
    parser.add_argument("--fp32", action="store_true", help="Build without the FP16 flag")
    parser.add_argument("--workspace_gb", type=float, default=4.0)
    parser.add_argument("--force", action="store_true", help="Rebuild over an existing engine")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import torch

    from gsr_pipeline.runtime.backend import ModelArtifact
    from gsr_pipeline.runtime.trt_runner import build_engine

    if not torch.cuda.is_available():
        # Fail loudly and specifically: a host with no GPU cannot build an engine, and the
        # operator should know that rather than read a TensorRT import traceback.
        print(
            "No CUDA device visible. A TensorRT engine can only be built on the GPU it will "
            "run on.\nThe service will still start -- it falls back to ONNX Runtime "
            "(see plan section 3.1) -- but this step must be re-run on the production host.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # Loading through ModelArtifact enforces the class list, the no-embedded-NMS contract
    # and the ONNX checksum before anything expensive happens.
    artifact = ModelArtifact.load(args.onnx)

    if args.engine:
        engine_path = Path(args.engine)
    elif args.fp32:
        # ModelArtifact.engine_path is hardcoded to "..._fp16.engine" -- it is also what
        # select_backend() looks for and loads unconditionally as the TensorRT tier. An
        # --fp32 build must NOT default to that name: select_backend would then silently
        # load an FP32 engine, report "backend selected: tensorrt" as if it were the normal
        # FP16 path, and nothing in its logging would show the precision mismatch. --fp32
        # is a debugging tool, so it must be given an explicit --engine path to be picked
        # up by select_backend at all.
        engine_path = artifact.engine_path.with_name(
            artifact.engine_path.name.replace("_fp16.engine", "_fp32.engine")
        )
        print(
            f"--fp32 without --engine: writing to {engine_path.name}, NOT "
            f"{artifact.engine_path.name}. select_backend() only ever loads "
            f"{artifact.engine_path.name}, so this engine will not be picked up "
            "automatically -- pass --engine explicitly to use it."
        )
    else:
        engine_path = artifact.engine_path

    if engine_path.exists() and not args.force:
        raise SystemExit(f"{engine_path} exists; pass --force to rebuild")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"ONNX: {artifact.onnx_path.name}  ({artifact.height}x{artifact.width})")
    print(f"engine -> {engine_path}   fp16={not args.fp32}")
    print("building; this takes several minutes at the maximum optimisation level...")

    build_engine(
        onnx_path=artifact.onnx_path,
        engine_path=engine_path,
        onnx_sha256=artifact.onnx_sha256,
        fp16=not args.fp32,
        workspace_gb=args.workspace_gb,
    )
    print(f"\ndone. {engine_path.name} and its sidecar are in {engine_path.parent}")


if __name__ == "__main__":
    main()
