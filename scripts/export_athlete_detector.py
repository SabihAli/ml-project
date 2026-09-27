"""
export_athlete_detector.py -- export the 2-class athlete/ball detector to ONNX.

Release-time step (docs/path_b_implementation/IMPLEMENTATION_PLAN.md section 3.2): run once
per model version on a dev machine, commit/ship the resulting .onnx + .onnx.json, and let
each production host build its own TensorRT engine from the ONNX via build_trt_engine.py.
TRT engines are specific to GPU model, driver and TRT version, so they cannot be shipped.

The graph is exported RAW -- no NMS (plan section 3.2, decision 7 of 2026-09-26). The
checkpoint's head is a plain `Detect` with end2end=False, so NMS is genuinely required, but
it runs host-side in torch, shared by every backend tier, so that TensorRT and ONNX Runtime
cannot diverge on it. If NMS were baked into the graph each runtime would use its own
kernel, and a parity-gate failure (plan section 3.5) could not be attributed to
preprocessing versus NMS.

The sidecar .onnx.json is the contract the runtime reads at load: class names, input size,
stride, output tensor layout, the NMS parameters the parity gate must match, and a checksum
of the source checkpoint. gsr_pipeline/detect_athlete.py refuses to load anything whose
names are not exactly {athlete, ball}.

Usage:
    python -m scripts.export_athlete_detector                       # 1280x736, rectangular
    python -m scripts.export_athlete_detector --imgsz 1280 1280     # square, for the A/B
    python -m scripts.export_athlete_detector --weights path/to/best.pt --out_dir model_testing/...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "model_testing" / "yolo26s_athlete_ball" / "best.pt"

# Fixed by the checkpoint; the exporter refuses to run against anything else so that a
# silently-swapped model cannot produce a plausible-looking artifact.
EXPECTED_NAMES = {0: "athlete", 1: "ball"}

# From model_testing/yolo26s_athlete_ball/args.yaml. These are the values the parity gate
# (plan section 3.5) must reproduce -- NOT detect.py's iou_threshold=0.45, which belongs to
# the old Path A detector.
TRAIN_IOU = 0.7
TRAIN_MAX_DET = 300

SIDECAR_SCHEMA = 1
OPSET = 17  # pinned: a floating opset would silently change the graph between exports


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the athlete/ball detector to ONNX")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=[736, 1280],
        help="Height and width of the static input. Default 736x1280 -- 16:9 at stride 32, "
             "which removes the ~44%% padding compute of a 1280x1280 square (plan 3.4). "
             "Pass '1280 1280' for the square variant the parity gate A/Bs against.",
    )
    parser.add_argument("--out_dir", default=None, help="Defaults to the weights' directory")
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing export")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    height, width = args.imgsz
    for name, value in (("height", height), ("width", width)):
        if value % 32:
            raise SystemExit(
                f"--imgsz {name} must be a multiple of the max stride (32); got {value}"
            )

    out_dir = Path(args.out_dir).resolve() if args.out_dir else weights.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{weights.stem}_{height}x{width}"
    onnx_path = out_dir / f"{stem}.onnx"
    sidecar_path = out_dir / f"{stem}.onnx.json"

    if onnx_path.exists() and not args.force:
        raise SystemExit(f"{onnx_path} exists; pass --force to overwrite")

    from ultralytics import YOLO  # lazy: heavy, and only the export path needs it
    import onnx
    import torch
    import ultralytics

    print(f"loading {weights}")
    model = YOLO(str(weights))

    names = {int(k): v for k, v in model.names.items()}
    if names != EXPECTED_NAMES:
        raise SystemExit(
            f"unexpected class names {names}; expected {EXPECTED_NAMES}. "
            "This script is specific to the 2-class athlete/ball detector."
        )

    head = model.model.model[-1]
    end2end = bool(getattr(head, "end2end", False))
    if end2end:
        # Not a failure, but it would invalidate the host-side-NMS decision, so it must not
        # pass silently -- the runtime would then apply NMS twice.
        raise SystemExit(
            "this checkpoint has an end-to-end (NMS-free) head, but the runtime applies NMS "
            "host-side. Revisit plan section 3.2 decision 7 before exporting."
        )

    stride = [int(s) for s in model.model.stride]
    print(f"names={names}  stride={stride}  end2end={end2end}  reg_max={getattr(head, 'reg_max', None)}")
    print(f"exporting ONNX at {height}x{width}, opset {args.opset}, nms=False (raw graph)")

    # Ultralytics writes the export next to the weights and returns its path.
    produced = model.export(
        format="onnx",
        imgsz=(height, width),
        opset=args.opset,
        simplify=True,
        dynamic=False,
        nms=False,
        half=False,  # FP16 is a TensorRT build flag (plan 3.4), not an ONNX-graph property
        device="cpu",
        verbose=False,
    )
    produced = Path(produced).resolve()
    if produced != onnx_path:
        shutil.move(str(produced), str(onnx_path))
    print(f"wrote {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Read the real output layout back off the graph rather than assuming it. The runtime
    # binds buffers from these values, so a wrong guess here is a silent corruption.
    graph = onnx.load(str(onnx_path)).graph

    def tensor_spec(t) -> dict:
        shape = [d.dim_value if d.HasField("dim_value") else d.dim_param
                 for d in t.type.tensor_type.shape.dim]
        return {"name": t.name, "shape": shape}

    inputs = [tensor_spec(t) for t in graph.input]
    outputs = [tensor_spec(t) for t in graph.output]
    if len(outputs) != 1:
        print(f"  note: {len(outputs)} graph outputs; the runtime reads the first")
    out_shape = outputs[0]["shape"]

    # YOLO detect heads emit (batch, 4 + nc, num_anchors) with boxes in xywh centre form.
    # Verify the channel count rather than trusting it, since the post-processing split
    # depends on it exactly.
    expected_ch = 4 + len(names)
    layout = "xywh+cls" if len(out_shape) == 3 and out_shape[1] == expected_ch else "unknown"
    if layout == "unknown":
        print(f"  WARNING: output shape {out_shape} does not match the expected "
              f"(1, {expected_ch}, N); detect_athlete.py will refuse this artifact")

    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_checkpoint": weights.name,
        "source_sha256": sha256_file(weights),
        "onnx_sha256": sha256_file(onnx_path),
        "imgsz": {"height": height, "width": width},
        "names": {str(k): v for k, v in sorted(names.items())},
        "stride": stride,
        "end2end": end2end,
        "nms": {
            # embedded=false is the contract: the runtime MUST apply NMS itself.
            "embedded": False,
            "iou": TRAIN_IOU,
            "max_det": TRAIN_MAX_DET,
        },
        "input": inputs[0] if inputs else None,
        "output": {**outputs[0], "layout": layout, "channels": expected_ch},
        "opset": args.opset,
        "versions": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
        },
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {sidecar_path}")
    print(f"\noutput layout: {out_shape} ({layout})")
    print("NMS is NOT in the graph -- the runtime applies it (plan 3.2, decision 7).")


if __name__ == "__main__":
    main()
