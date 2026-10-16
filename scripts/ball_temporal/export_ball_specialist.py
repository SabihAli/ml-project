"""
export_ball_specialist.py -- step 5 of docs/path_b/plans/BALL_TEMPORAL_PLAN.md §9: export a
temporal ball specialist checkpoint to ONNX with a sidecar, and check the ONNX against PyTorch.

The TensorRT engine is not built here. The pipeline builds it from this ONNX on first launch on
each host (gsr_pipeline/runtime/trt_runner.ensure_engine), as <stem>_fp16.engine beside it.

Graph (static batch 1):
  input  "frames"  float32 (1, 9, 640, 640): crops of frames t-2, t-1, t, RGB, 0..255, in that
                   order. Normalisation and the 5-channel difference images are inside the graph,
                   so the runtime only cuts and stacks crops.
  output "heatmap" float32 (1, 1, 320, 320): logits; sigmoid = ball-centre confidence per cell.
  output "offset"  float32 (1, 2, 320, 320): sub-pixel offset per cell.
  Decode (host): peak cell (ix, iy), point = ((ix, iy) + offset[:, iy, ix]) * 2 in crop pixels,
  confidence = sigmoid(peak).

Sidecar <onnx>.json: checkpoint, epoch, channels, input/output contract, stride, the system
threshold from step 4, and the ONNX sha256 (the engine sidecar is checked against it).

Parity: ONNX Runtime (CPU, and GPU if available) against PyTorch FP32 on real crops from the
step 2 dataset. Pass: every decoded point within 0.5 px and confidence within 1e-3.

    python scripts/ball_temporal/export_ball_specialist.py --ckpt runs/ball_temporal/c5b/best.pt \
        --out runs/ball_temporal/c5b/ball_specialist_c5b.onnx --threshold 0.75
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heatmap_model import STRIDE, BallHeatmapNet, build_input, decode  # noqa: E402

CROP = 640
POINT_TOL_PX = 0.5
CONF_TOL = 1e-3


class ExportWrapper(nn.Module):
    """float 0..255 frames in, (heatmap logits, offset) out; preprocessing inside the graph."""

    def __init__(self, net: BallHeatmapNet):
        super().__init__()
        self.net = net

    def forward(self, frames):
        return self.net(build_input(frames, self.net.channels))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def load_crops(data: Path, n: int, seed: int) -> list[np.ndarray]:
    """n crops (9, 640, 640) float32 from the step 2 test split, mixed kinds."""
    import csv

    import cv2
    with open(data / "manifest.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(len(rows), min(n, len(rows)), replace=False):
        img = cv2.cvtColor(cv2.imread(str(data / rows[i]["file"])), cv2.COLOR_BGR2RGB)
        f = np.stack([img[:, k * CROP:(k + 1) * CROP] for k in range(3)])     # (3, H, W, 3)
        out.append(f.transpose(0, 3, 1, 2).reshape(9, CROP, CROP).astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="ONNX path; the sidecar is written beside it")
    ap.add_argument("--threshold", type=float, required=True, help="system threshold from step 4, recorded in the sidecar")
    ap.add_argument("--data", type=Path, default=Path("data/outputs/ball_temporal/dataset"))
    ap.add_argument("--n", type=int, default=64, help="parity crops")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = BallHeatmapNet(ck["channels"], pretrained=False)
    net.load_state_dict(ck["model"])
    model = ExportWrapper(net).eval()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 9, CROP, CROP)
    torch.onnx.export(model, dummy, str(a.out), input_names=["frames"], output_names=["heatmap", "offset"],
                      opset_version=a.opset, do_constant_folding=True, dynamo=False)
    meta = {
        "schema": 1,
        "model": "ball_temporal_heatmap",
        "checkpoint": str(a.ckpt).replace("\\", "/"),
        "epoch": ck.get("epoch"),
        "channels": ck["channels"],
        "input": {"name": "frames", "shape": [1, 9, CROP, CROP], "dtype": "float32",
                  "layout": "RGB crops of frames t-2, t-1, t, values 0..255; preprocessing in graph"},
        "outputs": {"heatmap": {"shape": [1, 1, CROP // STRIDE, CROP // STRIDE], "activation": "sigmoid (host)"},
                    "offset": {"shape": [1, 2, CROP // STRIDE, CROP // STRIDE]}},
        "stride": STRIDE,
        "crop": CROP,
        "threshold": a.threshold,
        "onnx_sha256": sha256(a.out),
    }
    Path(str(a.out) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB) and its sidecar")

    import onnxruntime as ort
    crops = load_crops(a.data, a.n, seed=0)
    with torch.no_grad():
        ref = [decode(*model(torch.from_numpy(c)[None]))[0].numpy() for c in crops]
    providers = [p for p in ("CPUExecutionProvider", "CUDAExecutionProvider") if p in ort.get_available_providers()]
    ok_all = True
    for prov in providers:
        sess = ort.InferenceSession(str(a.out), providers=[prov])
        if prov not in sess.get_providers():
            print(f"{prov}: requested but not active, skipped")
            continue
        dp, dc = [], []
        for c, r in zip(crops, ref):
            hm, off = sess.run(None, {"frames": c[None]})
            p = decode(torch.from_numpy(hm), torch.from_numpy(off))[0].numpy()
            dp.append(float(np.hypot(*(p[:2] - r[:2]))))
            dc.append(float(abs(p[2] - r[2])))
        ok = max(dp) <= POINT_TOL_PX and max(dc) <= CONF_TOL
        ok_all &= ok
        print(f"{prov}: {len(crops)} crops, max point diff {max(dp):.4f} px, max conf diff {max(dc):.2e} "
              f"-> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
