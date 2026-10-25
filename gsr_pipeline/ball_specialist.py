"""
ball_specialist.py -- the temporal ball specialist (c5c) and the per-stream logic around it.

The main detector misses about 4 balls in 10 on held-out footage. When it misses, the
specialist looks at frames t-2, t-1 and t through a 640x640 window centred on the last
sighting and predicts a point. Measured on the 15 test sequences (PERF_LOG.md, 2026-10-16):
ball recall 59.4% -> 69.5%, precision 85.3% -> 86.5%.

`BallFuser.update` is the runtime form of the evaluator's fused chain
(scripts/eval/ball_det/eval_ball_temporal.py, `simulate` with --chain fused). Keep the two in
step, or the measured numbers stop describing what runs:

  * the main detector's top ball is the output and becomes the last sighting;
  * otherwise, if the last sighting is at most `max_miss_frames` old, the specialist runs on
    a window centred on it (shift-clamped into the frame, no camera-motion shift); a peak at or
    above the threshold is the output and becomes the new sighting (window re-centres, gap
    resets);
  * a missing t-1 or t-2 (start of a stream) is replaced by frame t, as in training.

The model is `runs/ball_temporal/c5c/ball_specialist_c5c.onnx`, exported with its
preprocessing in the graph (scripts/ball_temporal/export_ball_specialist.py): input is RGB
crops of t-2|t-1|t stacked to (1, 9, 640, 640), values 0..255; outputs are heatmap logits
(1, 1, 320, 320) and a sub-pixel offset (1, 2, 320, 320). Backend tiers are the detector's
(TensorRT FP16, built on first launch -> ORT GPU -> ORT CPU).

Not here yet: the Kalman `BallTracker` (BALL_TEMPORAL_PLAN.md section 7) and running the
specialist on its own CUDA stream in parallel with the main detector (plan section 5.1). It
runs inline, after the main detector, only on frames the main detector missed.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .runtime.backend import Backend, sha256_file

log = logging.getLogger(__name__)

BALL_CLASS = 1
MODEL_NAME = "ball_temporal_heatmap"


def clamp_window(cx: float, cy: float, width: int, height: int, crop: int) -> tuple[int, int]:
    """Top-left of a crop x crop window centred at (cx, cy), shifted (never padded) into the
    frame. Same rule as the dataset builder's clamp_window, so runtime windows match training."""
    x0, y0 = int(round(cx - crop / 2)), int(round(cy - crop / 2))
    return min(max(x0, 0), max(width - crop, 0)), min(max(y0, 0), max(height - crop, 0))


@dataclass(frozen=True)
class SpecialistArtifact:
    """The exported ONNX plus the sidecar export_ball_specialist.py writes beside it."""

    onnx_path: Path
    meta: dict

    @property
    def crop(self) -> int:
        return int(self.meta["crop"])

    @property
    def stride(self) -> int:
        return int(self.meta["stride"])

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    @property
    def onnx_sha256(self) -> str:
        return self.meta["onnx_sha256"]

    @property
    def engine_path(self) -> Path:
        return self.onnx_path.with_name(f"{self.onnx_path.stem}_fp16.engine")

    @classmethod
    def load(cls, onnx_path: str | Path) -> "SpecialistArtifact":
        onnx_path = Path(onnx_path)
        sidecar = Path(str(onnx_path) + ".json")
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} not found. Run: python scripts/ball_temporal/export_ball_specialist.py"
            )
        if not sidecar.exists():
            raise FileNotFoundError(f"{sidecar} not found; the ONNX must ship with its sidecar")
        meta = json.loads(sidecar.read_text())
        if meta.get("model") != MODEL_NAME:
            raise ValueError(f"{sidecar.name} describes {meta.get('model')!r}, not {MODEL_NAME!r}")
        shape = meta["input"]["shape"]
        if shape != [1, 9, meta["crop"], meta["crop"]]:
            raise ValueError(f"unsupported input shape {shape}; expected [1, 9, crop, crop]")
        actual = sha256_file(onnx_path)
        if actual != meta.get("onnx_sha256"):
            raise ValueError(
                f"{onnx_path.name} does not match its sidecar checksum "
                f"({actual[:12]}... != {str(meta.get('onnx_sha256'))[:12]}...). Re-export."
            )
        return cls(onnx_path=onnx_path, meta=meta)


class BallSpecialist:
    """c5c on the best available backend tier. Stateless: one call = one window."""

    def __init__(
        self,
        onnx_path: str = "runs/ball_temporal/c5c/ball_specialist_c5c.onnx",
        device: str = "cuda:0",
        backend: Optional[str] = None,
        auto_build_engine: bool = True,
    ) -> None:
        self.artifact = SpecialistArtifact.load(onnx_path)
        self.crop = self.artifact.crop
        self.stride = self.artifact.stride
        self.backend, self.runner, self.device = self._select(
            Backend(backend) if backend else None, device, auto_build_engine)
        log.info("BallSpecialist: %s backend=%s device=%s crop=%d sidecar threshold=%.2f",
                 self.artifact.onnx_path.name, self.backend.value, self.device, self.crop,
                 self.artifact.threshold)

    def _select(self, prefer: Optional[Backend], device: str, auto_build: bool):
        """The detector's tier walk (runtime/backend.select_backend), for a two-output model.
        A tier must load and pass a warm-up forward to be accepted; a pinned tier raises."""
        from .runtime.ort_runner import OrtRunner
        from .runtime.trt_runner import TrtRunner, check_sidecar, ensure_engine

        art = self.artifact
        tiers = (prefer,) if prefer else (Backend.TENSORRT, Backend.ORT_GPU, Backend.ORT_CPU)
        rejected = []
        for tier in tiers:
            try:
                if tier is not Backend.ORT_CPU and not torch.cuda.is_available():
                    raise RuntimeError("no CUDA device")
                if tier is Backend.TENSORRT:
                    ensure_engine(art.onnx_path, art.engine_path, art.onnx_sha256, auto_build=auto_build)
                    check_sidecar(art.engine_path, art.onnx_sha256)
                    runner, dev = TrtRunner(art.engine_path, device=device), torch.device(device)
                elif tier is Backend.ORT_GPU:
                    runner, dev = OrtRunner(art.onnx_path, device=device), torch.device(device)
                else:
                    runner, dev = OrtRunner(art.onnx_path, device="cpu"), torch.device("cpu")
                self.backend, self.runner, self.device = tier, runner, dev
                logits, _ = self._forward(torch.zeros((1, 9, self.crop, self.crop), device=dev))
                expected = (1, 1, self.crop // self.stride, self.crop // self.stride)
                if tuple(logits.shape) != expected:
                    raise RuntimeError(f"warm-up heatmap {tuple(logits.shape)} != {expected}")
            except Exception as exc:
                rejected.append(f"{tier.value}: {exc}")
                log.info("ball specialist backend %s unavailable: %s", tier.value, exc)
                if prefer:
                    raise RuntimeError(f"pinned backend {tier.value} unavailable: {exc}") from exc
                continue
            if tier is Backend.ORT_CPU:
                log.warning("ball specialist on the CPU tier: degraded, not real-time")
            return tier, runner, dev
        raise RuntimeError("no usable ball specialist backend. Rejections: " + "; ".join(rejected))

    def _forward(self, x: torch.Tensor):
        if self.backend is Backend.TENSORRT:
            self.runner.infer(x)
            return self.runner.buffers["heatmap"], self.runner.buffers["offset"]
        out = self.runner.infer_all(x)
        return out["heatmap"], out["offset"]

    @torch.no_grad()
    def detect(self, frames_bgr: list, x0: int, y0: int) -> tuple[float, float, float]:
        """Frames t-2, t-1, t (BGR, full size) and a window's top-left -> the strongest peak
        as (x, y, confidence), x/y in full-frame pixels."""
        c = self.crop
        crops = []
        for f in frames_bgr:
            crop = f[y0:y0 + c, x0:x0 + c, ::-1]                      # BGR -> RGB
            if crop.shape[:2] != (c, c):                              # frame smaller than the window
                padded = np.zeros((c, c, 3), np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            crops.append(crop)
        x = np.ascontiguousarray(np.concatenate(crops, axis=2).transpose(2, 0, 1))[None]
        logits, off = self._forward(torch.from_numpy(x).to(self.device, torch.float32))

        # heatmap_model.decode: the single strongest cell, refined by its offset.
        logits = logits.float().cpu()
        off = off.float().cpu()
        size = logits.shape[-1]
        p = logits.sigmoid().view(-1)
        conf, i = p.max(0)
        iy, ix = int(i) // size, int(i) % size
        ox, oy = float(off[0, 0, iy, ix]), float(off[0, 1, iy, ix])
        return (ix + ox) * self.stride + x0, (iy + oy) * self.stride + y0, float(conf)

    def close(self) -> None:
        if hasattr(self.runner, "close"):
            self.runner.close()


class BallFuser:
    """Per-sequence / per-stream ball state: picks this frame's one ball from the main detector
    or, when it missed, from the specialist. Create one per stream (or call `reset`)."""

    def __init__(
        self,
        specialist: Optional[BallSpecialist],
        threshold: Optional[float] = None,
        max_miss_frames: int = 15,
    ) -> None:
        """
        Args:
            specialist: None turns the specialist off (main detector only).
            threshold: specialist confidence floor. None = the sidecar's value (0.70 for c5c,
                chosen on `select` and frozen for the test gate).
            max_miss_frames: frames since the last sighting the specialist still runs. 15 is
                what gate 3 measured; recall past 1 frame is already under 20%.
        """
        self.specialist = specialist
        self.threshold = (specialist.artifact.threshold if specialist and threshold is None
                          else threshold)
        self.max_miss_frames = max_miss_frames
        self.reset()

    def reset(self) -> None:
        self._ring: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._last: Optional[tuple] = None          # (cx, cy, frame_idx, w, h)
        self.stats = {"main": 0, "specialist_calls": 0, "specialist_hits": 0}

    def update(self, frame_idx: int, frame_bgr: np.ndarray, balls: pd.DataFrame) -> Optional[dict]:
        """`balls`: this frame's main-detector ball rows (already at the detector's ball_conf).
        Returns one ball row (detect_athlete's columns plus `ball_source` and
        `frames_since_seen`), or None when neither model found the ball."""
        self._ring[frame_idx] = frame_bgr
        while len(self._ring) > 3:
            self._ring.popitem(last=False)

        if len(balls):
            top = balls.sort_values("bbox_conf", ascending=False).iloc[0]
            l, t, r, b = top["bbox_ltrb"]
            self._last = ((l + r) / 2, (t + b) / 2, frame_idx, r - l, b - t)
            self.stats["main"] += 1
            return self._row((l, t, r, b), float(top["bbox_conf"]), "main", 0)

        if self.specialist is None or self._last is None:
            return None
        cx, cy, seen, w, h = self._last
        gap = frame_idx - seen
        if gap > self.max_miss_frames:
            return None

        height, width = frame_bgr.shape[:2]
        x0, y0 = clamp_window(cx, cy, width, height, self.specialist.crop)
        frames = [self._ring.get(f, frame_bgr) for f in (frame_idx - 2, frame_idx - 1)] + [frame_bgr]
        px, py, conf = self.specialist.detect(frames, x0, y0)
        self.stats["specialist_calls"] += 1
        if conf < self.threshold:
            return None
        self._last = (px, py, frame_idx, w, h)
        self.stats["specialist_hits"] += 1
        # The specialist predicts a centre only; the box keeps the last main box's size so
        # bbox_ltwh consumers (pitch projection at the bottom-middle, the wire) still work.
        return self._row((px - w / 2, py - h / 2, px + w / 2, py + h / 2), conf, "specialist", gap)

    @staticmethod
    def _row(ltrb, conf: float, source: str, frames_since_seen: int) -> dict:
        l, t, r, b = (float(v) for v in ltrb)
        return {
            "bbox_ltwh": (l, t, r - l, b - t), "bbox_ltrb": (l, t, r, b),
            "bbox_conf": conf, "conf": conf, "class_id": BALL_CLASS, "cls_id": BALL_CLASS,
            "ball_source": source, "frames_since_seen": frames_since_seen,
        }
