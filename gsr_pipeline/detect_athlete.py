"""
detect_athlete.py -- Path B's 2-class athlete/ball detector.

Replaces detect.py's 4-class YOLOv8n (Path A) per the 2026-09-23 decision in
docs/path_b/DETECTOR_AND_DATA_BLOCKERS.md. Two differences matter beyond the class list:

  * **No Ultralytics on the hot path.** detect.py calls `YOLO(...)` and then loops over
    boxes calling `.cpu()` on each, where every call is a GPU synchronisation. Here the
    forward pass goes through gsr_pipeline/runtime's tiered runners and post-processing
    stays on-device until one transfer at the end.
  * **No roles.** The model emits `athlete` and `ball` only. player/goalkeeper/referee are
    recovered downstream by athlete_roles.py. This is the Path B split, and it is why
    AGENTS.md rule 7 ("role is owned by the detector") no longer describes this path.

NMS runs here, host-side, shared by every backend tier -- the checkpoint's head is a plain
`Detect` with end2end=False, so it is required (IMPLEMENTATION_PLAN.md section 3, corrected
2026-09-26). The iou and max_det come from the artifact sidecar, which carries the training
config's 0.7 / 300 rather than detect.py's Path A 0.45.

The output DataFrame keeps detect.py's column names so existing consumers
(evaluate_detection.py, the tracker's expectations) keep working unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from .runtime.backend import Backend, ModelArtifact, select_backend
from .runtime.postprocess import postprocess, preprocess

log = logging.getLogger(__name__)

# Path B class IDs, fixed by the checkpoint and asserted at load by ModelArtifact.load.
ATHLETE_CLASS = 0
BALL_CLASS = 1

_EMPTY_COLUMNS = ["bbox_ltwh", "bbox_ltrb", "bbox_conf", "class_id", "conf", "cls_id"]


class AthleteBallDetector:
    """The 2-class detector, running on the best available backend tier."""

    def __init__(
        self,
        onnx_path: str = "runs/yolo26s_athlete_ball/best_736x1280.onnx",
        athlete_conf: float = 0.25,
        ball_conf: float = 0.15,
        device: str = "cuda:0",
        backend: Optional[str] = None,
        batch_size: int = 8,
        auto_build_engine: bool = True,
    ) -> None:
        """
        Args:
            onnx_path: exported ONNX; its `.json` sidecar must sit beside it.
            athlete_conf: score floor for athletes. 0.25 is the blockers-doc reference
                point (97.0% / 95.5% people recall/precision on SNGS-021).
            ball_conf: score floor for the ball, deliberately lower -- the ball is the
                weakest class and the one every downstream event depends on. 0.15 gives
                79% / 71% open-play recall/precision; 0.25 gives 77% / 78%.
            device: torch device string for the GPU tiers; ignored on the CPU tier.
            backend: pin one tier ('tensorrt' | 'ort_gpu' | 'ort_cpu') instead of walking
                the fallback chain. Used by the parity gate. A pinned tier that fails
                raises rather than falling back.
            batch_size: frames per process_sequence chunk. The engine is static batch 1,
                so this is a loop bound, not a real batch -- kept for API compatibility
                with detect.py and for the batch path's future batch-8 profile (plan 11).
            auto_build_engine: build the TensorRT engine at startup if it is missing (first
                launch on a host; takes minutes, once). False keeps the old behaviour: no
                engine means falling back to ONNX Runtime.
        """
        self.artifact = ModelArtifact.load(onnx_path)
        self.athlete_conf = athlete_conf
        self.ball_conf = ball_conf
        self.batch_size = batch_size

        selection = select_backend(
            self.artifact,
            prefer=Backend(backend) if backend else None,
            device=device,
            auto_build=auto_build_engine,
        )
        self.selection = selection
        self.runner = selection.runner
        self.backend = selection.backend.value
        self.device = selection.device

        self.conf_by_class = {ATHLETE_CLASS: athlete_conf, BALL_CLASS: ball_conf}
        self.names = self.artifact.names

        # Preallocated input buffer -- reused across frames (plan principle 4).
        self._input = torch.empty(
            (1, 3, self.artifact.height, self.artifact.width),
            dtype=torch.float32,
            device=self.device,
        )

        log.info(
            "AthleteBallDetector: backend=%s device=%s imgsz=%dx%d athlete_conf=%.2f "
            "ball_conf=%.2f nms(iou=%.2f, max_det=%d)",
            self.backend, self.device, self.artifact.height, self.artifact.width,
            athlete_conf, ball_conf, self.artifact.nms_iou, self.artifact.nms_max_det,
        )

    def detect(self, frame_bgr: np.ndarray) -> pd.DataFrame:
        """Run detection on one BGR frame. Returns detect.py's column layout."""
        x, params = preprocess(
            frame_bgr,
            self.artifact.height,
            self.artifact.width,
            device=self.device,
            out=self._input,
        )
        raw = self.runner.infer(x)
        boxes, scores, class_ids = postprocess(
            raw.to(torch.float32),
            params,
            self.conf_by_class,
            iou=self.artifact.nms_iou,
            max_det=self.artifact.nms_max_det,
        )

        # One transfer, after post-processing -- not one per box.
        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy().astype(float)
        classes_np = class_ids.detach().cpu().numpy().astype(int)

        if len(boxes_np) == 0:
            return pd.DataFrame(columns=_EMPTY_COLUMNS)

        ltrb = [tuple(float(v) for v in box) for box in boxes_np]
        ltwh = [(l, t, r - l, b - t) for l, t, r, b in ltrb]
        return pd.DataFrame(
            {
                "bbox_ltwh": ltwh,
                "bbox_ltrb": ltrb,
                "bbox_conf": scores_np,
                "class_id": classes_np,
                # Duplicated for tracker compatibility, as in detect.py.
                "conf": scores_np,
                "cls_id": classes_np,
            }
        )

    def process_sequence(self, frames: List[np.ndarray]) -> List[pd.DataFrame]:
        """Run detection over a list of BGR frames, one DataFrame each.

        Mirrors YOLOv8nDetector.process_sequence so evaluate_detection.py can drive either
        detector through the same code path.
        """
        return [self.detect(frame) for frame in frames]

    def close(self) -> None:
        if hasattr(self.runner, "close"):
            self.runner.close()

    @staticmethod
    def default_onnx_for(imgsz: tuple[int, int], weights_dir: str | Path) -> Path:
        """Path of the exported ONNX for a given (height, width)."""
        height, width = imgsz
        return Path(weights_dir) / f"best_{height}x{width}.onnx"
