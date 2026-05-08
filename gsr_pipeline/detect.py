"""
detect.py — YOLOv8n person/ball detector.

Uses the Ultralytics YOLOv8 implementation.  The nano model (yolov8n.pt, ~6 MB)
is downloaded automatically by ultralytics on first run.

Output columns added to the per-frame DataFrame:
    bbox_ltwh       (l, t, w, h) in pixel coordinates
    bbox_conf       detection confidence
    class_id        0 = person, 32 = sports ball (COCO class ids)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# COCO class IDs we care about
_PERSON_CLASS = 0
_BALL_CLASS = 32


class YOLOv8nDetector:
    """Thin wrapper around Ultralytics YOLOv8n for person + ball detection."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        batch_size: int = 8,
        classes: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            model_path:      Path to a .pt weights file or an ultralytics model
                             name (e.g. 'yolov8n.pt').  Will auto-download.
            conf_threshold:  Minimum detection confidence.
            iou_threshold:   NMS IoU threshold.
            device:          Torch device string ('cpu', 'cuda', 'cuda:0', …).
            batch_size:      Number of frames to process per forward pass.
            classes:         COCO class IDs to keep.  Defaults to [0, 32]
                             (person + sports ball).
        """
        from ultralytics import YOLO  # lazy import

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.batch_size = batch_size
        self.classes = classes if classes is not None else [_PERSON_CLASS, _BALL_CLASS]

        log.info(
            "YOLOv8nDetector loaded: model=%s  conf=%.2f  device=%s",
            model_path, conf_threshold, device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_sequence(
        self, frames: List[np.ndarray]
    ) -> List[pd.DataFrame]:
        """Run detection on a list of BGR frames.

        Returns a list (one element per frame) of DataFrames with columns:
            bbox_ltwh, bbox_conf, class_id
        Each row is one detection.
        """
        results_per_frame: List[pd.DataFrame] = []

        # Process in batches
        for batch_start in range(0, len(frames), self.batch_size):
            batch = frames[batch_start : batch_start + self.batch_size]
            batch_results = self._run_batch(batch)
            results_per_frame.extend(batch_results)

        return results_per_frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_batch(self, batch: List[np.ndarray]) -> List[pd.DataFrame]:
        """Run YOLO on one batch of frames (BGR numpy arrays)."""
        results = self.model(
            batch,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                out.append(pd.DataFrame(columns=["bbox_ltwh", "bbox_conf", "class_id"]))
                continue

            rows: List[Dict[str, Any]] = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                cls = int(box.cls[0].cpu())
                rows.append(
                    {
                        "bbox_ltwh": (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                        "bbox_ltrb": (float(x1), float(y1), float(x2), float(y2)),
                        "bbox_conf": conf,
                        "class_id": cls,
                        "conf": conf,    # Duplicate for tracker compatibility
                        "cls_id": cls,  # Duplicate for tracker compatibility
                    }
                )

            out.append(pd.DataFrame(rows))

        return out
