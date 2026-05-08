"""
track.py — ByteTrack multi-object tracker.

Uses Ultralytics' built-in ByteTrack (no extra installation required).
Replaces the original bpbreid_strong_sort tracker from the baseline.

Output columns added to the per-frame DataFrame:
    track_id    integer track identifier (NaN for unassociated detections)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class ByteTracker:
    """
    Frame-by-frame ByteTrack tracker using Ultralytics' implementation.

    Takes per-frame detection DataFrames (with bbox_ltwh + bbox_conf) and
    assigns track_id, keeping it consistent across frames.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 25,
    ) -> None:
        """
        Args:
            track_high_thresh:  Detection confidence threshold for high-quality
                                detections.
            track_low_thresh:   Lower confidence threshold for second-pass
                                matching.
            new_track_thresh:   Minimum confidence to start a new track.
            track_buffer:       Number of frames to keep a lost track alive.
            match_thresh:       IoU matching threshold.
            frame_rate:         Video frame rate (affects buffer duration).
        """
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            from types import SimpleNamespace

            args = SimpleNamespace(
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                fuse_score=True,
            )
            self._tracker = BYTETracker(args, frame_rate=frame_rate)
        except ImportError:
            # Fallback: try the older ultralytics internal path
            try:
                from ultralytics.trackers import BYTETracker
                from types import SimpleNamespace

                args = SimpleNamespace(
                    track_high_thresh=track_high_thresh,
                    track_low_thresh=track_low_thresh,
                    new_track_thresh=new_track_thresh,
                    track_buffer=track_buffer,
                    match_thresh=match_thresh,
                    fuse_score=True,
                )
                self._tracker = BYTETracker(args, frame_rate=frame_rate)
            except ImportError as e:
                raise ImportError(
                    "Could not import BYTETracker from ultralytics. "
                    "Please make sure ultralytics>=8.0.0 is installed."
                ) from e

        log.info("ByteTracker initialised (buffer=%d frames)", track_buffer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections_df: pd.DataFrame,
        frame_shape: tuple,
    ) -> pd.DataFrame:
        """Update the tracker with detections from one frame.

        Args:
            detections_df:  DataFrame with columns ['bbox_ltwh', 'bbox_conf'].
            frame_shape:    (height, width) of the frame image.

        Returns:
            detections_df with new column 'track_id' (int, NaN if unmatched).
        """
        detections_df = detections_df.copy()
        detections_df["track_id"] = np.nan

        if detections_df.empty:
            return detections_df

        # Prepare detections for the tracker
        # Ultralytics BYTETracker expects an object with .conf, .xyxy, and .cls attributes
        # or it can be a Results object.
        
        class TrackerInput:
            def __init__(self, conf, xyxy, cls):
                self.conf = conf
                self.xyxy = xyxy
                self.cls = cls
                
                # Calculate xywh: [center_x, center_y, width, height]
                if len(xyxy) > 0:
                    w = xyxy[:, 2] - xyxy[:, 0]
                    h = xyxy[:, 3] - xyxy[:, 1]
                    cx = xyxy[:, 0] + w / 2
                    cy = xyxy[:, 1] + h / 2
                    self.xywh = np.stack([cx, cy, w, h], axis=1)
                else:
                    self.xywh = np.empty((0, 4))

                # Also include .boxes for deeper compatibility
                self.boxes = type('Boxes', (), {
                    'conf': self.conf,
                    'xyxy': self.xyxy,
                    'xywh': self.xywh,
                    'cls': self.cls
                })

            def __getitem__(self, index):
                return TrackerInput(self.conf[index], self.xyxy[index], self.cls[index])

            def __len__(self):
                return len(self.conf)

        tracker_input = TrackerInput(
            detections_df["conf"].values,
            np.vstack(detections_df["bbox_ltrb"].values),
            detections_df["cls_id"].values
        )
        
        # ByteTracker.update(results, img=None)
        tracks = self._tracker.update(tracker_input)

        if tracks is None or len(tracks) == 0:
            return detections_df

        # Match tracks back to original detections by IoU
        self._assign_track_ids(detections_df, tracks)
        return detections_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _df_to_numpy(df: pd.DataFrame) -> np.ndarray:
        """Convert a detections DataFrame to a (N, 6) numpy array."""
        rows = []
        for _, row in df.iterrows():
            l, t, w, h = row["bbox_ltwh"]
            conf = float(row.get("bbox_conf", 0.5))
            cls = float(row.get("class_id", 0))
            rows.append([l, t, l + w, t + h, conf, cls])
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _assign_track_ids(
        self, detections_df: pd.DataFrame, tracks: np.ndarray
    ) -> None:
        """Greedy IoU matching between tracks and original detections.

        Modifies detections_df in-place, setting 'track_id'.
        tracks: array of shape (M, 5+) where columns are [x1, y1, x2, y2, track_id, …]
        """
        det_boxes = []
        for _, row in detections_df.iterrows():
            l, t, w, h = row["bbox_ltwh"]
            det_boxes.append([l, t, l + w, t + h])
        det_boxes = np.array(det_boxes, dtype=np.float32)

        used = set()
        for track in tracks:
            t_box = track[:4]
            t_id = int(track[4])
            best_iou = 0.0
            best_idx: Optional[int] = None
            for i, d_box in enumerate(det_boxes):
                if i in used:
                    continue
                iou = self._iou(t_box, d_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx is not None and best_iou > 0.3:
                detections_df.iloc[best_idx, detections_df.columns.get_loc("track_id")] = t_id
                used.add(best_idx)
