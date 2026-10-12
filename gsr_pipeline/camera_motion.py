"""
camera_motion.py -- frame-to-frame camera motion as an image homography.

Lets calibration run NBJW every K frames instead of every frame
(docs/path_b/plans/IMPLEMENTATION_PLAN.md section 7): between NBJW runs the pitch
homography is propagated by the camera motion measured here.

Method: Shi-Tomasi corners on a downscaled greyscale frame, tracked into the next frame
with pyramidal Lucas-Kanade, then a RANSAC homography. Detection boxes are masked out of
corner selection because players move independently of the camera; the pitch, its lines,
the stands and the ad boards are what should drive the estimate. The broadcast camera
pans/tilts/zooms about a near-fixed centre, so one homography models it exactly for the
ground plane.

M maps pixel coordinates in the previous frame to the current frame, in full-resolution
pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MotionEstimate:
    M: Optional[np.ndarray]     # 3x3, prev -> cur full-res pixels; None when not estimable
    n_tracked: int              # corners successfully tracked
    inlier_ratio: float         # RANSAC inliers / tracked, 0 when M is None
    scale: float                # sqrt(|det| of M's linear part): >1 zoom in, <1 zoom out
    scene_cut: bool             # appearance changed too much to be the same shot


class CameraMotionEstimator:
    """Stateful: call `update(frame)` once per frame, in order."""

    def __init__(
        self,
        work_width: int = 640,
        max_corners: int = 400,
        quality_level: float = 0.01,
        min_distance: int = 8,
        ransac_px: float = 2.0,
        min_tracked: int = 30,
        scene_cut_hist_corr: float = 0.6,
        box_margin_px: int = 6,
    ) -> None:
        self.work_width = work_width
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_px = ransac_px
        self.min_tracked = min_tracked
        self.scene_cut_hist_corr = scene_cut_hist_corr
        self.box_margin_px = box_margin_px
        self._prev_grey: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._prev_mask: Optional[np.ndarray] = None
        self._s = 1.0          # work / full-res scale

    def reset(self) -> None:
        self._prev_grey = self._prev_hist = self._prev_mask = None

    def _prepare(self, frame_bgr: np.ndarray, boxes_ltrb: Optional[np.ndarray]):
        h, w = frame_bgr.shape[:2]
        self._s = self.work_width / float(w)
        size = (self.work_width, max(1, int(round(h * self._s))))
        grey = cv2.cvtColor(cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([grey], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        mask = np.full(grey.shape, 255, np.uint8)
        if boxes_ltrb is not None and len(boxes_ltrb):
            m = self.box_margin_px
            for l, t, r, b in np.asarray(boxes_ltrb, np.float64).reshape(-1, 4) * self._s:
                mask[max(0, int(t) - m):int(b) + m, max(0, int(l) - m):int(r) + m] = 0
        return grey, hist, mask

    def update(self, frame_bgr: np.ndarray, boxes_ltrb: Optional[np.ndarray] = None) -> MotionEstimate:
        """Motion from the previous frame to this one. `boxes_ltrb` (full-res) are this
        frame's detections; they are masked out when this frame becomes the next frame's
        reference."""
        grey, hist, mask = self._prepare(frame_bgr, boxes_ltrb)
        prev_grey, prev_hist, prev_mask = self._prev_grey, self._prev_hist, self._prev_mask
        self._prev_grey, self._prev_hist, self._prev_mask = grey, hist, mask

        if prev_grey is None or prev_grey.shape != grey.shape:
            return MotionEstimate(None, 0, 0.0, 1.0, scene_cut=prev_grey is not None)

        corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
        if corr < self.scene_cut_hist_corr:
            return MotionEstimate(None, 0, 0.0, 1.0, scene_cut=True)

        p0 = cv2.goodFeaturesToTrack(prev_grey, self.max_corners, self.quality_level,
                                     self.min_distance, mask=prev_mask)
        if p0 is None or len(p0) < self.min_tracked:
            return MotionEstimate(None, 0 if p0 is None else len(p0), 0.0, 1.0, scene_cut=False)

        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_grey, grey, p0, None, winSize=(21, 21), maxLevel=3)
        good = st.reshape(-1) == 1
        p0, p1 = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        n = len(p0)
        if n < self.min_tracked:
            return MotionEstimate(None, n, 0.0, 1.0, scene_cut=False)

        Mw, inl = cv2.findHomography(p0, p1, cv2.RANSAC, self.ransac_px)
        if Mw is None:
            return MotionEstimate(None, n, 0.0, 1.0, scene_cut=False)

        # Work-resolution homography -> full resolution: M = S^-1 . Mw . S
        S = np.diag([self._s, self._s, 1.0])
        M = np.linalg.inv(S) @ Mw @ S
        M /= M[2, 2]
        scale = float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))
        return MotionEstimate(M, n, float(inl.sum()) / n, scale, scene_cut=False)
