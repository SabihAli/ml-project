"""
calibration_live.py -- calibration off the critical path
(docs/path_b/plans/IMPLEMENTATION_PLAN.md section 7, items 2-4).

Full NBJW (two HRNets + homography fit) runs every `calib_every_k` frames, or sooner on a
trigger. Every other frame's homography is propagated from the last NBJW result ("anchor")
by the camera motion accumulated since it:

    H_t = H_anchor . M(anchor -> t)^-1

where H maps image pixels to pitch metres and M maps anchor-frame pixels to frame-t pixels
(camera_motion.py). Triggers for an early NBJW run: scene cut, motion not estimable, low
motion inlier ratio, a broken motion chain, zoom change since the anchor beyond a limit, or
the anchor older than `max_h_age_frames`. Those event triggers are rate-limited to one NBJW run
per `min_trigger_interval_frames`, so a shot where NBJW keeps failing does not run it on every
frame (docs/path_b/plans/OPTIMIZATION_PLAN.md, 0.2).

`mode`:
  inline -- NBJW runs synchronously on the frame that needs it. K=1 inline is exactly the
            old per-frame behaviour (the accuracy reference) and skips motion estimation.
  async  -- NBJW runs on one background thread; the frame loop never waits for it (except
            once, for the very first anchor). A result for frame a arrives at frame t > a
            and is re-anchored by composing the per-frame motions after a, kept in a ring.

Each result carries `h_source` (fresh / propagated / reused / none) and `h_age_frames` so
consumers and evaluation can see how a position was computed (plan section 7 item 4).
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gsr_pipeline.calibrate import NBJWHomography
from gsr_pipeline.camera_motion import CameraMotionEstimator

log = logging.getLogger(__name__)

H_FRESH, H_PROPAGATED, H_REUSED, H_NONE = "fresh", "propagated", "reused", "none"
_RATE_LIMITED = frozenset({"no_motion", "low_inliers", "chain_broken", "zoom", "max_age"})


@dataclass
class CalibResult:
    H: Optional[np.ndarray]                         # image px -> pitch m; None if unknown
    cam_params: Dict = field(default_factory=dict)
    keypoints: Dict = field(default_factory=dict)   # only on fresh frames
    lines: Dict = field(default_factory=dict)       # only on fresh frames
    h_source: str = H_NONE
    h_age_frames: int = 0
    nbjw_ran: bool = False                          # an inline NBJW forward ran on this call
    trigger: Optional[str] = None                   # why NBJW was requested on this frame


class LiveCalibrator:
    """Per-frame homography with NBJW run every K frames / on triggers. One per stream."""

    def __init__(
        self,
        kp_detector,
        image_width: int = 1920,
        image_height: int = 1080,
        use_prev_homography: bool = True,
        compute_cam_params: bool = False,
        calib_every_k: int = 1,
        mode: str = "inline",
        max_h_age_frames: int = 50,
        min_inlier_ratio: float = 0.5,
        max_zoom_change: float = 0.15,
        min_trigger_interval_frames: int = 3,
        ring_size: int = 256,
        motion: Optional[Dict[str, Any]] = None,
    ) -> None:
        if mode not in ("inline", "async"):
            raise ValueError(f"calibration mode must be inline or async, got {mode!r}")
        self.kp_detector = kp_detector
        # The calibrator owns the "reuse last H" fallback, so the fitter must report failure.
        self.homography = NBJWHomography(image_width, image_height, use_prev_homography=False,
                                         compute_cam_params=compute_cam_params)
        self.use_prev = use_prev_homography
        self.k = max(1, int(calib_every_k))
        self.mode = mode
        self.max_h_age = max_h_age_frames
        self.min_inlier_ratio = min_inlier_ratio
        self.max_log_zoom = math.log1p(max_zoom_change)
        self.min_trigger_interval = max(1, int(min_trigger_interval_frames))
        self.ring_size = ring_size
        self.use_motion = not (self.k == 1 and mode == "inline")
        self.motion = CameraMotionEstimator(**(motion or {})) if self.use_motion else None

        self._anchor_H: Optional[np.ndarray] = None
        self._anchor_params: Dict = {}
        self._anchor_frame: Optional[int] = None
        self._M_cum = np.eye(3)              # anchor -> current frame pixels
        self._motion_ok = False              # _M_cum is an unbroken chain from the anchor
        self._ring: "OrderedDict[int, Optional[np.ndarray]]" = OrderedDict()  # frame -> M(prev->frame)
        self._last_submit: Optional[int] = None
        self._last_out: Optional[np.ndarray] = None

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nbjw") if mode == "async" else None
        self._pending: Optional[Tuple[int, Future]] = None

    # -- NBJW ----------------------------------------------------------------------------

    def _run_nbjw(self, frame_bgr: np.ndarray):
        keypoints, lines = self.kp_detector.detect(frame_bgr)
        H, params = self.homography.compute(keypoints)
        return H, params, keypoints, lines

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    # -- anchor bookkeeping ----------------------------------------------------------------

    def _set_anchor(self, H, params, frame_idx: int, M_since=None) -> None:
        self._anchor_H, self._anchor_params, self._anchor_frame = H, params, frame_idx
        self._M_cum = np.eye(3) if M_since is None else M_since
        self._motion_ok = True

    def _motion_since(self, a: int, t: int) -> Optional[np.ndarray]:
        """Composed motion a -> t from the ring; None if a step is broken or evicted."""
        if a not in self._ring:
            return None
        M, seen_a = np.eye(3), False
        for f, step in self._ring.items():
            if f == a:
                seen_a = True
                continue
            if not seen_a or f > t:
                continue
            if step is None:
                return None
            M = step @ M
        return M

    def _propagated(self) -> np.ndarray:
        H = self._anchor_H @ np.linalg.inv(self._M_cum)
        return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

    # -- per frame -------------------------------------------------------------------------

    def update(self, frame_idx: int, frame_bgr: np.ndarray,
               boxes_ltrb: Optional[np.ndarray] = None) -> CalibResult:
        trigger = None
        if self.use_motion:
            est = self.motion.update(frame_bgr, boxes_ltrb)
            usable = est.M is not None and est.inlier_ratio >= self.min_inlier_ratio
            self._ring[frame_idx] = est.M if usable else None
            while len(self._ring) > self.ring_size:
                self._ring.popitem(last=False)
            if self._anchor_H is not None:
                if usable and self._motion_ok:
                    self._M_cum = est.M @ self._M_cum
                else:
                    self._motion_ok = False
            if est.scene_cut:
                trigger = "scene_cut"
            elif self._anchor_H is not None and not self._motion_ok:
                trigger = "no_motion" if est.M is None else (
                    "low_inliers" if not usable else "chain_broken")

        if trigger is None:
            age = None if self._anchor_frame is None else frame_idx - self._anchor_frame
            if self._last_submit is None:
                trigger = "first"
            elif frame_idx - self._last_submit >= self.k:
                trigger = "every_k"
            elif self._anchor_H is not None and self._motion_ok:
                zoom = float(np.sqrt(abs(np.linalg.det(self._M_cum[:2, :2]))))
                if abs(math.log(max(zoom, 1e-6))) > self.max_log_zoom:
                    trigger = "zoom"
            if trigger is None and age is not None and age > self.max_h_age:
                trigger = "max_age"

        # Event triggers repeat on every frame of a shot where NBJW keeps failing (close-ups,
        # replays): motion fails or the anchor stays stale, NBJW finds no H, the fallback reuses
        # the last H and the next frame triggers again -- a full NBJW run per frame exactly where
        # it is useless. Rate-limit them; first / scene_cut / every_k are not limited.
        if (trigger in _RATE_LIMITED and self._last_submit is not None
                and frame_idx - self._last_submit < self.min_trigger_interval):
            trigger = "every_k" if frame_idx - self._last_submit >= self.k else None

        if self.mode == "inline":
            return self._update_inline(frame_idx, frame_bgr, trigger)
        return self._update_async(frame_idx, frame_bgr, trigger)

    def _fallback(self, frame_idx: int, trigger, nbjw_ran: bool) -> CalibResult:
        """No fresh H on this frame: propagate the anchor if the motion chain holds, else
        reuse the last emitted H (the pre-propagation behaviour), else nothing."""
        age = 0 if self._anchor_frame is None else frame_idx - self._anchor_frame
        if self.use_motion and self._anchor_H is not None and self._motion_ok:
            H = self._propagated()
            self._last_out = H
            return CalibResult(H, self._anchor_params, h_source=H_PROPAGATED, h_age_frames=age,
                               nbjw_ran=nbjw_ran, trigger=trigger)
        if self.use_prev and self._last_out is not None:
            if self.use_motion:
                # Keep the anchor's age, but restart the motion chain from here so the
                # reused H is at least carried along with the camera from now on.
                self._M_cum, self._motion_ok = np.eye(3), True
                self._anchor_H = self._last_out
            return CalibResult(self._last_out, self._anchor_params, h_source=H_REUSED,
                               h_age_frames=age, nbjw_ran=nbjw_ran, trigger=trigger)
        return CalibResult(None, {}, h_source=H_NONE, nbjw_ran=nbjw_ran, trigger=trigger)

    def _update_inline(self, frame_idx, frame_bgr, trigger) -> CalibResult:
        if trigger is None:
            return self._fallback(frame_idx, None, nbjw_ran=False)
        self._last_submit = frame_idx
        H, params, kps, lines = self._run_nbjw(frame_bgr)
        if H is None:
            return self._fallback(frame_idx, trigger, nbjw_ran=True)
        self._set_anchor(H, params, frame_idx)
        self._last_out = H
        return CalibResult(H, params, kps, lines, H_FRESH, 0, nbjw_ran=True, trigger=trigger)

    def _submit(self, frame_idx: int, frame_bgr: np.ndarray) -> None:
        self._last_submit = frame_idx
        self._pending = (frame_idx, self._pool.submit(self._run_nbjw, frame_bgr.copy()))

    def _update_async(self, frame_idx, frame_bgr, trigger) -> CalibResult:
        if self._pending is None and trigger is not None:
            self._submit(frame_idx, frame_bgr)
        # Bootstrap: with nothing to propagate yet, the first result is worth waiting for.
        if self._pending is not None and self._anchor_H is None and self._last_out is None:
            self._pending[1].result()

        fresh_here = None
        if self._pending is not None and self._pending[1].done():
            a, fut = self._pending
            self._pending = None
            try:
                H, params, kps, lines = fut.result()
            except Exception:  # noqa: BLE001 -- a failed background run must not kill the stream
                log.exception("background NBJW run for frame %d failed", a)
                H = None
            if H is not None:
                M_since = np.eye(3) if a == frame_idx else self._motion_since(a, frame_idx)
                if M_since is not None:
                    self._set_anchor(H, params, a, M_since=M_since)
                    if a == frame_idx:
                        fresh_here = (H, params, kps, lines)
                else:
                    log.debug("NBJW result for frame %d dropped: motion chain to %d broken", a, frame_idx)
            # A trigger that arrived while the worker was busy is served now.
            if trigger is not None and self._last_submit != frame_idx:
                self._submit(frame_idx, frame_bgr)

        if fresh_here is not None:
            H, params, kps, lines = fresh_here
            self._last_out = H
            return CalibResult(H, params, kps, lines, H_FRESH, 0, trigger=trigger)
        return self._fallback(frame_idx, trigger, nbjw_ran=False)
