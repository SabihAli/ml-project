"""
reid_worker_pool.py — background worker pool for the expensive per-frame
appearance stages (ReID embeddings, jersey OCR), decoupled from the fast
detect->track->calibrate path. See docs/architecture/STREAMING_ARCHITECTURE.md §1.7.6.

Used only by process_stream (live mode). The batch path (`process_sequence`)
is unaffected and keeps calling `reid.process_frame`/`jersey.process_frame`
inline, synchronously, exactly as before -- this pool is additive
infrastructure for streaming, not a replacement for the batch algorithm.

Jobs are submitted per-frame; each job runs ReID embedding extraction and
jersey OCR on a worker thread and delivers its result via callback once
ready -- the caller (process_stream's main loop) never blocks on either.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class AppearanceJobResult:
    frame_idx: int
    detections_df: pd.DataFrame   # includes embeddings, color_features, jersey_number
    context: Dict[str, Any]       # whatever the submitter attached (is_pitch_view, timestamp_s, frame_duration_s)


@dataclass
class ReidWorkerPoolConfig:
    max_workers: int = 2
    # Bound the in-flight job queue so a slow ReID/jersey model can't let
    # unbounded work pile up if frames arrive faster than jobs complete --
    # once full, submit() drops the OLDEST pending frame's appearance job
    # rather than blocking the fast per-frame loop (position/role events
    # already went out for that frame regardless; only the appearance
    # features -- team/jersey -- are late/missing for it).
    max_queue_size: int = 8

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "ReidWorkerPoolConfig":
        cfg = cfg or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class ReidWorkerPool:
    """Submits (image, detections_df) appearance-extraction jobs to a
    background thread pool; results are delivered via a callback invoked
    from a worker thread (NOT the submitting thread) -- callbacks must be
    safe to run concurrently with the main per-frame loop, or do their own
    synchronization (e.g. LiveTeamAssigner is documented as not thread-safe;
    a caller wiring this pool to it must serialize those calls itself, e.g.
    via a single-threaded results queue drained by the main loop -- see
    process_stream, which does exactly that rather than calling into
    LiveTeamAssigner directly from a worker thread).
    """

    def __init__(
        self,
        reid,
        jersey,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._reid = reid
        self._jersey = jersey
        self.cfg = ReidWorkerPoolConfig.from_dict(cfg)
        self._executor = ThreadPoolExecutor(max_workers=self.cfg.max_workers, thread_name_prefix="reid-worker")
        self._results: "queue.Queue[AppearanceJobResult]" = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self._dropped = 0

    def submit(self, frame_idx: int, image_bgr, detections_df: pd.DataFrame, context: Optional[Dict[str, Any]] = None) -> None:
        """Non-blocking. Drops the job (and logs it) if the queue is already
        at capacity, rather than applying backpressure to the caller -- the
        fast per-frame loop must never block on this.
        """
        with self._lock:
            if self._pending >= self.cfg.max_queue_size:
                self._dropped += 1
                log.warning(
                    "reid_worker_pool: dropping appearance job for frame %d -- "
                    "%d in-flight jobs already at capacity (%d dropped total)",
                    frame_idx, self._pending, self._dropped,
                )
                return
            self._pending += 1

        future = self._executor.submit(self._run_job, frame_idx, image_bgr, detections_df, context or {})
        future.add_done_callback(self._on_done)

    def _run_job(self, frame_idx, image_bgr, detections_df, context) -> AppearanceJobResult:
        # reid is None on the live path today (teams come from live_roles), so only OCR runs.
        df = detections_df if self._reid is None else self._reid.extract_appearance_features(image_bgr, detections_df)
        if self._jersey is not None:
            df = self._jersey.process_frame(image_bgr, df)
        return AppearanceJobResult(frame_idx=frame_idx, detections_df=df, context=context)

    def _on_done(self, future) -> None:
        with self._lock:
            self._pending -= 1
        try:
            result = future.result()
        except Exception:
            log.exception("reid_worker_pool: appearance job failed")
            return
        self._results.put(result)

    def poll_results(self, block: bool = False, timeout: Optional[float] = None):
        """Drain-friendly: returns one completed AppearanceJobResult, or None
        if none are available (non-blocking by default). Caller (the main
        per-frame loop) should call this each iteration to consume results
        as they arrive, keeping all downstream state mutation (e.g.
        LiveTeamAssigner) on a single thread.
        """
        try:
            return self._results.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
