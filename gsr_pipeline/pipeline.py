"""
pipeline.py — Main GSRPipeline orchestrator.

Coordinates all modules (detection, reid, tracking, calibration, jersey, team)
into a single end-to-end inference process.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from gsr_pipeline.frame_source import FrameSource, ImageDirFrameSource
from gsr_pipeline.detect import YOLOv8nDetector
from gsr_pipeline.reid import OSNetReID
from gsr_pipeline.track import ByteTracker
from gsr_pipeline.calibrate import (
    NBJWKeypointDetector,
    project_detections
)
from gsr_pipeline.calibration_live import LiveCalibrator
from gsr_pipeline.jersey import EasyOCRJerseyDetector
from gsr_pipeline.team import TeamClassifier
from gsr_pipeline.tracklet_agg import TrackletAggregator
from gsr_pipeline.refine import refine_tracks
from gsr_pipeline.pitch_view_gate import PitchViewGate
from gsr_pipeline.team_live import LiveTeamAssigner
from gsr_pipeline.reid_worker_pool import ReidWorkerPool, AppearanceJobResult

log = logging.getLogger(__name__)


class GSRPipeline:
    """End-to-end SoccerNet Game State Reconstruction pipeline."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """
        Args:
            cfg: Configuration dictionary (usually loaded from default.yaml).
        """
        self.cfg = cfg
        
        # Initialize modules
        self.detector = YOLOv8nDetector(**cfg.get("detector", {}))
        self.reid = OSNetReID(**cfg.get("reid", {}))
        self.tracker = ByteTracker(**cfg.get("tracker", {}))
        
        calib_cfg = cfg.get("calibration", {})
        self.kp_detector = NBJWKeypointDetector(
            checkpoint_kp=calib_cfg.get("checkpoint_kp"),
            checkpoint_l=calib_cfg.get("checkpoint_l"),
            device=calib_cfg.get("device", "cpu"),
            backend=calib_cfg.get("backend", "torch"),
            onnx_kp=calib_cfg.get("onnx_kp"),
            onnx_l=calib_cfg.get("onnx_l"),
            fp16=calib_cfg.get("fp16", False),
            fast_decode=calib_cfg.get("fast_decode", True),
        )
        self._calib_cfg = calib_cfg
        self.calibrator = self._new_calibrator()
        
        self.jersey = EasyOCRJerseyDetector(**cfg.get("jersey", {}))
        self.team = TeamClassifier(**cfg.get("team", {}))
        self.tracklet_agg = TrackletAggregator()

        log.info("GSRPipeline initialized successfully.")

    def _new_calibrator(self) -> LiveCalibrator:
        """Fresh per-sequence/per-stream calibration state (anchor H, motion chain)."""
        c = self._calib_cfg
        return LiveCalibrator(
            self.kp_detector,
            image_width=c.get("image_width", 1920),
            image_height=c.get("image_height", 1080),
            use_prev_homography=c.get("use_prev_homography", True),
            compute_cam_params=c.get("compute_cam_params", False),
            calib_every_k=c.get("calib_every_k", 1),
            mode=c.get("mode", "inline"),
            max_h_age_frames=c.get("max_h_age_frames", 50),
            min_inlier_ratio=c.get("min_inlier_ratio", 0.5),
            max_zoom_change=c.get("max_zoom_change", 0.15),
            min_trigger_interval_frames=c.get("min_trigger_interval_frames", 3),
            motion=c.get("motion"),
        )

    def _detect_track_calibrate(self, frame_idx: int, img: np.ndarray):
        """The latency-critical subset shared by both process_sequence and
        process_stream: detect -> assign roles (cheap, no ReID forward pass)
        -> track -> calibrate. Deliberately excludes ReID embeddings and
        jersey OCR (both expensive) -- process_sequence calls those inline
        right after this (unchanged batch behavior); process_stream defers
        them to a background ReidWorkerPool (see reid_worker_pool.py and
        docs/architecture/STREAMING_ARCHITECTURE.md §1.7.6) so position/role events are
        never gated behind them.

        Returns (df, cam_params, keypoints, lines).
        """
        df = self.detector.process_sequence([img])[0]
        df["image_id"] = frame_idx

        roles, role_confs = self.reid.assign_roles(df, img.shape[0], img.shape[1])
        df["role_detection"] = roles
        df["role_confidence"] = role_confs

        df = self.tracker.update(df, img.shape)

        boxes = None
        if len(df):
            boxes = np.asarray(df["bbox_ltwh"].tolist(), dtype=np.float64).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]          # ltwh -> ltrb, for the camera-motion mask
        calib = self.calibrator.update(frame_idx, img, boxes)
        keypoints, lines, cam_params = calib.keypoints, calib.lines, calib.cam_params
        df = project_detections(df, calib.H)
        df["h_source"] = calib.h_source
        df["h_age_frames"] = calib.h_age_frames

        # Flatten pitch coordinates for easier downstream use (tracking, refinement)
        if "bbox_pitch" in df.columns and not df["bbox_pitch"].isna().all():
            # Correctly expand dict to columns while preserving index
            pitch_coords = df["bbox_pitch"].apply(pd.Series)
            # Drop existing columns if they already exist to avoid duplicates
            cols_to_drop = [c for c in pitch_coords.columns if c in df.columns]
            df = pd.concat([df.drop(columns=cols_to_drop), pitch_coords], axis=1)

        return df, cam_params, keypoints, lines

    def process_sequence(
        self,
        source: Union[str, "FrameSource"],
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run the pipeline over a complete sequence (batch mode).

        Args:
            source: Either a path to a directory containing an img1/ folder
                (str/Path -- wrapped in an ImageDirFrameSource automatically,
                preserving the exact behavior this method had before the
                FrameSource abstraction existed), or any FrameSource instance
                directly (e.g. VideoFileFrameSource) for callers that want to
                run batch-style processing over a non-image-directory source.
            max_frames:   Limit processing to the first N frames.

        Returns:
            Dictionary containing final DataFrames and metadata.

        Note: this reads the ENTIRE source into memory up front before the
        per-frame loop starts (same as the pre-FrameSource behavior) --
        Phase 2 (tracklet aggregation, team, refinement) below needs the
        complete sequence regardless of what FrameSource it came from. This
        method is not a streaming/live-mode entry point; see
        docs/architecture/STREAMING_ARCHITECTURE.md §1.6 for that (process_stream, not yet
        implemented).
        """
        frame_source = source if isinstance(source, FrameSource) else ImageDirFrameSource(str(source))
        self.calibrator.close()
        self.calibrator = self._new_calibrator()

        frame_source.connect()
        try:
            frames: List[tuple] = []
            while max_frames is None or len(frames) < max_frames:
                record = frame_source.read_frame()
                if record is None:
                    break
                frames.append((record.frame_idx, record.image))
        finally:
            frame_source.close()

        seq_name = getattr(frame_source, "_sequence_dir", None) or getattr(frame_source, "_path", None) or "sequence"
        log.info("Processing sequence: %s (%d frames)", seq_name, len(frames))

        # 1. Frame-level processing
        all_detections = []
        metadata = {}

        for frame_idx, img in tqdm(frames, desc="Frame processing"):
            # A-D. Detect, assign roles, track, calibrate (shared fast path)
            df, cam_params, keypoints, lines = self._detect_track_calibrate(frame_idx, img)

            # E. ReID embeddings + color histograms (expensive; batch mode
            # still runs this inline, synchronously, right here -- unchanged)
            df = self.reid.extract_appearance_features(img, df)

            # F. Jersey Number OCR
            df = self.jersey.process_frame(img, df)

            all_detections.append(df)
            metadata[frame_idx] = {
                "parameters": cam_params,
                "keypoints": keypoints,
                "lines": lines
            }

        # Combine all frame detections
        full_df = pd.concat(all_detections, ignore_index=True)
        meta_df = pd.DataFrame.from_dict(metadata, orient="index")
        meta_df.index.name = "image_id"

        # 2. Sequence-level processing (Post-processing)
        log.info("Applying sequence-level post-processing...")
        
        # F. Tracklet Aggregation (Voting)
        full_df = self.tracklet_agg.process_sequence(full_df)
        
        # G. Team Clustering & Side Labeling
        full_df = self.team.process_sequence(full_df)

        # H. Tracklet Refinement (Splitting/Merging)
        refine_cfg = self.cfg.get("refiner", {})
        if refine_cfg.get("enabled", True):
            log.info("Refining tracklets (splitting/merging)...")
            full_df = refine_tracks(
                full_df,
                speed_threshold=refine_cfg.get("speed_threshold", 10.0),
                reid_threshold=refine_cfg.get("reid_threshold", 0.3),
                interpolation_limit=refine_cfg.get("interpolation_limit", 15)
            )

        return {
            "detections": full_df,
            "metadata": meta_df,
            "frames": frames # Keep for visualization if needed
        }

    # ------------------------------------------------------------------
    # Live-mode counterpart to process_sequence.
    # See docs/architecture/STREAMING_ARCHITECTURE.md §1.6-1.7, §8 build-order step 4.
    # ------------------------------------------------------------------

    def process_stream(
        self,
        frame_source: FrameSource,
        on_frame_result: Callable[[Dict[str, Any]], None],
        stop_event: Optional[threading.Event] = None,
        pitch_view_gate: Optional[PitchViewGate] = None,
        live_team_assigner: Optional[LiveTeamAssigner] = None,
        reid_pool: Optional[ReidWorkerPool] = None,
        fps: float = 25.0,
    ) -> Dict[str, Any]:
        """Run the pipeline over a live/streaming frame source, Phase 1 only.

        Unlike process_sequence, this does NOT accumulate a full_df and does
        NOT run Phase 2 (tracklet aggregation, batch team clustering,
        refinement) -- those require the complete sequence, which a live
        stream never has (§1.6). Team assignment is instead handled
        incrementally by `live_team_assigner` (LiveTeamAssigner, §1.7), gated
        by `pitch_view_gate` (§1.7.2).

        Per frame:
          1. detect -> assign roles -> track -> calibrate (shared fast path,
             `_detect_track_calibrate`) runs inline, synchronously -- this is
             the latency-critical subset (PIPELINE_EVOLUTION.md §2).
          2. `on_frame_result` is called immediately with a "fast" result
             (position + role data for this frame) -- never blocked on ReID.
          3. If `reid_pool` is provided, the frame's crops are submitted for
             background embedding/color/jersey extraction; any already-
             completed jobs from previous frames are drained (non-blocking)
             and fed to `live_team_assigner`, then reported via
             `on_frame_result` as an "appearance" result. All appearance-job
             consumption happens on THIS thread (not a worker thread), so
             `live_team_assigner` (documented non-thread-safe) is never
             touched concurrently.

        Args:
            frame_source: typically a LiveBroadcastFrameSource in production,
                or a VideoFileFrameSource as a stand-in (no
                LiveBroadcastFrameSource exists yet -- §8 build-order step 9).
            on_frame_result: called once per fast result and once per
                completed appearance result. Never called concurrently by
                this method (single-threaded call site).
            stop_event: set externally to request a clean stop; checked once
                per frame.
            pitch_view_gate: if omitted, every frame is treated as a pitch
                view (gate effectively disabled) -- useful for tests that
                don't care about cutaway handling.
            live_team_assigner: if omitted, no team assignment happens and
                appearance results carry an empty team_results dict.
            reid_pool: if omitted, ReID/jersey are never computed in this
                loop at all (position/role only) -- useful for tests of the
                fast path in isolation.
            fps: source frame rate, used only to derive frame_duration_s for
                LiveTeamAssigner's bootstrap accumulation (§1.7.3) -- NOT
                derived from frame_source timestamps, per that method's own
                documented reasoning (a timestamp delta across a
                non-pitch-view gap would wrongly count as gated time).

        Returns:
            A summary dict: {"frame_count": int, "disconnect_reason": Any}.
            `disconnect_reason` is whatever `frame_source` exposes (if
            anything) when `read_frame()` returns None -- distinguishing a
            live source's unexpected drop from a finite source's normal
            completion is `frame_source`'s responsibility (§1.3), not this
            method's.
        """
        self.calibrator.close()
        self.calibrator = self._new_calibrator()
        frame_source.connect()
        frame_duration_s = 1.0 / fps
        frame_count = 0
        disconnect_reason = None

        try:
            while stop_event is None or not stop_event.is_set():
                record = frame_source.read_frame()
                if record is None:
                    disconnect_reason = getattr(frame_source, "disconnect_reason", None)
                    break

                timestamp_s = record.timestamp_ms / 1000.0
                df, cam_params, keypoints, lines = self._detect_track_calibrate(record.frame_idx, record.image)

                is_pitch_view = pitch_view_gate.is_pitch_view(df) if pitch_view_gate is not None else True

                on_frame_result({
                    "type": "fast",
                    "frame_idx": record.frame_idx,
                    "timestamp_s": timestamp_s,
                    "is_pitch_view": is_pitch_view,
                    "detections": df,
                })

                if reid_pool is not None:
                    reid_pool.submit(
                        record.frame_idx, record.image, df,
                        context={
                            "is_pitch_view": is_pitch_view,
                            "timestamp_s": timestamp_s,
                            "frame_duration_s": frame_duration_s,
                        },
                    )
                    self._drain_appearance_results(reid_pool, live_team_assigner, on_frame_result, block=False)

                frame_count += 1
        finally:
            frame_source.close()
            self.calibrator.close()
            if reid_pool is not None:
                reid_pool.shutdown(wait=True)
                self._drain_appearance_results(reid_pool, live_team_assigner, on_frame_result, block=False)

        return {"frame_count": frame_count, "disconnect_reason": disconnect_reason}

    def _drain_appearance_results(
        self,
        reid_pool: ReidWorkerPool,
        live_team_assigner: Optional[LiveTeamAssigner],
        on_frame_result: Callable[[Dict[str, Any]], None],
        block: bool,
    ) -> None:
        while True:
            result = reid_pool.poll_results(block=block)
            if result is None:
                return
            self._handle_appearance_result(result, live_team_assigner, on_frame_result)

    @staticmethod
    def _handle_appearance_result(
        result: AppearanceJobResult,
        live_team_assigner: Optional[LiveTeamAssigner],
        on_frame_result: Callable[[Dict[str, Any]], None],
    ) -> None:
        ctx = result.context
        team_results: Dict[Any, dict] = {}
        if live_team_assigner is not None:
            team_results = live_team_assigner.process_frame(
                result.detections_df,
                ctx.get("is_pitch_view", True),
                ctx.get("timestamp_s", 0.0),
                ctx.get("frame_duration_s", 1.0 / 25.0),
            )
        on_frame_result({
            "type": "appearance",
            "frame_idx": result.frame_idx,
            "detections": result.detections_df,
            "team_results": team_results,
        })
