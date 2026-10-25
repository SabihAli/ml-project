"""
pipeline.py — Main GSRPipeline orchestrator.

Coordinates detection, ball fusion, tracking, calibration, roles/teams and jersey OCR into one
end-to-end process, batch (`process_sequence`) or live (`process_stream`).

The detector is the 2-class athlete/ball model (detect_athlete.py). It emits no roles, so roles
and teams come from kit colour (live_roles.py over athlete_roles.py). The ball is one row per
frame: the detector's top ball, or the c5c specialist's when the detector missed
(ball_specialist.py). Path A (the 4-class YOLOv8n) is in archive/path_a/.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from gsr_pipeline.athlete_roles import torso_descriptor
from gsr_pipeline.ball_specialist import BallFuser, BallSpecialist
from gsr_pipeline.calibrate import NBJWKeypointDetector, project_detections
from gsr_pipeline.calibration_live import LiveCalibrator
from gsr_pipeline.detect_athlete import ATHLETE_CLASS, BALL_CLASS, AthleteBallDetector
from gsr_pipeline.frame_source import FrameSource, ImageDirFrameSource
from gsr_pipeline.jersey import EasyOCRJerseyDetector
from gsr_pipeline.live_roles import LiveAthleteRoles, assign_sequence_roles
from gsr_pipeline.refine import refine_tracks
from gsr_pipeline.reid_worker_pool import AppearanceJobResult, ReidWorkerPool
from gsr_pipeline.track import ByteTracker
from gsr_pipeline.tracklet_agg import TrackletAggregator

log = logging.getLogger(__name__)

# One ball per frame under one constant id: a stand-in until the BallTracker exists
# (BALL_TEMPORAL_PLAN.md section 7). Same ids as notebooks/path_b_eval.py.
BALL_TRACK_ID = 90_000
SINGLETON_ID_START = 100_000    # batch: ids for athletes the tracker did not associate


class GSRPipeline:
    """End-to-end SoccerNet Game State Reconstruction pipeline."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """
        Args:
            cfg: Configuration dictionary (usually loaded from default.yaml).
        """
        self.cfg = cfg

        self.detector = AthleteBallDetector(**cfg.get("detector", {}))

        spec_cfg = cfg.get("ball_specialist", {})
        self.ball_specialist: Optional[BallSpecialist] = None
        if spec_cfg.get("enabled", True):
            self.ball_specialist = BallSpecialist(
                onnx_path=spec_cfg.get("onnx_path", "runs/ball_temporal/c5c/ball_specialist_c5c.onnx"),
                device=spec_cfg.get("device", "cuda:0"),
                backend=spec_cfg.get("backend"),
                auto_build_engine=spec_cfg.get("auto_build_engine", True),
            )
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
            auto_build_engine=calib_cfg.get("auto_build_engine", True),
        )
        self._calib_cfg = calib_cfg
        self.calibrator = self._new_calibrator()

        self.jersey = EasyOCRJerseyDetector(**cfg.get("jersey", {}))
        self.tracklet_agg = TrackletAggregator()
        self._reid = None             # OSNet, loaded only for the batch refiner (see `reid`)

        log.info("GSRPipeline initialized (detector backend %s, ball specialist %s).",
                 self.detector.backend,
                 self.ball_specialist.backend.value if self.ball_specialist else "off")

    @property
    def reid(self):
        """OSNet ReID, built on first use. Only the batch refiner needs embeddings."""
        if self._reid is None:
            from gsr_pipeline.reid import OSNetReID
            self._reid = OSNetReID(**self.cfg.get("reid", {}))
        return self._reid

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

    def _new_ball_fuser(self) -> BallFuser:
        """Fresh per-sequence/per-stream ball state (frame ring, last sighting)."""
        c = self.cfg.get("ball_specialist", {})
        return BallFuser(self.ball_specialist, threshold=c.get("threshold"),
                         max_miss_frames=c.get("max_miss_frames", 15))

    def _new_live_roles(self) -> LiveAthleteRoles:
        c = self.cfg.get("live_roles", {})
        return LiveAthleteRoles(
            self.cfg.get("athlete_roles", {}),
            fit_frames=c.get("fit_frames", 250),
            refit_every_frames=c.get("refit_every_frames", 50),
            max_buffer_frames=c.get("max_buffer_frames", 1500),
            min_athletes_for_pitch_view=c.get("min_athletes_for_pitch_view", 4),
        )

    def _detect_track_calibrate(self, frame_idx: int, img: np.ndarray, ball_fuser: BallFuser):
        """The latency-critical subset shared by process_sequence and process_stream:
        detect -> ball fusion (specialist on a miss) -> track athletes -> calibrate -> project.
        Roles, teams and jersey OCR come after, per mode.

        Returns (df, cam_params, keypoints, lines). df has one row per athlete plus at most one
        ball row (track_id BALL_TRACK_ID, `ball_source` main | specialist).
        """
        det = self.detector.detect(img)
        athletes = det[det["class_id"] == ATHLETE_CLASS].reset_index(drop=True)
        ball = ball_fuser.update(frame_idx, img, det[det["class_id"] == BALL_CLASS])

        df = self.tracker.update(athletes, img.shape[:2])
        if ball is not None:
            ball_row = pd.DataFrame([{**ball, "track_id": BALL_TRACK_ID}])
            df = ball_row if df.empty else pd.concat([df, ball_row], ignore_index=True)
        df["image_id"] = frame_idx

        boxes = None
        if len(det):
            boxes = np.asarray(det["bbox_ltrb"].tolist(), dtype=np.float64).reshape(-1, 4)
        calib = self.calibrator.update(frame_idx, img, boxes)
        keypoints, lines, cam_params = calib.keypoints, calib.lines, calib.cam_params
        df = project_detections(df, calib.H)
        df["h_source"] = calib.h_source
        df["h_age_frames"] = calib.h_age_frames

        # Flatten pitch coordinates for easier downstream use (roles, refinement, the wire)
        if "bbox_pitch" in df.columns and not df["bbox_pitch"].isna().all():
            pitch_coords = df["bbox_pitch"].apply(pd.Series)
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
                (wrapped in an ImageDirFrameSource), or any FrameSource instance.
            max_frames: Limit processing to the first N frames.

        Returns:
            Dictionary containing final DataFrames and metadata.

        Reads the whole source into memory first: roles (fit on the first
        live_roles.fit_frames frames, then a vote per tracklet) and the optional refiner need
        the complete sequence.
        """
        frame_source = source if isinstance(source, FrameSource) else ImageDirFrameSource(str(source))
        self.calibrator.close()
        self.calibrator = self._new_calibrator()
        ball_fuser = self._new_ball_fuser()
        refine_cfg = self.cfg.get("refiner", {})
        use_refiner = refine_cfg.get("enabled", False)

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
        next_singleton = SINGLETON_ID_START

        for frame_idx, img in tqdm(frames, desc="Frame processing"):
            df, cam_params, keypoints, lines = self._detect_track_calibrate(frame_idx, img, ball_fuser)
            is_athlete = (df["class_id"] == ATHLETE_CLASS) if len(df) else pd.Series(dtype=bool)

            # Untracked athletes get their own ids, as in notebooks/path_b_eval.py.
            for i in df.index[is_athlete & df["track_id"].isna()] if len(df) else []:
                df.at[i, "track_id"] = next_singleton
                next_singleton += 1

            df["torso_desc"] = [torso_descriptor(img, df.at[i, "bbox_ltrb"]) if is_athlete.at[i] else None
                                for i in df.index]
            # Jersey OCR reads player/goalkeeper rows; every athlete is a candidate until roles exist.
            df["role_detection"] = np.where(is_athlete, "player", "ball") if len(df) else []
            df = self.jersey.process_frame(img, df)
            if use_refiner:
                df = self.reid.extract_appearance_features(img, df)

            all_detections.append(df)
            metadata[frame_idx] = {
                "parameters": cam_params,
                "keypoints": keypoints,
                "lines": lines
            }

        full_df = pd.concat(all_detections, ignore_index=True)
        meta_df = pd.DataFrame.from_dict(metadata, orient="index")
        meta_df.index.name = "image_id"
        log.info("Ball: %s", ball_fuser.stats)

        # 2. Sequence-level processing
        role_info = assign_sequence_roles(full_df, self.cfg.get("athlete_roles", {}),
                                          self.cfg.get("live_roles", {}).get("fit_frames", 250))
        log.info("Roles: %s", role_info)
        full_df = self.tracklet_agg.process_sequence(full_df)
        full_df.loc[~full_df["role"].isin(["player", "goalkeeper"]), "jersey_number"] = None

        if use_refiner:
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
            "frames": frames,  # Keep for visualization if needed
            "ball_stats": dict(ball_fuser.stats),
            "role_info": role_info,
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
        reid_pool: Optional[ReidWorkerPool] = None,
        fps: float = 25.0,
    ) -> Dict[str, Any]:
        """Run the pipeline over a live/streaming frame source.

        Per frame, on this thread:
          1. detect -> ball fusion -> track -> calibrate (`_detect_track_calibrate`);
          2. roles and teams from `LiveAthleteRoles` (kit colour, fitted on the first
             live_roles.fit_frames frames, then a running vote per track);
          3. `on_frame_result` gets a "fast" result (positions, roles, teams, the ball) --
             never blocked on OCR;
          4. if `reid_pool` is given, the frame is submitted for background jersey OCR, and
             finished jobs are drained (non-blocking) and reported as "appearance" results
             carrying each track's current team.

        Args:
            frame_source: typically a LiveBroadcastFrameSource, or a VideoFileFrameSource as a
                stand-in.
            on_frame_result: called once per fast result and once per completed appearance
                result. Never called concurrently by this method.
            stop_event: set externally to request a clean stop; checked once per frame.
            reid_pool: if omitted, no jersey OCR runs (positions, roles and teams only).
            fps: source frame rate, for `frame_duration_s` in the appearance-job context.

        Returns:
            {"frame_count", "disconnect_reason", "ball_stats", "roles_fitted"}.
        """
        self.calibrator.close()
        self.calibrator = self._new_calibrator()
        ball_fuser = self._new_ball_fuser()
        roles = self._new_live_roles()
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
                df, cam_params, keypoints, lines = self._detect_track_calibrate(
                    record.frame_idx, record.image, ball_fuser)
                df, is_pitch_view = roles.update(record.frame_idx, record.image, df)

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
                    self._drain_appearance_results(reid_pool, roles, on_frame_result, block=False)

                frame_count += 1
        finally:
            frame_source.close()
            self.calibrator.close()
            if reid_pool is not None:
                reid_pool.shutdown(wait=True)
                self._drain_appearance_results(reid_pool, roles, on_frame_result, block=False)

        log.info("process_stream: %d frames, ball %s, roles fitted=%s",
                 frame_count, ball_fuser.stats, roles.fitted)
        return {"frame_count": frame_count, "disconnect_reason": disconnect_reason,
                "ball_stats": dict(ball_fuser.stats), "roles_fitted": roles.fitted}

    def _drain_appearance_results(
        self,
        reid_pool: ReidWorkerPool,
        roles: LiveAthleteRoles,
        on_frame_result: Callable[[Dict[str, Any]], None],
        block: bool,
    ) -> None:
        while True:
            result = reid_pool.poll_results(block=block)
            if result is None:
                return
            self._handle_appearance_result(result, roles, on_frame_result)

    @staticmethod
    def _handle_appearance_result(
        result: AppearanceJobResult,
        roles: LiveAthleteRoles,
        on_frame_result: Callable[[Dict[str, Any]], None],
    ) -> None:
        # Teams are a running vote, never locked for good; `locked` stays False on the wire.
        team_results: Dict[Any, dict] = {}
        for track in result.detections_df.get("track_id", pd.Series(dtype=float)).dropna().unique():
            if int(track) == BALL_TRACK_ID:
                continue
            team, confidence = roles.track_team(track)
            team_results[track] = {"team": team, "confidence": confidence, "locked": False}
        on_frame_result({
            "type": "appearance",
            "frame_idx": result.frame_idx,
            "detections": result.detections_df,
            "team_results": team_results,
        })
