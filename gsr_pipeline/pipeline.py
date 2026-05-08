"""
pipeline.py — Main GSRPipeline orchestrator.

Coordinates all modules (detection, reid, tracking, calibration, jersey, team)
into a single end-to-end inference process.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from gsr_pipeline.utils import load_frames_from_dir
from gsr_pipeline.detect import YOLOv8nDetector
from gsr_pipeline.reid import OSNetReID
from gsr_pipeline.track import ByteTracker
from gsr_pipeline.calibrate import (
    NBJWKeypointDetector, 
    NBJWHomography, 
    project_detections
)
from gsr_pipeline.jersey import EasyOCRJerseyDetector
from gsr_pipeline.team import TeamClassifier
from gsr_pipeline.tracklet_agg import TrackletAggregator
from gsr_pipeline.refine import refine_tracks

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
            device=calib_cfg.get("device", "cpu")
        )
        self.homography = NBJWHomography(
            image_width=calib_cfg.get("image_width", 1920),
            image_height=calib_cfg.get("image_height", 1080),
            use_prev_homography=calib_cfg.get("use_prev_homography", True)
        )
        
        self.jersey = EasyOCRJerseyDetector(**cfg.get("jersey", {}))
        self.team = TeamClassifier(**cfg.get("team", {}))
        self.tracklet_agg = TrackletAggregator()

        log.info("GSRPipeline initialized successfully.")

    def process_sequence(
        self, 
        sequence_dir: str, 
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run the pipeline on a SoccerNet sequence directory.

        Args:
            sequence_dir: Path to directory containing img1/ folder.
            max_frames:   Limit processing to the first N frames.

        Returns:
            Dictionary containing final DataFrames and metadata.
        """
        seq_path = Path(sequence_dir)
        img1_dir = seq_path / "img1"
        if not img1_dir.exists():
            img1_dir = seq_path  # Fallback to direct dir

        frames = load_frames_from_dir(img1_dir)
        if max_frames:
            frames = frames[:max_frames]

        log.info("Processing sequence: %s (%d frames)", seq_path.name, len(frames))

        # 1. Frame-level processing
        all_detections = []
        metadata = {}

        for frame_idx, img in tqdm(frames, desc="Frame processing"):
            # A. Detection
            df = self.detector.process_sequence([img])[0]
            df["image_id"] = frame_idx
            
            # B. ReID & Role Heuristic
            df = self.reid.process_frame(img, df)
            
            # C. Tracking
            df = self.tracker.update(df, img.shape)
            
            # D. Calibration (NBJW)
            keypoints, lines = self.kp_detector.detect(img)
            H, cam_params = self.homography.compute(keypoints)
            df = project_detections(df, H)
            
            # Flatten pitch coordinates for easier downstream use (tracking, refinement)
            if "bbox_pitch" in df.columns and not df["bbox_pitch"].isna().all():
                # Correctly expand dict to columns while preserving index
                pitch_coords = df["bbox_pitch"].apply(pd.Series)
                # Drop existing columns if they already exist to avoid duplicates
                cols_to_drop = [c for c in pitch_coords.columns if c in df.columns]
                df = pd.concat([df.drop(columns=cols_to_drop), pitch_coords], axis=1)
            
            # E. Jersey Number OCR
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
