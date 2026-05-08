"""
visualize.py — Visualization and video generation.

Provides utilities to draw detections, tracks, and metadata on frames
and save the result as a video file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from gsr_pipeline.utils import ltwh_to_ltrb

log = logging.getLogger(__name__)

# Colors (BGR)
COLORS = {
    "left": (255, 100, 100),    # Blue-ish
    "right": (100, 100, 255),   # Red-ish
    "goalkeeper": (0, 255, 255), # Yellow
    "ball": (0, 255, 0),        # Green
    "referee": (200, 200, 200), # Gray
    "unknown": (150, 150, 150)
}

def draw_detections(
    frame: np.ndarray,
    detections: pd.DataFrame,
) -> np.ndarray:
    """Draw bounding boxes and labels on a single frame."""
    canvas = frame.copy()

    for _, row in detections.iterrows():
        # Skip if bounding box is missing or invalid (can happen after interpolation)
        bbox = row.get("bbox_ltwh")
        if bbox is None or not isinstance(bbox, (list, tuple, np.ndarray)):
            continue
        # Check for NaN in the bbox
        if any(pd.isna(x) for x in bbox):
            continue
            
        l, t, w, h = bbox
        l, t, r, b = [int(x) for x in ltwh_to_ltrb(l, t, w, h)]
        
        team = row.get("team", "unknown")
        role = row.get("role", "unknown")
        track_id = row.get("track_id", None)
        jn = row.get("jersey_number", None)

        # Select color based on role first, then team
        if role == "ball":
            color = COLORS["ball"]
        elif role == "referee":
            color = COLORS["referee"]
        elif role == "goalkeeper":
            color = COLORS["goalkeeper"]
        elif team in COLORS:
            color = COLORS[team]
        else:
            color = COLORS["unknown"]

        # Draw box
        cv2.rectangle(canvas, (l, t), (r, b), color, 2)

        # Prepare label
        label_parts = []
        if track_id is not None and not pd.isna(track_id):
            label_parts.append(f"#{int(track_id)}")
        if jn:
            label_parts.append(f"J:{jn}")
        
        label = " ".join(label_parts)
        if label:
            cv2.putText(
                canvas, label, (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    return canvas

def draw_pitch_minimap(
    frame: np.ndarray,
    detections: pd.DataFrame,
    scale: float = 2.0,
    padding: int = 20,
) -> np.ndarray:
    """Draw a 2D minimap with projected player positions."""
    # Pitch dimensions in meters (SoccerNet standard)
    pitch_w, pitch_h = 105.0, 68.0
    
    # Minimap size
    mini_w = int(pitch_w * scale)
    mini_h = int(pitch_h * scale)
    
    # Create white background for minimap
    minimap = np.full((mini_h + 2*padding, mini_w + 2*padding, 3), 255, dtype=np.uint8)
    
    # Draw pitch boundary
    cv2.rectangle(minimap, (padding, padding), (mini_w + padding, mini_h + padding), (0, 0, 0), 2)
    
    # Draw halfway line
    cv2.line(minimap, (mini_w // 2 + padding, padding), (mini_w // 2 + padding, mini_h + padding), (0, 0, 0), 2)
    
    # Helper to convert pitch coords to minimap pixels
    def to_mini(x, y):
        # SoccerNet pitch coords: center is (0,0), x in [-52.5, 52.5], y in [-34, 34]
        mx = int((x + pitch_w/2) * scale) + padding
        my = int((y + pitch_h/2) * scale) + padding
        return mx, my

    # Draw players
    for _, row in detections.iterrows():
        bp = row.get("bbox_pitch")
        if not bp or pd.isna(bp): continue
        
        x, y = bp.get("x_bottom_middle"), bp.get("y_bottom_middle")
        if pd.isna(x) or pd.isna(y): continue
        
        mx, my = to_mini(x, y)
        
        team = row.get("team", "unknown")
        role = row.get("role", "unknown")
        color = COLORS.get(team, COLORS["unknown"])
        if role == "goalkeeper": color = COLORS["goalkeeper"]
        
        cv2.circle(minimap, (mx, my), 5, color, -1)
        cv2.circle(minimap, (mx, my), 5, (0, 0, 0), 1)

    # Overlay minimap onto frame (top-right corner)
    fh, fw = frame.shape[:2]
    mh, mw = minimap.shape[:2]
    frame[10:10+mh, fw-10-mw:fw-10] = minimap
    return frame

def draw_pitch_localization(
    frame: np.ndarray,
    keypoints: Dict[str, Any],
    lines: List[Any],
) -> np.ndarray:
    """Draw detected pitch keypoints and lines on the frame."""
    # Draw keypoints
    if keypoints:
        for name, kp in keypoints.items():
            if kp is not None:
                try:
                    # Handle dict format {'x': ..., 'y': ...} or sequence format [x, y]
                    if isinstance(kp, dict):
                        x, y = kp.get('x', kp.get(0)), kp.get('y', kp.get(1))
                    else:
                        x, y = kp[0], kp[1]
                    
                    if x is not None and y is not None:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                        cv2.putText(frame, str(name), (int(x), int(y)-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                except (IndexError, KeyError, TypeError):
                    continue
    
    # Draw lines
    if lines:
        # If lines is a dict (from NBJW), iterate over values
        line_items = lines.values() if isinstance(lines, dict) else lines
        for line in line_items:
            # Handle nbjw line object or raw list of points
            points = getattr(line, "coords", line)
            if points is not None and len(points) > 0:
                try:
                    # Convert list of dicts/tuples to numpy array of (x, y)
                    pts_list = []
                    for p in points:
                        if isinstance(p, dict):
                            pts_list.append([p.get('x', p.get(0)), p.get('y', p.get(1))])
                        else:
                            pts_list.append([p[0], p[1]])
                    
                    pts = np.array(pts_list, np.int32)
                    cv2.polylines(frame, [pts], False, (255, 0, 255), 2)
                except (IndexError, KeyError, TypeError):
                    continue
                
    return frame

def generate_video(
    frames: List[Tuple[int, np.ndarray]],
    detections_df: pd.DataFrame,
    output_path: Path,
    metadata_df: Optional[pd.DataFrame] = None,
    fps: int = 25,
) -> None:
    """Combine frames and detections into an annotated video file."""
    if not frames:
        return

    h, w = frames[0][1].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    log.info("Generating annotated video: %s", output_path)
    frames = sorted(frames, key=lambda x: x[0])

    for frame_idx, frame in tqdm(frames, desc="Annotating video"):
        frame_dets = detections_df[detections_df["image_id"] == frame_idx]
        
        # 1. Base annotations (boxes/IDs)
        annotated = draw_detections(frame, frame_dets)
        
        # 2. Pitch Localization (Metadata)
        if metadata_df is not None and frame_idx in metadata_df.index:
            meta = metadata_df.loc[frame_idx]
            annotated = draw_pitch_localization(
                annotated, 
                meta.get("keypoints"), 
                meta.get("lines")
            )
        
        # 3. Minimap
        annotated = draw_pitch_minimap(annotated, frame_dets)
        
        out.write(annotated)

    out.release()
    log.info("Video saved successfully.")
