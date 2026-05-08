"""
output.py — Result formatting and serialization.

Converts the internal DataFrame state into the SoccerNet Game State 
Reconstruction JSON format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def save_results_json(
    detections_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save pipeline results to a SoccerNet-compatible JSON file.

    Args:
        detections_df: DataFrame with 'image_id', 'track_id', 'bbox_ltwh',
                       'bbox_pitch', 'team', 'role', 'jersey_number'.
        metadata_df:   DataFrame indexed by image_id with 'parameters'.
        output_path:   Path to save the JSON.
    """
    output_data = {
        "predictions": []
    }

    # Group by image_id
    for image_id, group in detections_df.groupby("image_id"):
        # Image-level predictions
        img_meta = metadata_df.loc[image_id] if image_id in metadata_df.index else {}
        
        # Build predictions list
        # Standard SoccerNet GSR format is a list of frame objects
        # Each frame object contains "image_id" and "detections"
        
        frame_predictions = []
        for _, row in group.iterrows():
            det = {
                "track_id": int(row["track_id"]) if pd.notna(row["track_id"]) else None,
                "bbox_ltwh": row["bbox_ltwh"],
                "bbox_pitch": row["bbox_pitch"],
                "team": row["team"],
                "role": row["role"],
                "jersey_number": str(row["jersey_number"]) if pd.notna(row["jersey_number"]) else None,
            }
            # Clean up None values if they aren't expected by the challenge format
            # But usually None is acceptable for optional fields
            frame_predictions.append(det)

        output_data["predictions"].append({
            "image_id": int(image_id),
            "detections": frame_predictions,
            "camera_parameters": img_meta.get("parameters", {}),
            "keypoints": img_meta.get("keypoints", {}),
            "lines": img_meta.get("lines", {})
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)


def format_for_minimap(detections_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Helper to format detections for visual minimap plotting."""
    minimap_data = []
    for _, row in detections_df.iterrows():
        if row["bbox_pitch"] is not None:
            minimap_data.append({
                "x": row["bbox_pitch"]["x_bottom_middle"],
                "y": row["bbox_pitch"]["y_bottom_middle"],
                "team": row["team"],
                "role": row["role"],
                "track_id": row["track_id"]
            })
    return minimap_data
