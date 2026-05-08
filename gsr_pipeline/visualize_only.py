"""
visualize_only.py — Temporary script to run visualization on existing JSON results.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from gsr_pipeline.utils import load_frames_from_dir
from gsr_pipeline.visualize import generate_video

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run visualization on existing JSON predictions")
    parser.add_argument("--json_path", type=str, required=True, help="Path to predictions.json")
    parser.add_argument("--sequence_dir", type=str, required=True, help="Path to original sequence directory (img1/)")
    parser.add_argument("--output_path", type=str, default="annotated_offline.mp4", help="Output video path")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames")

    args = parser.parse_args()

    # 1. Load JSON
    print(f"Loading predictions from {args.json_path}...")
    with open(args.json_path, "r") as f:
        data = json.load(f)

    # 2. Convert JSON to DataFrame
    # Note: SoccerNet format is list of {image_id, detections: [...] }
    rows = []
    for frame_data in data.get("predictions", []):
        img_id = frame_data["image_id"]
        for det in frame_data["detections"]:
            det["image_id"] = img_id
            rows.append(det)
    
    detections_df = pd.DataFrame(rows)
    
    # 2.5 Extract Metadata (Pitch localization, etc.)
    metadata = {}
    for frame_data in data.get("predictions", []):
        img_id = frame_data["image_id"]
        metadata[img_id] = {
            "parameters": frame_data.get("camera_parameters", {}),
            "keypoints": frame_data.get("keypoints", {}),
            "lines": frame_data.get("lines", {})
        }
    meta_df = pd.DataFrame.from_dict(metadata, orient="index")
    meta_df.index.name = "image_id"

    # 3. Load Frames
    print(f"Loading frames from {args.sequence_dir}...")
    img1_dir = Path(args.sequence_dir)
    if (img1_dir / "img1").exists():
        img1_dir = img1_dir / "img1"
        
    frames = load_frames_from_dir(img1_dir)
    if args.max_frames:
        frames = frames[:args.max_frames]

    # 4. Generate Video
    print(f"Generating video with metadata (pitch localization enabled)...")
    generate_video(frames, detections_df, Path(args.output_path), metadata_df=meta_df)
    print(f"Done. Video saved to {args.output_path}")

if __name__ == "__main__":
    main()
