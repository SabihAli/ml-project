"""
run.py — CLI entry point for the gsr_pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path to allow running as a script
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import argparse
import logging
import yaml

from gsr_pipeline.pipeline import GSRPipeline
from gsr_pipeline.output import save_results_json

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="SoccerNet Game State Reconstruction Pipeline")
    parser.add_argument("--sequence_dir", type=str, required=True, help="Path to sequence directory")
    parser.add_argument("--output_dir", type=str, default="outputs/gsr_run", help="Output directory")
    parser.add_argument("--config", type=str, default="gsr_pipeline/configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames to process")
    parser.add_argument("--save_video", action="store_true", help="Generate annotated video")
    parser.add_argument("--no-refine", action="store_true", help="Disable tracklet refinement")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    if args.device:
        cfg["detector"]["device"] = args.device
        cfg["reid"]["device"] = args.device
        cfg["calibration"]["device"] = args.device
        # EasyOCR uses a boolean flag
        cfg["jersey"]["gpu"] = ("cuda" in args.device or "gpu" in args.device)
    
    if args.no_refine:
        cfg["refiner"]["enabled"] = False

    # Initialize and run
    pipeline = GSRPipeline(cfg)
    results = pipeline.process_sequence(args.sequence_dir, max_frames=args.max_frames)
    
    # Save output
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save JSON
    json_path = output_path / "predictions.json"
    save_results_json(results["detections"], results["metadata"], json_path)
    
    # 2. Generate Video
    if args.save_video or cfg.get("output", {}).get("save_video", False):
        from gsr_pipeline.visualize import generate_video
        video_path = output_path / "annotated.mp4"
        generate_video(results["frames"], results["detections"], video_path, metadata_df=results["metadata"])
    
    print(f"\nPipeline finished. Results saved to {output_path}")

if __name__ == "__main__":
    main()
