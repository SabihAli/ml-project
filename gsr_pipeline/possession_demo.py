"""
possession_demo.py — Run possession/turnover assignment against an existing
predictions.json (output of `gsr_pipeline.run`) and write the resulting
POSSESSION/TURNOVER records to a JSON file.

This is an offline validation tool, not a streaming/live component -- see
the module docstring in `possession.py` for why it currently operates on the
final post-pipeline DataFrame rather than the detect+track stage directly.

Usage:
    python -m gsr_pipeline.possession_demo \
        --json_path data/outputs/SNGS-021_smoke/predictions.json \
        --output_path data/outputs/SNGS-021_smoke/possession_events.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from gsr_pipeline.possession import compute_possession_and_turnovers


def load_detections_df(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for frame_data in data.get("predictions", []):
        img_id = frame_data["image_id"]
        for det in frame_data["detections"]:
            det = dict(det)
            det["image_id"] = img_id
            rows.append(det)
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run possession/turnover assignment on existing predictions.json")
    parser.add_argument("--json_path", type=str, required=True, help="Path to predictions.json")
    parser.add_argument("--output_path", type=str, default="possession_events.json", help="Where to write events")
    parser.add_argument("--config", type=str, default="gsr_pipeline/configs/default.yaml", help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    possession_cfg = cfg.get("possession", {})

    print(f"Loading predictions from {args.json_path}...")
    detections_df = load_detections_df(Path(args.json_path))

    possession_records, turnover_records = compute_possession_and_turnovers(detections_df, possession_cfg)

    output = {
        "possession": possession_records,
        "turnover": turnover_records,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(
        f"\n{len(possession_records)} POSSESSION records, "
        f"{len(turnover_records)} TURNOVER records written to {output_path}"
    )


if __name__ == "__main__":
    main()
