"""
evaluate_team_live.py — live-mode team-assignment accuracy harness.

Implements docs/architecture/STREAMING_ARCHITECTURE.md §1.7.8: replays a labeled sequence
(e.g. a clip from the valid split) through the LIVE per-frame path
(GSRPipeline.process_stream + PitchViewGate + LiveTeamAssigner, driven by
ImageDirFrameSource -- itself a FrameSource, so it drives process_stream just
as well as a VideoFileFrameSource does) and compares each predicted team
assignment against ground truth (Labels-GameState.json), reusing
evaluate.py's existing GT loader rather than reimplementing it.

This is DESCRIPTIVE, not pass/fail -- there is no agreed accuracy target
(docs/architecture/REQUIREMENTS.md explicitly says not to invent one). It reports:
  - accuracy among LOCKED predictions only (the number that matters most --
    a locked prediction is asserted permanently, so its correctness is what
    actually reaches downstream consumers like possession.py's TURNOVER)
  - accuracy among ALL matched player/goalkeeper detections (locked and
    still-classifying together, treating an unconfirmed/null team as wrong)
  - lock coverage: what fraction of matched detections were locked by the
    time this run ended

Ground-truth matching is deliberately simple (nearest position + role match
within a distance threshold, per frame) -- NOT the full GS-HOTA alpha-sweep
Hungarian matching in evaluate.py, since this harness only needs "which GT
player is this," not a tracking-accuracy metric.

Usage:
    python -m gsr_pipeline.evaluate_team_live \
        --sequence_dir data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021 \
        --gt_path data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021/Labels-GameState.json \
        --max_frames 40 --bootstrap_window_s 0.2 --device cpu
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gsr_pipeline.pipeline import GSRPipeline
from gsr_pipeline.frame_source import ImageDirFrameSource
from gsr_pipeline.pitch_view_gate import PitchViewGate
from gsr_pipeline.team_live import LiveTeamAssigner
from gsr_pipeline.reid_worker_pool import ReidWorkerPool
from gsr_pipeline.evaluate import load_gt

MATCH_DIST_M = 3.0  # nearest-GT-detection match radius, meters on pitch


def _normalize_id(img_id) -> int:
    s = str(img_id)
    return int(s[-6:]) if len(s) > 6 else int(s)


def _row_xy(row):
    if "x_bottom_middle" in row and pd.notna(row.get("x_bottom_middle")):
        return row["x_bottom_middle"], row["y_bottom_middle"]
    bp = row.get("bbox_pitch")
    if isinstance(bp, dict):
        return bp.get("x_bottom_middle"), bp.get("y_bottom_middle")
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Live team-assignment accuracy harness")
    parser.add_argument("--sequence_dir", default="data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021")
    parser.add_argument("--gt_path", default="data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021/Labels-GameState.json")
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--bootstrap_window_s", type=float, default=None, help="Override team_live.bootstrap_window_s (lower = bootstrap completes sooner, useful to keep this harness's runtime bounded)")
    parser.add_argument("--config", default="gsr_pipeline/configs/default.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["detector"]["device"] = args.device
    cfg["reid"]["device"] = args.device
    cfg["calibration"]["device"] = args.device
    cfg["jersey"]["gpu"] = "cuda" in args.device
    if args.bootstrap_window_s is not None:
        cfg.setdefault("team_live", {})["bootstrap_window_s"] = args.bootstrap_window_s

    print(f"Loading ground truth from {args.gt_path}...")
    gt_df = load_gt(Path(args.gt_path))
    gt_df["norm_id"] = gt_df["image_id"].apply(_normalize_id)
    gt_df = gt_df[gt_df["role"].isin(["player", "goalkeeper"])]

    print(f"Initializing GSRPipeline (device={args.device})...")
    pipeline = GSRPipeline(cfg)

    frame_source = _MaxFramesImageDirFrameSource(args.sequence_dir, args.max_frames)
    gate = PitchViewGate(cfg.get("pitch_view_gate", {}))
    assigner = LiveTeamAssigner(cfg.get("team_live", {}))
    pool = ReidWorkerPool(pipeline.reid, pipeline.jersey, {"max_workers": 1, **cfg.get("reid_worker_pool", {})})

    matched_rows = []  # dicts: image_id, pred_team, locked, gt_team

    lock = threading.Lock()

    def on_frame_result(event):
        if event["type"] != "appearance":
            return
        with lock:
            image_id = _normalize_id(event["frame_idx"])
            gt_frame = gt_df[gt_df["norm_id"] == image_id]
            if gt_frame.empty:
                return
            gt_coords = gt_frame[["x", "y"]].dropna().values
            if len(gt_coords) == 0:
                return

            for _, row in event["detections"].iterrows():
                if row.get("role_detection") not in ("player", "goalkeeper"):
                    continue
                px, py = _row_xy(row)
                if px is None or py is None:
                    continue
                dists = np.sqrt(((gt_coords - np.array([px, py])) ** 2).sum(axis=1))
                nearest_idx = int(np.argmin(dists))
                if dists[nearest_idx] > MATCH_DIST_M:
                    continue
                gt_row = gt_frame.iloc[nearest_idx]

                team_result = event["team_results"].get(row["track_id"], {})
                matched_rows.append({
                    "image_id": event["frame_idx"],
                    "track_id": row["track_id"],
                    "pred_team": team_result.get("team"),
                    "locked": team_result.get("locked", False),
                    "gt_team": gt_row["team"],
                })

    print(f"Replaying {args.sequence_dir} through process_stream (max {args.max_frames} frames)...")
    summary = pipeline.process_stream(
        frame_source, on_frame_result,
        pitch_view_gate=gate, live_team_assigner=assigner, reid_pool=pool,
    )

    print(f"\nprocess_stream summary: frame_count={summary['frame_count']} disconnect_reason={summary['disconnect_reason']}")

    if not matched_rows:
        print("\nNo GT-matched player/goalkeeper detections were produced -- cannot report accuracy. "
              "Try a larger --max_frames or a lower --bootstrap_window_s.")
        return

    df = pd.DataFrame(matched_rows)
    n_total = len(df)
    n_locked = int(df["locked"].sum())

    def team_correct(row):
        return row["pred_team"] is not None and str(row["pred_team"]) == str(row["gt_team"])

    df["correct"] = df.apply(team_correct, axis=1)
    acc_all = df["correct"].mean()
    acc_locked = df.loc[df["locked"], "correct"].mean() if n_locked > 0 else float("nan")
    coverage_locked = n_locked / n_total if n_total > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"{'LIVE TEAM-ASSIGNMENT ACCURACY (descriptive, no target)':^60}")
    print("=" * 60)
    print(f"GT-matched player/goalkeeper detections: {n_total}")
    print(f"  Locked:      {n_locked} ({coverage_locked*100:.1f}% lock coverage)")
    print(f"  Unconfirmed: {n_total - n_locked}")
    print("-" * 60)
    print(f"Accuracy among LOCKED predictions:  {acc_locked*100:.1f}%" if n_locked else "Accuracy among LOCKED predictions:  n/a (none locked)")
    print(f"Accuracy among ALL matched (locked+unconfirmed): {acc_all*100:.1f}%")
    print("=" * 60)


class _MaxFramesImageDirFrameSource(ImageDirFrameSource):
    def __init__(self, sequence_dir: str, max_frames: int) -> None:
        super().__init__(sequence_dir)
        self._max_frames = max_frames
        self._read_count = 0

    def read_frame(self):
        if self._read_count >= self._max_frames:
            return None
        record = super().read_frame()
        if record is not None:
            self._read_count += 1
        return record


if __name__ == "__main__":
    main()
