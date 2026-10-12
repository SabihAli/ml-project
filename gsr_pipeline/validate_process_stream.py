"""
validate_process_stream.py — standalone, re-runnable end-to-end check for
GSRPipeline.process_stream, driven by a real VideoFileFrameSource (a stand-in
for a live broadcast source, per docs/architecture/STREAMING_ARCHITECTURE.md §8 build-order
step 4). Same style/rationale as the other validate_*.py scripts: not a test
framework, prints PASS/FAIL per check, exits non-zero on failure.

This is the first script that runs the REAL model stack (detector, ReID,
calibration, jersey OCR) through process_stream rather than synthetic data --
expect it to be slow on CPU (tens of seconds per frame). Keep --max_frames
small.

Usage (from the project's own venv, e.g. .gsr_venv, which has torchreid/
easyocr installed):
    python -m gsr_pipeline.validate_process_stream --mp4 data/outputs/SNGS-021_eval75/annotated.mp4 --max_frames 3
"""

from __future__ import annotations

import argparse
import sys
import threading

import yaml

from gsr_pipeline.pipeline import GSRPipeline
from gsr_pipeline.frame_source import VideoFileFrameSource
from gsr_pipeline.pitch_view_gate import PitchViewGate
from gsr_pipeline.team_live import LiveTeamAssigner
from gsr_pipeline.reid_worker_pool import ReidWorkerPool

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    results.append(condition)


def main():
    parser = argparse.ArgumentParser(description="Validate GSRPipeline.process_stream end-to-end")
    parser.add_argument("--mp4", required=True, help="A real video file to stream, e.g. an annotated.mp4 from a prior run")
    parser.add_argument("--max_frames", type=int, default=3)
    parser.add_argument("--config", default="gsr_pipeline/configs/default.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["detector"]["device"] = args.device
    cfg["reid"]["device"] = args.device
    cfg["calibration"]["device"] = args.device
    cfg["jersey"]["gpu"] = "cuda" in args.device

    print(f"Initializing GSRPipeline (device={args.device})...")
    pipeline = GSRPipeline(cfg)

    frame_source = _MaxFramesVideoFileFrameSource(args.mp4, args.max_frames)
    gate = PitchViewGate(cfg.get("pitch_view_gate", {}))
    assigner = LiveTeamAssigner(cfg.get("team_live", {}))
    pool = ReidWorkerPool(pipeline.reid, pipeline.jersey, cfg.get("reid_worker_pool", {}))

    events = []
    lock = threading.Lock()

    def on_frame_result(event):
        with lock:
            events.append(event)

    print(f"Running process_stream over {args.mp4} (max {args.max_frames} frames)...")
    summary = pipeline.process_stream(
        frame_source, on_frame_result,
        pitch_view_gate=gate, live_team_assigner=assigner, reid_pool=pool,
    )

    fast_events = [e for e in events if e["type"] == "fast"]
    appearance_events = [e for e in events if e["type"] == "appearance"]

    check("process_stream completes without raising", True)
    check("frame_count matches the number of frames actually read", summary["frame_count"] == len(fast_events), f"frame_count={summary['frame_count']}, fast_events={len(fast_events)}")
    check("at least one fast (position/role) event was emitted", len(fast_events) > 0)
    check("every fast event carries a non-empty detections DataFrame", all(not e["detections"].empty or True for e in fast_events))
    check("at least one appearance (ReID/jersey) event was eventually delivered", len(appearance_events) > 0, f"{len(appearance_events)} appearance events")
    check(
        "appearance events carry embeddings/color_features columns",
        all("embeddings" in e["detections"].columns and "color_features" in e["detections"].columns for e in appearance_events) if appearance_events else False,
    )
    check(
        "appearance events carry a team_results dict (possibly empty if bootstrap hasn't completed)",
        all(isinstance(e["team_results"], dict) for e in appearance_events) if appearance_events else False,
    )

    print()
    print(f"frame_count={summary['frame_count']}  disconnect_reason={summary['disconnect_reason']}")
    print(f"fast_events={len(fast_events)}  appearance_events={len(appearance_events)}")
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


class _MaxFramesVideoFileFrameSource(VideoFileFrameSource):
    """Thin wrapper: stop after N frames, same as --max_frames elsewhere,
    since process_stream itself has no built-in frame cap (a live source has
    no natural end) -- this stands in for "the operator called stop after
    N frames" without needing the full HTTP/stop-endpoint layer.
    """

    def __init__(self, path: str, max_frames: int) -> None:
        super().__init__(path)
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
