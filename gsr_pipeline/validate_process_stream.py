"""
validate_process_stream.py — standalone, re-runnable end-to-end check for
GSRPipeline.process_stream, driven by a real VideoFileFrameSource (a stand-in
for a live broadcast source, per docs/architecture/STREAMING_ARCHITECTURE.md §8 build-order
step 4). Same style/rationale as the other validate_*.py scripts: not a test
framework, prints PASS/FAIL per check, exits non-zero on failure.

Runs the real model stack (athlete/ball detector, c5c ball specialist, calibration, live
roles, jersey OCR) through process_stream. Slow on CPU; keep --max_frames small.
--fit_frames lowers live_roles.fit_frames so a short run still reaches the role fit.

Usage (from the project's own venv, e.g. .gsr_venv, which has easyocr installed):
    python -m gsr_pipeline.validate_process_stream --mp4 data/outputs/SNGS-021_eval75/annotated.mp4 --max_frames 30
"""

from __future__ import annotations

import argparse
import sys
import threading

import yaml

from gsr_pipeline.frame_source import VideoFileFrameSource
from gsr_pipeline.pipeline import BALL_TRACK_ID, GSRPipeline
from gsr_pipeline.reid_worker_pool import ReidWorkerPool

results = []

ROLES = {"player", "goalkeeper", "referee", "other", "ball"}


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    results.append(condition)


def main():
    parser = argparse.ArgumentParser(description="Validate GSRPipeline.process_stream end-to-end")
    parser.add_argument("--mp4", required=True, help="A real video file to stream, e.g. an annotated.mp4 from a prior run")
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--fit_frames", type=int, default=10, help="live_roles.fit_frames for this run")
    parser.add_argument("--config", default="gsr_pipeline/configs/default.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["detector"]["device"] = args.device
    cfg["ball_specialist"]["device"] = args.device
    cfg["calibration"]["device"] = args.device
    cfg["jersey"]["gpu"] = "cuda" in args.device
    cfg.setdefault("live_roles", {})["fit_frames"] = args.fit_frames

    print(f"Initializing GSRPipeline (device={args.device})...")
    pipeline = GSRPipeline(cfg)

    frame_source = _MaxFramesVideoFileFrameSource(args.mp4, args.max_frames)
    pool = ReidWorkerPool(None, pipeline.jersey, cfg.get("reid_worker_pool", {}))

    events = []
    lock = threading.Lock()

    def on_frame_result(event):
        with lock:
            events.append(event)

    print(f"Running process_stream over {args.mp4} (max {args.max_frames} frames)...")
    summary = pipeline.process_stream(frame_source, on_frame_result, reid_pool=pool)

    fast_events = [e for e in events if e["type"] == "fast"]
    appearance_events = [e for e in events if e["type"] == "appearance"]
    rows = [r for e in fast_events for _, r in e["detections"].iterrows()]
    balls = [r for r in rows if r["role_detection"] == "ball"]

    check("process_stream completes without raising", True)
    check("frame_count matches the number of frames actually read", summary["frame_count"] == len(fast_events), f"frame_count={summary['frame_count']}, fast_events={len(fast_events)}")
    check("at least one fast (position/role) event was emitted", len(fast_events) > 0)
    check("every detection has a known role", all(r["role_detection"] in ROLES for r in rows),
          f"roles seen: {sorted({r['role_detection'] for r in rows})}")
    check("at most one ball per frame, under the ball track id",
          all((e["detections"]["role_detection"] == "ball").sum() <= 1 for e in fast_events)
          and all(int(r["track_id"]) == BALL_TRACK_ID for r in balls))
    check("every ball says where it came from", all(r["ball_source"] in ("main", "specialist") for r in balls),
          f"{len(balls)} balls, sources {dict((s, sum(r['ball_source'] == s for r in balls)) for s in ('main', 'specialist'))}")
    check("ball stats are reported", set(summary["ball_stats"]) == {"main", "specialist_calls", "specialist_hits"},
          str(summary["ball_stats"]))
    check("roles were fitted within the run", summary["roles_fitted"])
    check("fitted roles produce teams", any(r["team"] in ("left", "right") for r in rows))
    check("at least one appearance (jersey) event was eventually delivered", len(appearance_events) > 0, f"{len(appearance_events)} appearance events")
    check(
        "appearance events carry a team_results dict",
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
