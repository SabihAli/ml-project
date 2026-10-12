"""
validate_frame_source.py — standalone, re-runnable checks for the FrameSource
abstraction (frame_source.py). Not a test framework (this repo has none, see
AGENTS.md's Testing section) -- a runnable script in the same style as
possession_demo.py, printing PASS/FAIL per check and exiting non-zero on any
failure, so it's something a human (or CI, later) can actually run and see.

Usage:
    python -m gsr_pipeline.validate_frame_source
    python -m gsr_pipeline.validate_frame_source --mp4 path/to/video.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gsr_pipeline.frame_source import (
    FrameRecord,
    ImageDirFrameSource,
    VideoFileFrameSource,
    _is_black_or_frozen,
    FrameSourceValidationConfig,
)

DEFAULT_SEQUENCE_DIR = "data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021"

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    results.append(condition)


def main():
    parser = argparse.ArgumentParser(description="Validate FrameSource implementations")
    parser.add_argument("--sequence_dir", default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--mp4", default=None, help="Optional real .mp4 to validate against (e.g. an annotated.mp4 from a prior run)")
    args = parser.parse_args()

    print("== ImageDirFrameSource ==")
    if Path(args.sequence_dir).exists():
        src = ImageDirFrameSource(args.sequence_dir)
        src.connect()
        count, first = 0, None
        while True:
            rec = src.read_frame()
            if rec is None:
                break
            first = first or rec
            count += 1
        src.close()
        check("reads at least one frame from the committed sample sequence", count > 0, f"{count} frames")
        check("first frame has a valid BGR image shape", first is not None and len(first.image.shape) == 3)

        src2 = ImageDirFrameSource(args.sequence_dir)
        result = src2.validate({"max_frames": 10, "timeout_s": 10.0})
        check("validate() accepts real football footage", result.ok, str(result))
    else:
        print(f"  (skipped -- {args.sequence_dir} not found)")

    print("== VideoFileFrameSource ==")
    if args.mp4 and Path(args.mp4).exists():
        vsrc = VideoFileFrameSource(args.mp4)
        result = vsrc.validate({"max_frames": 10, "timeout_s": 10.0})
        check("validate() accepts a real .mp4", result.ok, str(result))
    else:
        print("  (skipped -- pass --mp4 <path> to check against a real video file, "
              "e.g. an annotated.mp4 produced by gsr_pipeline.run)")

    bad = VideoFileFrameSource("does/not/exist.mp4")
    result = bad.validate({"max_frames": 5, "timeout_s": 3.0})
    check(
        "validate() rejects a missing file as 'unreachable'",
        not result.ok and result.failure_reason == "unreachable",
        str(result),
    )

    print("== Black/frozen detection (synthetic) ==")
    cfg = FrameSourceValidationConfig(black_intensity_threshold=10.0, frozen_diff_threshold=1.0)

    black_frames = [FrameRecord(i, np.zeros((10, 10, 3), dtype=np.uint8), i * 40) for i in range(5)]
    check("all-black frames flagged as no_usable_video", _is_black_or_frozen(black_frames, cfg))

    frozen_frames = [FrameRecord(i, np.full((10, 10, 3), 128, dtype=np.uint8), i * 40) for i in range(5)]
    check("static (non-black) frozen frames flagged as no_usable_video", _is_black_or_frozen(frozen_frames, cfg))

    rng = np.random.RandomState(0)
    normal_frames = [FrameRecord(i, rng.randint(0, 255, (10, 10, 3), dtype=np.uint8), i * 40) for i in range(5)]
    check("varying (normal) frames NOT flagged as no_usable_video", not _is_black_or_frozen(normal_frames, cfg))

    print()
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
