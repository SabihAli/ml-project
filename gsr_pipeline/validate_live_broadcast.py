"""
validate_live_broadcast.py — standalone, re-runnable checks for
LiveBroadcastFrameSource (frame_source.py), same PASS/FAIL style as the
other validate_*.py scripts.

No real RTMP/RTSP/HLS feed exists in this dev environment (and per this
session's standing constraint, thorough/network-dependent evaluation belongs
on the GPU machine, not here) -- so this validates the reconnect/backoff/
disconnect_reason LOGIC via synthetic failure injection: a fake
cv2.VideoCapture stand-in that fails opens/reads on a scripted schedule,
monkeypatched in place of the real cv2.VideoCapture for the duration of each
check. This proves the state machine is correct; it does NOT prove OpenCV's
FFmpeg backend behaves as assumed against a real live source -- that needs a
real feed, out of scope here.

Usage:
    python -m gsr_pipeline.validate_live_broadcast
"""

from __future__ import annotations

import sys
from typing import List, Optional

import numpy as np

import gsr_pipeline.frame_source as frame_source_module
from gsr_pipeline.frame_source import LiveBroadcastFrameSource

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    results.append(condition)


class FakeCapture:
    """Stand-in for cv2.VideoCapture. `script` is a list of outcomes
    consumed one-per-open(): each entry is either "fail_open" (isOpened()
    returns False) or a list of per-read outcomes (True -> a real frame,
    False -> a failed read) consumed one-per-.read() call; once a capture's
    read-outcome list is exhausted, subsequent reads keep returning False.
    """

    _script: List = []
    _script_cursor = 0
    _open_calls = 0

    def __init__(self, url):
        self.url = url
        cls = type(self)
        if cls._script_cursor >= len(cls._script):
            self._outcomes = []
            self._opened = False
            return
        entry = cls._script[cls._script_cursor]
        cls._script_cursor += 1
        cls._open_calls += 1
        if entry == "fail_open":
            self._opened = False
            self._outcomes = []
        else:
            self._opened = True
            self._outcomes = list(entry)
        self._read_cursor = 0

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        return True

    def read(self):
        if self._read_cursor < len(self._outcomes) and self._outcomes[self._read_cursor]:
            self._read_cursor += 1
            return True, np.zeros((4, 4, 3), dtype=np.uint8)
        self._read_cursor += 1
        return False, None

    def release(self):
        self._opened = False

    @classmethod
    def reset(cls, script):
        cls._script = script
        cls._script_cursor = 0
        cls._open_calls = 0


def _patch_cv2(monkey_script) -> None:
    FakeCapture.reset(monkey_script)
    frame_source_module.cv2.VideoCapture = FakeCapture


def main():
    real_video_capture = frame_source_module.cv2.VideoCapture

    try:
        print("== LiveBroadcastFrameSource: happy path ==")
        _patch_cv2([[True, True, True]])
        src = LiveBroadcastFrameSource("rtmp://fake/stream", {"reconnect_max_attempts": 3, "reconnect_backoff_base_s": 0.01})
        src.connect()
        recs = [src.read_frame() for _ in range(3)]
        check("all 3 frames read successfully with no reconnect needed", all(r is not None for r in recs))
        check("frame_idx increments 1,2,3", [r.frame_idx for r in recs] == [1, 2, 3])
        check("disconnect_reason stays None on a clean run", src.disconnect_reason is None)
        src.close()

        print("== LiveBroadcastFrameSource: transient drop, reconnect succeeds ==")
        # First open: 1 good frame then a failed read. Second open (the
        # reconnect): opens fine and immediately yields a good frame.
        _patch_cv2([[True, False], [True]])
        src = LiveBroadcastFrameSource("rtmp://fake/stream", {"reconnect_max_attempts": 3, "reconnect_backoff_base_s": 0.01})
        src.connect()
        first = src.read_frame()
        second = src.read_frame()  # triggers the failed read -> reconnect -> new frame
        check("frame read before the drop succeeds", first is not None and first.frame_idx == 1)
        check("frame read after reconnect succeeds and frame_idx keeps counting (not reset)", second is not None and second.frame_idx == 2)
        check("disconnect_reason stays None once reconnect succeeds", src.disconnect_reason is None)
        src.close()

        print("== LiveBroadcastFrameSource: reconnect attempts exhausted ==")
        # Initial open succeeds with one good frame then a failed read.
        # Every subsequent open attempt fails to open at all.
        _patch_cv2([[True, False], "fail_open", "fail_open", "fail_open"])
        src = LiveBroadcastFrameSource("rtmp://fake/stream", {"reconnect_max_attempts": 3, "reconnect_backoff_base_s": 0.01, "reconnect_backoff_max_s": 0.02})
        src.connect()
        first = src.read_frame()
        result = src.read_frame()  # drop -> 3 failed reconnect attempts -> None
        check("frame before the drop succeeds", first is not None)
        check("read_frame() returns None once reconnect_max_attempts is exhausted", result is None)
        check(
            "disconnect_reason is set to 'reconnect_attempts_exhausted' (NOT normal end-of-stream)",
            src.disconnect_reason == "reconnect_attempts_exhausted",
            f"got {src.disconnect_reason!r}",
        )
        src.close()

        print("== LiveBroadcastFrameSource: reconnect succeeds on a later attempt (not the first) ==")
        _patch_cv2([[True, False], "fail_open", [True, True]])
        src = LiveBroadcastFrameSource("rtmp://fake/stream", {"reconnect_max_attempts": 3, "reconnect_backoff_base_s": 0.01})
        src.connect()
        first = src.read_frame()
        second = src.read_frame()  # 1st reconnect attempt fails to open, 2nd succeeds
        check("recovers on the 2nd reconnect attempt after the 1st fails to open", second is not None)
        check("disconnect_reason stays None after eventually recovering", src.disconnect_reason is None)
        src.close()

        print("== LiveBroadcastFrameSource: connect() failure surfaces as ConnectionError ==")
        _patch_cv2(["fail_open"])
        src = LiveBroadcastFrameSource("rtmp://fake/stream")
        raised = False
        try:
            src.connect()
        except ConnectionError:
            raised = True
        check("connect() raises ConnectionError when the initial open fails", raised)

    finally:
        frame_source_module.cv2.VideoCapture = real_video_capture

    print()
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
