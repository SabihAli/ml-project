"""
frame_source.py — abstraction over "where frames come from."

Lets the rest of the pipeline consume frames the same way regardless of
whether they originate from a pre-extracted SoccerNet image directory, a
local video file, or a live broadcast stream. See
docs/architecture/STREAMING_ARCHITECTURE.md §1 for the full design.

Three implementations:
- ImageDirFrameSource: wraps the existing SoccerNet img1/ directory behavior
  (utils.load_frames_from_dir), unchanged -- used for the offline benchmark
  path.
- VideoFileFrameSource: a local video file via OpenCV, used both for real
  local-file input and, during development, as a stand-in for "live" mode.
- LiveBroadcastFrameSource: RTMP/RTSP/HLS via OpenCV's FFmpeg backend, with
  transparent reconnect-on-drop (docs/architecture/STREAMING_ARCHITECTURE.md §8
  build-order step 8). Built and proven against `VideoFileFrameSource` as a
  stand-in first (steps 1-7), per the deliberate build order.

`read_frame() -> None` semantics differ by source type -- see `FrameSource`'s
docstring below. `ImageDirFrameSource`/`VideoFileFrameSource` have
unambiguous "sequence complete" as their only None case;
`LiveBroadcastFrameSource` is the one that actually surfaces the
"connection lost / gave up reconnecting" case via `disconnect_reason`.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .utils import load_frames_from_dir

log = logging.getLogger(__name__)


@dataclass
class FrameRecord:
    frame_idx: int          # 1-based, monotonically increasing within a source
    image: np.ndarray       # BGR, as read by OpenCV
    timestamp_ms: int       # source-relative -- see per-implementation notes


@dataclass
class ValidationResult:
    ok: bool
    failure_reason: Optional[str] = None   # 'unreachable' | 'undecodable' | 'no_usable_video' | None
    detail: str = ""


@dataclass
class FrameSourceValidationConfig:
    max_frames: int = 30
    timeout_s: float = 10.0
    black_intensity_threshold: float = 10.0
    frozen_diff_threshold: float = 1.0

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "FrameSourceValidationConfig":
        cfg = cfg or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class FrameSource(ABC):
    """Common interface every frame origin implements.

    `read_frame() -> None` semantics differ by source type and MUST NOT be
    conflated by callers:
      - For a finite source (image directory, video file): None means the
        sequence is complete. A normal, expected end state.
      - For a live source (not implemented in this file yet): None means the
        connection was lost or the source stopped sending frames
        unexpectedly -- NOT a normal end state. A future LiveBroadcastFrameSource
        must surface this distinction explicitly (e.g. a `.disconnect_reason`
        property) rather than leaving callers to guess from source type.
    """

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def read_frame(self) -> Optional[FrameRecord]: ...

    @abstractmethod
    def close(self) -> None: ...

    def validate(self, cfg: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Default validation: connect, read up to `max_frames` frames (or
        until `timeout_s` elapses), run the black/frozen check, close.

        Implements FR-002's checks (docs/architecture/REQUIREMENTS.md, docs/architecture/STREAMING_ARCHITECTURE.md
        §1.4). Subclasses may override if a source type needs a different check
        -- none currently do.
        """
        import time

        vcfg = FrameSourceValidationConfig.from_dict(cfg)

        try:
            self.connect()
        except Exception as exc:
            return ValidationResult(ok=False, failure_reason="unreachable", detail=str(exc))

        frames: List[FrameRecord] = []
        start = time.monotonic()
        try:
            while len(frames) < vcfg.max_frames and (time.monotonic() - start) < vcfg.timeout_s:
                record = self.read_frame()
                if record is None:
                    break
                frames.append(record)
        except Exception as exc:
            self.close()
            return ValidationResult(ok=False, failure_reason="undecodable", detail=str(exc))

        self.close()

        if not frames:
            return ValidationResult(
                ok=False, failure_reason="undecodable",
                detail="No frames could be decoded within the validation window.",
            )

        if _is_black_or_frozen(frames, vcfg):
            return ValidationResult(
                ok=False, failure_reason="no_usable_video",
                detail="Sampled frames are black and/or frozen (no usable video content).",
            )

        return ValidationResult(ok=True)


def _is_black_or_frozen(frames: List[FrameRecord], cfg: FrameSourceValidationConfig) -> bool:
    """True if every sampled frame is near-black, OR every consecutive pair
    of sampled frames is near-identical (frozen/static image passed off as a
    stream). Either condition alone across the WHOLE sample is required to
    reject -- a single dark or static frame among otherwise-normal ones is
    not itself a failure.
    """
    intensities = [float(np.mean(f.image)) for f in frames]
    all_black = all(i < cfg.black_intensity_threshold for i in intensities)
    if all_black:
        return True

    if len(frames) < 2:
        return False

    diffs = [
        float(np.mean(np.abs(frames[i].image.astype(np.int16) - frames[i - 1].image.astype(np.int16))))
        for i in range(1, len(frames))
    ]
    all_frozen = all(d < cfg.frozen_diff_threshold for d in diffs)
    return all_frozen


class ImageDirFrameSource(FrameSource):
    """Wraps the existing SoccerNet img1/-directory behavior unchanged.

    `timestamp_ms` is synthesized from `nominal_fps` (no real timestamps
    exist in the dataset) -- purely informational, nothing downstream in the
    batch path currently depends on it being accurate.
    """

    def __init__(self, sequence_dir: str, nominal_fps: float = 25.0) -> None:
        self._sequence_dir = Path(sequence_dir)
        self._nominal_fps = nominal_fps
        self._frames: Optional[List[tuple]] = None
        self._cursor = 0

    def connect(self) -> None:
        img1_dir = self._sequence_dir / "img1"
        if not img1_dir.exists():
            img1_dir = self._sequence_dir  # fallback to direct dir, matches prior pipeline.py behavior
        self._frames = load_frames_from_dir(img1_dir)
        self._cursor = 0

    def read_frame(self) -> Optional[FrameRecord]:
        if self._frames is None:
            raise RuntimeError("ImageDirFrameSource.read_frame() called before connect()")
        if self._cursor >= len(self._frames):
            return None
        frame_idx, image = self._frames[self._cursor]
        self._cursor += 1
        timestamp_ms = int(round((frame_idx - 1) * 1000.0 / self._nominal_fps))
        return FrameRecord(frame_idx=frame_idx, image=image, timestamp_ms=timestamp_ms)

    def close(self) -> None:
        self._frames = None
        self._cursor = 0


class VideoFileFrameSource(FrameSource):
    """A local video file via OpenCV's VideoCapture.

    Used both for genuine local-file input and, during development, as a
    stand-in for "live" mode without needing a real broadcast feed (see
    docs/architecture/STREAMING_ARCHITECTURE.md §8 build-order steps 2-3).
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx = 0

    def connect(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise ConnectionError(f"Could not open video file: {self._path}")
        self._cap = cap
        self._frame_idx = 0

    def read_frame(self) -> Optional[FrameRecord]:
        if self._cap is None:
            raise RuntimeError("VideoFileFrameSource.read_frame() called before connect()")
        ok, image = self._cap.read()
        if not ok:
            return None
        self._frame_idx += 1
        timestamp_ms = int(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        return FrameRecord(frame_idx=self._frame_idx, image=image, timestamp_ms=timestamp_ms)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


@dataclass
class LiveBroadcastConfig:
    # Reconnect policy: on a failed read, retry with exponential backoff
    # before surfacing None (and disconnect_reason) to the caller. A single
    # bad read is NOT treated as end-of-stream -- that would make
    # process_stream tear down the whole pipeline on a transient network
    # hiccup, which §1.3/§4.3 both explicitly decided against for the WS
    # layer and applies with equal force at the source itself.
    reconnect_max_attempts: int = 5
    reconnect_backoff_base_s: float = 1.0
    reconnect_backoff_max_s: float = 30.0
    # Bounds how long a single open()/read() call is allowed to block on the
    # network before OpenCV's FFmpeg backend gives up and returns failure --
    # without this, a stalled (not dropped) connection can hang read_frame()
    # indefinitely instead of surfacing as a failure the reconnect loop can
    # react to. Best-effort: older OpenCV/FFmpeg builds silently ignore these
    # properties rather than erroring, so this is a mitigation, not a guarantee.
    read_timeout_s: float = 10.0

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "LiveBroadcastConfig":
        cfg = cfg or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class LiveBroadcastFrameSource(FrameSource):
    """RTMP/RTSP/HLS via OpenCV's VideoCapture (FFmpeg backend).

    Reconnects transparently inside `read_frame()` on a failed read --
    process_stream's caller never sees an individual dropped read, only a
    final `None` once `reconnect_max_attempts` is exhausted. Per
    `FrameSource`'s class docstring, that final `None` means "connection
    lost, gave up" here, NOT "sequence complete" -- `disconnect_reason` is
    set to distinguish this from `VideoFileFrameSource`'s use of `None`.

    `timestamp_ms` is wall-clock-since-connect (`time.monotonic()` based),
    not `CAP_PROP_POS_MSEC` -- OpenCV's FFmpeg backend does not reliably
    report meaningful position timestamps for live RTMP/RTSP/HLS sources
    (unlike a seekable local file, where VideoFileFrameSource uses it
    directly). `frame_idx` keeps counting up across a reconnect -- it is
    NOT reset to 1, since it identifies this source's frame sequence, not
    any one underlying TCP/RTSP session.
    """

    def __init__(self, url: str, cfg: Optional[Dict[str, Any]] = None) -> None:
        self._url = url
        self.cfg = LiveBroadcastConfig.from_dict(cfg)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx = 0
        self._connect_time: Optional[float] = None
        # Only set once retries are exhausted -- None while connected or
        # while a reconnect attempt is still in progress. See FrameSource's
        # docstring: this is what lets process_stream/StreamManager tell a
        # genuine disconnect apart from a source that just finished.
        self.disconnect_reason: Optional[str] = None

    def connect(self) -> None:
        self._cap = self._open()
        self._connect_time = time.monotonic()
        self._frame_idx = 0
        self.disconnect_reason = None

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._url)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.cfg.read_timeout_s * 1000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.cfg.read_timeout_s * 1000)
        except Exception:
            pass  # older OpenCV/FFmpeg builds don't support these -- best-effort only
        if not cap.isOpened():
            cap.release()
            raise ConnectionError(f"Could not open live source: {self._url}")
        return cap

    def read_frame(self) -> Optional[FrameRecord]:
        if self._cap is None:
            raise RuntimeError("LiveBroadcastFrameSource.read_frame() called before connect()")

        ok, image = self._cap.read()
        if ok:
            return self._make_record(image)

        log.warning("LiveBroadcastFrameSource: read failed on %s -- attempting reconnect", self._url)
        self._cap.release()
        self._cap = None
        return self._reconnect_and_read()

    def _make_record(self, image: np.ndarray) -> FrameRecord:
        self._frame_idx += 1
        timestamp_ms = int((time.monotonic() - self._connect_time) * 1000)
        return FrameRecord(frame_idx=self._frame_idx, image=image, timestamp_ms=timestamp_ms)

    def _reconnect_and_read(self) -> Optional[FrameRecord]:
        backoff = self.cfg.reconnect_backoff_base_s
        for attempt in range(1, self.cfg.reconnect_max_attempts + 1):
            log.warning(
                "LiveBroadcastFrameSource: reconnect attempt %d/%d for %s (waiting %.1fs)",
                attempt, self.cfg.reconnect_max_attempts, self._url, backoff,
            )
            time.sleep(backoff)
            try:
                self._cap = self._open()
            except ConnectionError as exc:
                log.warning("LiveBroadcastFrameSource: reconnect attempt %d failed to open: %s", attempt, exc)
                backoff = min(backoff * 2, self.cfg.reconnect_backoff_max_s)
                continue

            ok, image = self._cap.read()
            if ok:
                log.info("LiveBroadcastFrameSource: reconnected to %s after %d attempt(s)", self._url, attempt)
                return self._make_record(image)

            self._cap.release()
            self._cap = None
            backoff = min(backoff * 2, self.cfg.reconnect_backoff_max_s)

        log.error(
            "LiveBroadcastFrameSource: giving up on %s after %d reconnect attempts",
            self._url, self.cfg.reconnect_max_attempts,
        )
        self.disconnect_reason = "reconnect_attempts_exhausted"
        return None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
