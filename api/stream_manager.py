"""
stream_manager.py — the single-active-stream state machine and the
background thread driving GSRPipeline.process_stream.

Implements the state machine in docs/architecture/STREAMING_ARCHITECTURE.md §2:

    IDLE --register--> VALIDATING --ok--> STREAMING --stop/disconnect--> STOPPING --> IDLE
                            |
                            +--fail--> IDLE

Deliberately enforces the v1 single-active-stream constraint
(docs/architecture/REQUIREMENTS.md) -- not a general multi-stream manager. A registration
attempt while a stream is active is rejected with 409, decided and confirmed
in docs/architecture/STREAMING_ARCHITECTURE.md §3.1.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from api.event_broadcaster import EventBroadcaster
from api.event_serializer import serialize_event
from gsr_pipeline.frame_source import FrameSource, LiveBroadcastFrameSource, VideoFileFrameSource
from gsr_pipeline.pipeline import GSRPipeline
from gsr_pipeline.pitch_view_gate import PitchViewGate
from gsr_pipeline.reid_worker_pool import ReidWorkerPool
from gsr_pipeline.team_live import LiveTeamAssigner

log = logging.getLogger(__name__)

# "file" -> VideoFileFrameSource (confirmed dev/staging-only, not a promised
# production input, docs/architecture/STREAMING_ARCHITECTURE.md §3.2). rtmp/rtsp/hls ->
# LiveBroadcastFrameSource (§1.2, build-order step 8) -- real reconnect-on-drop
# semantics, not the VideoFileFrameSource best-effort stand-in used before
# step 8 landed. LiveBroadcastFrameSource has only been exercised against
# synthetic failure injection (gsr_pipeline/validate_live_broadcast.py), not a
# real RTMP/RTSP/HLS feed -- none exists in this dev environment; see that
# script's module docstring.
SUPPORTED_SOURCE_TYPES = ("rtmp", "rtsp", "hls", "file")

_STATUS_BY_FAILURE_REASON = {
    "unreachable": 424,
    "undecodable": 422,
    "no_usable_video": 422,
}
_ERROR_BY_FAILURE_REASON = {
    "unreachable": "source_unreachable",
    "undecodable": "source_undecodable",
    "no_usable_video": "source_no_usable_video",
}


class StreamState(str, Enum):
    IDLE = "idle"
    VALIDATING = "validating"
    STREAMING = "streaming"
    STOPPING = "stopping"


@dataclass
class StreamManagerConfig:
    max_active_streams: int = 1
    process_stream_fps: float = 25.0
    log_progress_every_n_frames: int = 250

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "StreamManagerConfig":
        cfg = cfg or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class StreamManager:
    def __init__(self, pipeline: GSRPipeline, cfg: Dict[str, Any], loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._pipeline = pipeline
        self._cfg = cfg
        self._smcfg = StreamManagerConfig.from_dict(cfg.get("streaming"))
        # The asyncio event loop serving the FastAPI app -- captured at
        # startup (see api/main.py's lifespan) so background process_stream
        # threads can hand events off to the WS connection via
        # EventBroadcaster.publish (call_soon_threadsafe). None only in
        # tests/contexts that don't exercise the WS feed.
        self._loop = loop
        self._lock = threading.Lock()

        self._state = StreamState.IDLE
        self._stream_id: Optional[str] = None
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._last_summary: Dict[str, Any] = {}
        self._broadcaster: Optional[EventBroadcaster] = None

    @property
    def state(self) -> StreamState:
        with self._lock:
            return self._state

    # ------------------------------------------------------------------
    # Public API -- each returns (http_status, body_dict)
    # ------------------------------------------------------------------

    def register(self, source_type: str, source_url: str) -> Tuple[int, dict]:
        """Validates and, on success, starts streaming. Blocks the calling
        request thread for the duration of validation -- validation is a
        gate, not a background check (docs/architecture/STREAMING_ARCHITECTURE.md §1.4).
        """
        with self._lock:
            if self._state != StreamState.IDLE:
                return 409, {"error": "stream_already_active", "active_stream_id": self._stream_id}
            self._state = StreamState.VALIDATING

        try:
            frame_source = self._build_frame_source(source_type, source_url)
        except ValueError as exc:
            with self._lock:
                self._state = StreamState.IDLE
            return 422, {"error": "invalid_request", "detail": str(exc)}

        validation_cfg = self._cfg.get("frame_source", {}).get("validation", {})
        result = frame_source.validate(validation_cfg)

        if not result.ok:
            with self._lock:
                self._state = StreamState.IDLE
            status = _STATUS_BY_FAILURE_REASON.get(result.failure_reason, 422)
            error = _ERROR_BY_FAILURE_REASON.get(result.failure_reason, "source_invalid")
            return status, {"error": error, "detail": result.detail}

        stream_id = str(uuid.uuid4())
        stop_event = threading.Event()
        broadcaster = None
        if self._loop is not None:
            broadcaster = EventBroadcaster(self._loop, self._cfg.get("ws"))
        thread = threading.Thread(
            target=self._run_stream,
            args=(stream_id, frame_source, stop_event, broadcaster),
            name=f"process-stream-{stream_id[:8]}",
            daemon=True,
        )

        with self._lock:
            self._stream_id = stream_id
            self._stop_event = stop_event
            self._thread = thread
            self._state = StreamState.STREAMING
            self._broadcaster = broadcaster

        thread.start()
        log.info("stream %s registered (source_type=%s) and now streaming", stream_id, source_type)
        return 201, {"stream_id": stream_id, "state": StreamState.STREAMING.value}

    def stop(self, stream_id: str) -> Tuple[int, dict]:
        with self._lock:
            if self._stream_id != stream_id or self._state not in (StreamState.STREAMING, StreamState.VALIDATING):
                return 404, {"error": "stream_not_found", "detail": f"No active stream with id '{stream_id}'."}
            self._state = StreamState.STOPPING
            stop_event = self._stop_event
            thread = self._thread

        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=30.0)
            if thread.is_alive():
                log.warning("stream %s: process_stream did not exit within 30s of stop()", stream_id)

        with self._lock:
            self._state = StreamState.IDLE
            self._stream_id = None
            self._stop_event = None
            self._thread = None
            self._broadcaster = None

        log.info("stream %s stopped", stream_id)
        return 200, {"stream_id": stream_id, "state": StreamState.IDLE.value}

    def get_broadcaster(self, stream_id: str) -> Optional[EventBroadcaster]:
        """Used by the WS endpoint (api/main.py) to attach to the currently
        active stream's event feed. Returns None if `stream_id` doesn't
        match the active stream (caller should close the WS connection)."""
        with self._lock:
            if self._stream_id != stream_id:
                return None
            return self._broadcaster

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_frame_source(self, source_type: str, source_url: str) -> FrameSource:
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"Unsupported source_type '{source_type}'. Supported: {SUPPORTED_SOURCE_TYPES}")
        if not source_url:
            raise ValueError("source_url is required")
        if source_type == "file":
            return VideoFileFrameSource(source_url)
        return LiveBroadcastFrameSource(source_url, self._cfg.get("live_broadcast"))

    def _run_stream(
        self, stream_id: str, frame_source: FrameSource, stop_event: threading.Event,
        broadcaster: Optional[EventBroadcaster],
    ) -> None:
        gate = PitchViewGate(self._cfg.get("pitch_view_gate", {}))
        assigner = LiveTeamAssigner(self._cfg.get("team_live", {}))
        pool = ReidWorkerPool(self._pipeline.reid, self._pipeline.jersey, self._cfg.get("reid_worker_pool", {}))

        counters = {"fast": 0, "appearance": 0}

        def on_frame_result(event: dict) -> None:
            # Aggregated, not per-frame, logging -- REQUIREMENTS.md's
            # observability requirement explicitly calls out avoiding
            # per-frame log spam at streaming throughput.
            counters[event["type"]] += 1
            if counters["fast"] and counters["fast"] % self._smcfg.log_progress_every_n_frames == 0:
                log.info(
                    "stream %s: %d fast events, %d appearance events so far",
                    stream_id, counters["fast"], counters["appearance"],
                )
            if broadcaster is not None:
                # publish() is non-blocking and thread-safe (hands off to the
                # event loop via call_soon_threadsafe) -- safe to call here,
                # on this background thread, per event.
                try:
                    broadcaster.publish(serialize_event(event))
                except Exception:
                    log.exception("stream %s: failed to serialize event for WS feed", stream_id)

        try:
            summary = self._pipeline.process_stream(
                frame_source, on_frame_result,
                stop_event=stop_event, pitch_view_gate=gate,
                live_team_assigner=assigner, reid_pool=pool,
                fps=self._smcfg.process_stream_fps,
            )
            with self._lock:
                self._last_summary = summary
            log.info("stream %s: process_stream ended (%s)", stream_id, summary)
        except Exception:
            log.exception("stream %s: process_stream crashed", stream_id)
        finally:
            if broadcaster is not None:
                broadcaster.close()
            # Covers BOTH an explicit stop() (already transitioning through
            # STOPPING) and a live-source disconnect / natural exhaustion
            # that process_stream reported on its own (frame_source returned
            # None without stop_event being set) -- either way, the state
            # machine must not get stuck outside IDLE.
            with self._lock:
                if self._stream_id == stream_id and self._state in (StreamState.STREAMING, StreamState.STOPPING):
                    self._state = StreamState.IDLE
                    self._stream_id = None
                    self._stop_event = None
                    self._thread = None
                    self._broadcaster = None
