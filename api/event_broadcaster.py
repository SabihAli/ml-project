"""
event_broadcaster.py — thread-safe hand-off from process_stream's background
worker thread to the asyncio event loop serving the WebSocket connection.

Implements docs/architecture/STREAMING_ARCHITECTURE.md §4 (build-order step 7). Design
decisions (asked and confirmed with the user before building this):

  - Single consumer per stream: a second WS connect attempt while one is
    already attached is rejected (closed) rather than fanned out to both.
  - No replay buffer: a client that connects after the stream has already
    started only sees events from its connect-time onward -- consistent
    with the "no accumulation, per-frame" philosophy applied everywhere
    else in this service (team_live.py, possession.py's grace window
    aside, which is a detection-domain rule, not a buffering strategy).
  - Backpressure: bounded queue, drop-oldest -- if the consumer can't keep
    up, freshness wins over completeness (same policy as
    reid_worker_pool.py's job queue).
  - Same static API-key auth as the REST endpoints, passed as a query
    param on the WS handshake (WS clients can't set custom headers from a
    browser; a query param is the pragmatic standard for WS auth, and
    this is an internal service-to-service call, not a public endpoint --
    see auth.py's docstring for the same reasoning applied to /streams).

`GSRPipeline.process_stream` calls `on_frame_result` synchronously from the
stream's dedicated background thread (see stream_manager.py's
`_run_stream`). `publish()` is the only method called from that thread; it
must never block and must never raise into the caller. Everything else
(subscribe/close) runs on the asyncio event loop thread serving requests.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

log = logging.getLogger(__name__)

# Sentinel placed on the queue to signal "no more events, stream ended" --
# lets the WS send-loop exit cleanly instead of hanging on queue.get().
_CLOSED = object()


@dataclass
class EventBroadcasterConfig:
    max_queue_size: int = 100

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "EventBroadcasterConfig":
        cfg = cfg or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class EventBroadcaster:
    """One instance per active stream. Bridges a background worker thread
    (producer, calls `publish`) to a single WebSocket connection (consumer,
    calls `subscribe`/`attach`/`detach`) living on the asyncio event loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, cfg: Optional[Dict[str, Any]] = None) -> None:
        self._loop = loop
        self._cfg = EventBroadcasterConfig.from_dict(cfg)
        self._queue: Deque[Any] = deque(maxlen=self._cfg.max_queue_size)
        self._not_empty = asyncio.Event()
        self._closed = False
        self._has_subscriber = False
        self._dropped_count = 0

    # ------------------------------------------------------------------ #
    # Producer side -- called from the background process_stream thread.
    # ------------------------------------------------------------------ #

    def publish(self, event: dict) -> None:
        """Never blocks, never raises. Safe to call from any thread."""
        try:
            self._loop.call_soon_threadsafe(self._enqueue, event)
        except RuntimeError:
            # Event loop already closed (e.g. server shutting down mid-stream) --
            # nothing useful to do with this event.
            pass

    def close(self) -> None:
        """Signal end-of-stream to whoever is subscribed. Safe to call from
        any thread; safe to call more than once."""
        try:
            self._loop.call_soon_threadsafe(self._enqueue, _CLOSED)
        except RuntimeError:
            pass

    def _enqueue(self, item: Any) -> None:
        # Runs on the event loop thread (via call_soon_threadsafe).
        if self._closed:
            return
        if item is _CLOSED:
            self._closed = True
        elif len(self._queue) == self._queue.maxlen:
            self._dropped_count += 1
            if self._dropped_count % 100 == 1:
                log.warning(
                    "event_broadcaster: consumer falling behind, dropped %d events so far "
                    "(queue capacity %d)", self._dropped_count, self._queue.maxlen,
                )
        self._queue.append(item)
        self._not_empty.set()

    # ------------------------------------------------------------------ #
    # Consumer side -- called from the WS handler coroutine, event loop thread.
    # ------------------------------------------------------------------ #

    def try_attach(self) -> bool:
        """Returns False if another consumer is already attached (single-
        consumer policy) -- caller should reject the connection."""
        if self._has_subscriber:
            return False
        self._has_subscriber = True
        return True

    def detach(self) -> None:
        self._has_subscriber = False

    async def get(self) -> Any:
        """Waits for and returns the next event, or the `_CLOSED` sentinel
        once the stream has ended and no events remain."""
        while not self._queue and not self._closed:
            self._not_empty.clear()
            await self._not_empty.wait()
        if self._queue:
            return self._queue.popleft()
        return _CLOSED

    @property
    def is_closed_and_drained(self) -> bool:
        return self._closed and not self._queue
