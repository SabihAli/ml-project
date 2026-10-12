"""
main.py — the FastAPI app: /streams, /streams/{stream_id}/stop, /health,
and the WS event feed at /streams/{stream_id}/events.

Implements docs/architecture/STREAMING_ARCHITECTURE.md §3 (build-order step 6) and §4
(step 7).

Run (from the cv_service/ directory, with a venv that has fastapi/uvicorn
installed -- see requirements.txt):

    CV_SERVICE_API_KEY=some-secret .gsr_venv/Scripts/python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Loading the GSRPipeline (all models) happens once at startup and can take
a while -- the process is not ready to accept registrations until that
completes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import yaml
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from api.auth import API_KEY_ENV_VAR, require_api_key
from api.event_broadcaster import _CLOSED
from api.schemas import StreamRegisterRequest
from api.stream_manager import StreamManager
from gsr_pipeline.pipeline import GSRPipeline

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "gsr_pipeline/configs/default.yaml"

_state: Dict[str, Any] = {"pipeline": None, "manager": None}


def _load_cfg(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    cfg = _load_cfg()
    log.info("api: loading GSRPipeline (this loads all models -- may take a while)...")
    _state["pipeline"] = GSRPipeline(cfg)
    # Capture the running event loop so background process_stream threads
    # can hand WS events off to it (EventBroadcaster.publish, via
    # call_soon_threadsafe) -- see api/event_broadcaster.py.
    loop = asyncio.get_running_loop()
    _state["manager"] = StreamManager(_state["pipeline"], cfg, loop=loop)
    log.info("api: ready")
    yield
    _state["pipeline"] = None
    _state["manager"] = None


app = FastAPI(title="cv-service", version="0.1.0", lifespan=lifespan)


def _manager() -> StreamManager:
    manager = _state["manager"]
    if manager is None:
        raise RuntimeError("StreamManager not initialized -- app startup has not completed yet.")
    return manager


@app.post("/streams", status_code=201, dependencies=[Depends(require_api_key)])
def register_stream(request: StreamRegisterRequest) -> JSONResponse:
    status_code, body = _manager().register(request.source_type, request.source_url)
    return JSONResponse(status_code=status_code, content=body)


@app.post("/streams/{stream_id}/stop", dependencies=[Depends(require_api_key)])
def stop_stream(stream_id: str) -> JSONResponse:
    status_code, body = _manager().stop(stream_id)
    return JSONResponse(status_code=status_code, content=body)


@app.get("/health")
def health() -> dict:
    manager = _state["manager"]
    return {
        "status": "ok" if manager is not None else "starting",
        "stream_state": manager.state.value if manager is not None else "not_ready",
    }


@app.websocket("/streams/{stream_id}/events")
async def stream_events(websocket: WebSocket, stream_id: str) -> None:
    """Single-consumer, no-replay event feed for an active stream (§4).

    Auth: same static API key as the REST endpoints, passed as a query
    param (`?api_key=...`) since browser/most WS clients can't set custom
    headers on the handshake -- see api/event_broadcaster.py's docstring.

    Close codes used (no formal registry needed -- internal service, both
    ends are ours):
      4401 - missing/invalid API key
      4404 - stream_id doesn't match the currently active stream
      4409 - a consumer is already attached to this stream's feed (single-
             consumer policy)

    NOTE: the handshake must be `accept()`-ed before a custom close code can
    be delivered to the client -- closing pre-accept makes uvicorn reject
    the handshake at the HTTP level (403) instead, and the ASGI spec does
    not guarantee a code/reason survive that path. So every rejection path
    below accepts first, then immediately closes with the intended code.
    """
    expected_key = os.environ.get(API_KEY_ENV_VAR)
    provided_key = websocket.query_params.get("api_key")
    if not expected_key or provided_key != expected_key:
        await websocket.accept()
        await websocket.close(code=4401, reason="Invalid or missing API key.")
        return

    manager = _state["manager"]
    broadcaster = manager.get_broadcaster(stream_id) if manager is not None else None
    if broadcaster is None:
        await websocket.accept()
        await websocket.close(code=4404, reason=f"No active stream with id '{stream_id}'.")
        return

    if not broadcaster.try_attach():
        await websocket.accept()
        await websocket.close(code=4409, reason="A consumer is already attached to this stream's event feed.")
        return

    await websocket.accept()
    try:
        while True:
            event = await broadcaster.get()
            if event is _CLOSED:
                break
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("stream %s: WS event feed crashed", stream_id)
    finally:
        broadcaster.detach()
