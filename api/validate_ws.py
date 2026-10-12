"""
validate_ws.py — standalone, re-runnable end-to-end check for the WS event
feed (docs/architecture/STREAMING_ARCHITECTURE.md §4, build-order step 7). Same
PASS/FAIL style as validate_api.py.

Requires a real running server (see api/main.py's module docstring).
Registers a real stream via the REST API, attaches a WS client, and checks:
  1. wrong/missing api_key on the WS handshake -> closed with code 4401
  2. events actually arrive and are well-formed JSON with the expected shape
  3. a second concurrent WS client on the same stream is rejected (4409)
  4. connecting to a stream_id that isn't active -> closed with code 4404
  5. the feed closes cleanly (with no more sends) once the stream is stopped

Usage:
    python -m api.validate_ws --base_url http://127.0.0.1:8000 --api_key some-secret --mp4 data/outputs/SNGS-021_eval75/annotated.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import requests
import websockets

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    results.append(condition)


async def main():
    parser = argparse.ArgumentParser(description="Validate the cv-service WS event feed end-to-end")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000")
    parser.add_argument("--ws_base_url", default=None, help="Defaults to base_url with ws:// scheme")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--mp4", required=True)
    parser.add_argument("--wait_ready_s", type=float, default=180.0)
    args = parser.parse_args()

    ws_base_url = args.ws_base_url or args.base_url.replace("http://", "ws://").replace("https://", "wss://")
    headers = {"X-CV-Service-Key": args.api_key}

    print(f"Waiting for {args.base_url}/health to be ready (up to {args.wait_ready_s:.0f}s)...")
    deadline = time.monotonic() + args.wait_ready_s
    ready = False
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{args.base_url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                ready = True
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    check("server becomes ready within the wait window", ready)
    if not ready:
        sys.exit(1)

    # Rejections happen AFTER the WS handshake accepts (uvicorn maps a
    # pre-accept close to a bare HTTP 403 with no code/reason delivered --
    # see api/main.py's docstring on stream_events) -- so the handshake
    # itself succeeds and the custom close code only surfaces once we try
    # to actually receive from the connection.
    async def _expect_rejection(url: str, expected_code: int) -> tuple:
        try:
            async with websockets.connect(url) as ws:
                await asyncio.wait_for(ws.recv(), timeout=10.0)
                return False, "a message was received instead of a close"
        except websockets.exceptions.ConnectionClosed as e:
            return e.code == expected_code, f"got close code {e.code}"
        except asyncio.TimeoutError:
            return False, "connection stayed open, no close received"

    # 0. Connecting to a stream_id that isn't active at all -> 4404.
    ok, detail = await _expect_rejection(f"{ws_base_url}/streams/not-a-real-id/events?api_key={args.api_key}", 4404)
    check("connecting to a nonexistent stream is rejected with 4404", ok, detail)

    # 1. Register a real stream.
    r = requests.post(f"{args.base_url}/streams", json={"source_type": "file", "source_url": args.mp4}, headers=headers)
    body = r.json()
    check("stream registers successfully", r.status_code == 201, f"status={r.status_code} body={body}")
    stream_id = body.get("stream_id")
    if not stream_id:
        print("No stream_id -- aborting remaining checks.")
        sys.exit(1)

    events_url = f"{ws_base_url}/streams/{stream_id}/events"

    # 2. Wrong api_key on the WS handshake -> 4401.
    ok, detail = await _expect_rejection(f"{events_url}?api_key=wrong", 4401)
    check("wrong api_key on WS handshake is rejected with 4401", ok, detail)

    # 3. Attach the real (first) consumer and read a few events.
    ws1 = await websockets.connect(f"{events_url}?api_key={args.api_key}")
    received = []
    try:
        for _ in range(5):
            raw = await asyncio.wait_for(ws1.recv(), timeout=30.0)
            received.append(raw)
    except asyncio.TimeoutError:
        pass
    check("at least one event received on the WS feed", len(received) > 0, f"received {len(received)} events")

    if received:
        import json
        parsed = json.loads(received[0])
        check(
            "event has the expected shape (type + frame_idx + detections)",
            parsed.get("type") in ("fast", "appearance") and "frame_idx" in parsed and "detections" in parsed,
            f"got keys {list(parsed.keys())}",
        )

    # 4. A second concurrent consumer on the SAME stream -> 4409.
    ok, detail = await _expect_rejection(f"{events_url}?api_key={args.api_key}", 4409)
    check("a second concurrent WS consumer is rejected with 4409", ok, detail)

    # 5. Stop the stream; the WS connection should close cleanly on its own.
    r = requests.post(f"{args.base_url}/streams/{stream_id}/stop", headers=headers)
    check("stream stops successfully", r.status_code == 200, f"status={r.status_code}")

    closed_cleanly = False
    try:
        await asyncio.wait_for(ws1.wait_closed(), timeout=30.0)
        closed_cleanly = True
    except asyncio.TimeoutError:
        pass
    check("WS feed closes on its own once the stream stops", closed_cleanly)
    await ws1.close()

    print()
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
