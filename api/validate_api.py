"""
validate_api.py — standalone, re-runnable end-to-end check for the FastAPI
/streams layer. Same style/rationale as the other validate_*.py scripts:
not a test framework, prints PASS/FAIL, exits non-zero on failure.

Requires the server to already be running (see api/main.py's module
docstring for the run command) -- this script is an HTTP client, it does
not start the server itself, since the server takes ~a minute to load all
models and this script may be re-run against an already-warm server.

Usage:
    python -m api.validate_api --base_url http://127.0.0.1:8000 --api_key some-secret --mp4 data/outputs/SNGS-021_eval75/annotated.mp4
"""

from __future__ import annotations

import argparse
import sys
import time

import requests

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    results.append(condition)


def main():
    parser = argparse.ArgumentParser(description="Validate the cv-service /streams API end-to-end")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--mp4", required=True, help="A real video file the server process can read from disk")
    parser.add_argument("--wait_ready_s", type=float, default=180.0, help="How long to wait for /health to report ready (model loading)")
    args = parser.parse_args()

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
        print("Server never became ready -- aborting remaining checks.")
        sys.exit(1)

    # 1. Auth: missing key -> 401
    r = requests.post(f"{args.base_url}/streams", json={"source_type": "file", "source_url": args.mp4})
    check("missing API key is rejected with 401", r.status_code == 401, f"got {r.status_code}")

    # 2. Auth: wrong key -> 401
    r = requests.post(f"{args.base_url}/streams", json={"source_type": "file", "source_url": args.mp4}, headers={"X-CV-Service-Key": "wrong"})
    check("wrong API key is rejected with 401", r.status_code == 401, f"got {r.status_code}")

    # 3. Unreachable source -> 424
    r = requests.post(f"{args.base_url}/streams", json={"source_type": "file", "source_url": "does/not/exist.mp4"}, headers=headers)
    body = r.json()
    check(
        "unreachable source is rejected with 424 source_unreachable",
        r.status_code == 424 and body.get("error") == "source_unreachable",
        f"status={r.status_code} body={body}",
    )

    # 4. Valid registration -> 201
    r = requests.post(f"{args.base_url}/streams", json={"source_type": "file", "source_url": args.mp4}, headers=headers)
    body = r.json()
    check("valid source registers with 201 and a stream_id", r.status_code == 201 and "stream_id" in body, f"status={r.status_code} body={body}")
    stream_id = body.get("stream_id")

    # 5. Health reflects streaming state
    r = requests.get(f"{args.base_url}/health")
    check("health reports streaming after registration", r.json().get("stream_state") == "streaming", str(r.json()))

    # 6. A second registration while active -> 409
    r = requests.post(f"{args.base_url}/streams", json={"source_type": "file", "source_url": args.mp4}, headers=headers)
    body = r.json()
    check(
        "registering while a stream is active is rejected with 409",
        r.status_code == 409 and body.get("error") == "stream_already_active" and body.get("active_stream_id") == stream_id,
        f"status={r.status_code} body={body}",
    )

    # 7. Stop with a bogus id -> 404
    r = requests.post(f"{args.base_url}/streams/not-a-real-id/stop", headers=headers)
    check("stopping a nonexistent stream returns 404", r.status_code == 404, f"got {r.status_code}")

    # 8. Stop the real stream -> 200
    r = requests.post(f"{args.base_url}/streams/{stream_id}/stop", headers=headers)
    body = r.json()
    check("stopping the active stream returns 200 and state=idle", r.status_code == 200 and body.get("state") == "idle", f"status={r.status_code} body={body}")

    # 9. Health back to idle
    r = requests.get(f"{args.base_url}/health")
    check("health reports idle after stop", r.json().get("stream_state") == "idle", str(r.json()))

    # 10. A fresh registration succeeds again (state machine fully cleared)
    r = requests.post(f"{args.base_url}/streams", json={"source_type": "file", "source_url": args.mp4}, headers=headers)
    body2 = r.json()
    check("a new registration succeeds after a clean stop", r.status_code == 201, f"status={r.status_code} body={body2}")
    if r.status_code == 201:
        requests.post(f"{args.base_url}/streams/{body2['stream_id']}/stop", headers=headers)

    print()
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
