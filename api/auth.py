"""
auth.py — static API-key auth for the /streams endpoints.

Decided in docs/architecture/STREAMING_ARCHITECTURE.md §3.3: a static API-key header is
sufficient for an internal service-to-service call (cv-service <-
data_ingestion's CVPipelineSource), not a public-facing endpoint. The key
lives in an environment variable, never committed. Upgradeable to something
stronger (mTLS, OAuth) later without changing the request/response shapes in
schemas.py.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException

API_KEY_ENV_VAR = "CV_SERVICE_API_KEY"
API_KEY_HEADER = "X-CV-Service-Key"


def require_api_key(x_cv_service_key: str | None = Header(default=None, alias=API_KEY_HEADER)) -> None:
    expected = os.environ.get(API_KEY_ENV_VAR)
    if not expected:
        # Fail closed, not open: an unset key means auth cannot be enforced,
        # which must never silently degrade to "endpoint is open."
        raise HTTPException(
            status_code=500,
            detail=f"Server misconfigured: {API_KEY_ENV_VAR} environment variable is not set.",
        )
    if x_cv_service_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
