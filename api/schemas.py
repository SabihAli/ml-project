"""
schemas.py — request/response models for the /streams endpoints.

Shapes match docs/architecture/STREAMING_ARCHITECTURE.md §3.1-3.2 exactly.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class StreamRegisterRequest(BaseModel):
    source_type: Literal["rtmp", "rtsp", "hls", "file"]
    source_url: str


class StreamRegisterResponse(BaseModel):
    stream_id: str
    state: str


class StreamStopResponse(BaseModel):
    stream_id: str
    state: str


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    active_stream_id: Optional[str] = None
