"""
event_serializer.py — converts the raw dicts GSRPipeline.process_stream hands
to `on_frame_result` (which carry a pandas DataFrame of detections) into
plain JSON-safe dicts for the WebSocket feed.

Deliberately thin: no accumulation, no reshaping beyond picking the columns
a consumer (data_ingestion's CVPipelineSource) actually needs. Ball
detections are included alongside player/goalkeeper/referee/other rows -- the
consumer is responsible for interpreting `role`. "other" (off-pitch staff,
gsr_pipeline/athlete_roles.py) is new with the athlete/ball detector;
CVPipelineSource skips roles it does not map, with one warning.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


def _detection_row_to_dict(row: pd.Series) -> Dict[str, Any]:
    role_col = "role_detection" if "role_detection" in row.index else "role"
    return {
        "track_id": _clean(row.get("track_id")),
        "role": _clean(row.get(role_col)),
        "role_confidence": _clean(row.get("role_confidence")),
        "bbox_ltwh": [float(v) for v in row["bbox_ltwh"]] if row.get("bbox_ltwh") is not None else None,
        "pitch_x": _clean(row.get("x_bottom_middle")),
        "pitch_y": _clean(row.get("y_bottom_middle")),
    }


def _fast_row_to_dict(row: pd.Series) -> Dict[str, Any]:
    """Fast rows also carry the live team (gsr_pipeline/live_roles.py: "left" | "right" | None,
    a running vote) and, for the ball, where it came from (main detector or c5c specialist)."""
    d = _detection_row_to_dict(row)
    d["team"] = _clean(row.get("team"))
    d["team_confidence"] = _clean(row.get("team_confidence"))
    if d["role"] == "ball":
        d["ball_source"] = _clean(row.get("ball_source"))
        d["frames_since_seen"] = _clean(row.get("frames_since_seen"))
    return d


def serialize_fast_event(event: dict) -> dict:
    df = event["detections"]
    return {
        "type": "fast",
        "frame_idx": event["frame_idx"],
        "timestamp_s": event["timestamp_s"],
        "is_pitch_view": event["is_pitch_view"],
        "detections": [_fast_row_to_dict(row) for _, row in df.iterrows()],
    }


def serialize_appearance_event(event: dict) -> dict:
    df = event["detections"]
    team_results = event.get("team_results") or {}

    def _row(row: pd.Series) -> Dict[str, Any]:
        d = _detection_row_to_dict(row)
        tr = team_results.get(row.get("track_id"), {})
        d["jersey_number"] = _clean(row.get("jersey_number"))
        d["team"] = tr.get("team")
        d["team_confidence"] = _clean(tr.get("confidence"))
        d["team_locked"] = tr.get("locked")
        return d

    return {
        "type": "appearance",
        "frame_idx": event["frame_idx"],
        "detections": [_row(row) for _, row in df.iterrows()],
    }


def serialize_event(event: dict) -> dict:
    if event["type"] == "fast":
        return serialize_fast_event(event)
    if event["type"] == "appearance":
        return serialize_appearance_event(event)
    raise ValueError(f"Unknown event type: {event['type']!r}")
