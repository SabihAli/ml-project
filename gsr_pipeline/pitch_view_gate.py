"""
pitch_view_gate.py — "is this frame actually a pitch view" gate.

A reusable primitive, not a bootstrap-only detail. Used to:
  - Gate live team-assignment bootstrap accumulation (team_live.py) so a
    pre-match crowd shot, graphics overlay, or replay doesn't count toward
    (or corrupt) the initial team-centroid fit.
  - Pause ongoing per-frame team classification / centroid updates during a
    mid-match cutaway to a non-pitch camera angle (bench, crowd, replay,
    graphics overlay), so a cutaway doesn't spawn spurious "new tracks" that
    corrupt a centroid.

See docs/architecture/STREAMING_ARCHITECTURE.md §1.7.2.

A frame counts as "on the pitch" only if:
  1. Pitch calibration succeeded for that frame (bbox_pitch populated), AND
  2. At least `min_player_detections` player/goalkeeper detections are present.

Both signals are already computed by the pipeline (calibrate.py, detect.py) --
this gate adds no new model, it only combines two existing signals into one
reusable decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

PITCH_ROLES = ("player", "goalkeeper")


@dataclass
class PitchViewGateConfig:
    min_player_detections: int = 4

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "PitchViewGateConfig":
        cfg = cfg or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class PitchViewGate:
    """Stateless per-frame gate -- call `is_pitch_view` once per frame."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = PitchViewGateConfig.from_dict(cfg)

    def is_pitch_view(self, frame_detections: pd.DataFrame) -> bool:
        """Args:
            frame_detections: detection rows for a SINGLE frame. Accepts either
                'role_detection' (Phase 1, pre-tracklet-aggregation) or 'role'
                (post-aggregation/batch) -- whichever is present is used, so this
                gate works identically in both the live and batch code paths.

        Returns:
            True if this frame should count as a usable pitch view.
        """
        if frame_detections is None or frame_detections.empty:
            return False

        role_col = "role_detection" if "role_detection" in frame_detections.columns else "role"
        if role_col not in frame_detections.columns:
            return False

        if "bbox_pitch" not in frame_detections.columns:
            return False
        calibrated = frame_detections["bbox_pitch"].notna().any()
        if not calibrated:
            return False

        player_count = frame_detections[role_col].isin(PITCH_ROLES).sum()
        return bool(player_count >= self.cfg.min_player_detections)
