"""
possession.py — Rule-based possession assignment and turnover detection.

Implements the two rule-based (no clip model) events from
docs/architecture/PIPELINE_EVOLUTION.md §3.1:

- POSSESSION: per-frame, which track/team currently has the ball.
- TURNOVER: discrete, fired when the possessing team changes.

Scope / data-availability notes
--------------------------------
- Distance is computed on the pitch-plane (x, y) coordinates `calibrate.py`
  already produces (`bbox_pitch['x_bottom_middle'/'y_bottom_middle']`). The
  pipeline has no ball-height/z data (monocular homography assumes every
  detection sits on the ground plane), so "nearest player to ball" cannot
  distinguish a grounded ball from an airborne one by height. Instead, an
  elliptical proximity zone is used with a stricter threshold on the
  pitch-width (y) axis than the pitch-length (x) axis, per
  `configs/default.yaml`'s `possession` section — see that file for the
  rationale.
- This module consumes the *final*, post-pipeline per-frame DataFrame (the
  same shape written to `predictions.json` by `output.py`): one row per
  detection per frame, with `image_id`, `track_id`, `role`, `team`, and
  `bbox_pitch`. It does not run directly off the detect+track stage the way
  PIPELINE_EVOLUTION.md's real-time design eventually calls for -- `team` is
  only available after the whole-sequence team-classification step (Phase 2),
  so for now this is a batch/offline post-processing step over a completed
  sequence, consistent with the "validate against a sample clip" scope of
  this build. Decoupling it to run incrementally off tracking-only data is
  the real-time redesign already flagged in AGENTS.md and is not attempted
  here.
- `team.py` assigns goalkeepers a side via nearest-centroid distance against
  the two fitted team clusters (fixed -- see `AGENTS.md` "Resolved since
  initial documentation"), so a goalkeeper possessor now carries a real
  `team` value in the same way a player does. The one remaining case where
  `team` is legitimately `None` for a possessor is when clustering itself
  didn't run (fewer than 2 player tracklets in the sequence) -- in that
  degenerate case no TURNOVER can be derived either, since there is no
  second team to compare against.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import require_columns

log = logging.getLogger(__name__)

# Roles eligible to "have" the ball. Referees are excluded on purpose --
# a referee never possesses the ball for tactical/possession purposes.
POSSESSOR_ROLES = ("player", "goalkeeper")


@dataclass
class PossessionConfig:
    proximity_x_m: float = 3.0
    proximity_y_m: float = 1.5
    hold_frames_threshold: int = 20

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "PossessionConfig":
        cfg = cfg or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


def _pitch_xy(bbox_pitch: Optional[dict]) -> Optional[tuple]:
    if not bbox_pitch:
        return None
    x = bbox_pitch.get("x_bottom_middle")
    y = bbox_pitch.get("y_bottom_middle")
    if x is None or y is None:
        return None
    return (x, y)


def _nearest_candidate(
    ball_xy: tuple,
    candidates: pd.DataFrame,
    cfg: PossessionConfig,
) -> Optional[dict]:
    """Return the closest candidate row (as a dict) within the proximity
    ellipse, or None if no candidate is close enough.

    A candidate is "in range" if:
        (dx / proximity_x_m)^2 + (dy / proximity_y_m)^2 <= 1
    Among in-range candidates, the one with the smallest normalized distance
    (i.e. closest to the ball, accounting for the asymmetric ellipse) wins.
    """
    bx, by = ball_xy
    best = None
    best_norm_dist = None
    for _, row in candidates.iterrows():
        xy = _pitch_xy(row.get("bbox_pitch"))
        if xy is None:
            continue
        dx, dy = xy[0] - bx, xy[1] - by
        norm_dist_sq = (dx / cfg.proximity_x_m) ** 2 + (dy / cfg.proximity_y_m) ** 2
        if norm_dist_sq > 1.0:
            continue
        if best_norm_dist is None or norm_dist_sq < best_norm_dist:
            best_norm_dist = norm_dist_sq
            best = row.to_dict()
    return best


def compute_possession_and_turnovers(
    detections_df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Compute per-frame POSSESSION records and discrete TURNOVER records.

    Args:
        detections_df: final per-frame detection DataFrame (post role/team
            assignment), with columns 'image_id', 'track_id', 'role', 'team',
            'bbox_pitch'.
        cfg: the 'possession' section of configs/default.yaml (or None to use
            defaults).

    Returns:
        (possession_records, turnover_records) — two lists of plain dicts,
        one per POSSESSION/TURNOVER event, ordered by image_id. Shaped to be
        trivially convertible into `data_ingestion.DataEvent` payloads by a
        caller, but intentionally not constructing DataEvent here (that type
        lives in the sibling `data_ingestion` repo and this module has no
        dependency on it).
    """
    require_columns(detections_df, ["image_id", "track_id", "role", "team", "bbox_pitch"], "possession")

    pcfg = PossessionConfig.from_dict(cfg)

    possession_records: List[dict] = []
    turnover_records: List[dict] = []

    last_possessor_track_id: Optional[int] = None
    last_possessor_team: Optional[str] = None
    last_confirmed_team: Optional[str] = None
    frames_since_contact = 0

    for image_id, group in detections_df.sort_values("image_id").groupby("image_id", sort=True):
        ball_rows = group[group["role"] == "ball"]
        ball_xy = _pitch_xy(ball_rows.iloc[0]["bbox_pitch"]) if not ball_rows.empty else None

        candidates = group[group["role"].isin(POSSESSOR_ROLES)]

        possessor = None
        if ball_xy is not None and not candidates.empty:
            possessor = _nearest_candidate(ball_xy, candidates, pcfg)

        if possessor is not None:
            frames_since_contact = 0
            last_possessor_track_id = possessor.get("track_id")
            last_possessor_team = possessor.get("team")
            state = "controlled"
        elif last_possessor_track_id is not None:
            frames_since_contact += 1
            if frames_since_contact <= pcfg.hold_frames_threshold:
                # Held over: ball not currently near anyone, but within the
                # grace window -- treat as still in transit to/with the last
                # known possessor rather than declaring it loose.
                state = "controlled"
            else:
                last_possessor_track_id = None
                last_possessor_team = None
                state = "uncontrolled"
        else:
            state = "uncontrolled"

        possession_records.append(
            {
                "image_id": int(image_id),
                "track_id": int(last_possessor_track_id) if last_possessor_track_id is not None else None,
                "team": last_possessor_team,
                "state": state,
            }
        )

        # TURNOVER: only fire on a genuine team-to-team change in *confirmed*
        # possession, i.e. transitioning into "controlled" with a known team
        # that differs from the last confirmed team. Loose-ball frames never
        # trigger a turnover on their own, and a team regaining the ball
        # after a loose spell (same team before and after) does not either.
        if state == "controlled" and last_possessor_team is not None:
            if last_confirmed_team is not None and last_possessor_team != last_confirmed_team:
                turnover_records.append(
                    {
                        "image_id": int(image_id),
                        "team": last_possessor_team,
                    }
                )
            last_confirmed_team = last_possessor_team

    log.info(
        "possession: %d frames processed, %d turnovers detected",
        len(possession_records),
        len(turnover_records),
    )
    return possession_records, turnover_records
