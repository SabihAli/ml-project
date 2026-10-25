"""
team_live.py — incremental, per-frame, adaptive live team assignment.

See docs/architecture/STREAMING_ARCHITECTURE.md §1.7 for the full design rationale. Summary:

- Bootstrap once per stream: accumulate appearance features from gated
  (PitchViewGate-passing) frames until `bootstrap_window_s` of *gated* time has
  been collected (not wall-clock stream time -- a pre-match crowd shot or a
  graphics overlay doesn't count and doesn't corrupt the fit), then run one
  KMeans(2) fit exactly like the batch `TeamClassifier`, reusing its feature
  construction (`gsr_pipeline.team.combine_color_reid`).
- Per track: classify on the FIRST gated frame it appears in, using that single
  frame's feature vector -- no multi-frame buffering/averaging at track birth.
  Confident (>= confidence_threshold) -> locked permanently, never reclassified.
  Unconfident -> retried every `reclassify_interval_s`, each retry a fresh
  single-frame sample, until it locks or the track ends.
- Centroids are NOT frozen after bootstrap: every classification attempt
  updates the assigned team's centroid via a confidence-weighted running mean
  (no separate learning-rate constant needed).
- Classification and centroid updates pause while `is_pitch_view` is False
  (mid-stream camera cutaway) -- see pitch_view_gate.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .pitch_view_gate import PITCH_ROLES
from .team import TeamClassifier, combine_color_reid

log = logging.getLogger(__name__)

SIDES = ("left", "right")


@dataclass
class LiveTeamConfig:
    reid_weight: float = 0.3
    bootstrap_window_s: float = 1.0
    bootstrap_min_tracklets: int = 6
    confidence_threshold: float = 0.25
    reclassify_interval_s: float = 3.0
    # Weight the one-time bootstrap KMeans fit is treated as equivalent to,
    # when seeding the adaptive running-mean centroids -- i.e. the bootstrap
    # fit counts as one confident observation, not an infinitely strong prior
    # that later per-track updates can never meaningfully shift.
    bootstrap_seed_weight: float = 1.0

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "LiveTeamConfig":
        cfg = cfg or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class _TrackState:
    team: Optional[str] = None
    confidence: Optional[float] = None
    locked: bool = False
    next_retry_s: float = 0.0


@dataclass
class _Centroid:
    W: float = 0.0
    S: Optional[np.ndarray] = None

    def vector(self) -> Optional[np.ndarray]:
        if self.W <= 0 or self.S is None:
            return None
        return self.S / self.W

    def update(self, feature: np.ndarray, weight: float) -> None:
        if weight <= 0:
            return
        if self.S is None:
            self.S = np.zeros_like(feature)
        self.W += weight
        self.S += weight * feature


class LiveTeamAssigner:
    """Incremental per-track team assignment for a single stream/match.

    One instance per active stream -- holds all bootstrap/centroid/track state
    for that stream. Not thread-safe; callers driving concurrent per-frame
    processing must serialize calls to `process_frame`.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = LiveTeamConfig.from_dict(cfg)

        self._bootstrapped = False
        self._bootstrap_rows: list = []  # dicts: track_id, embeddings, color_features
        self._bootstrap_gated_seconds = 0.0

        self._centroids: Dict[str, _Centroid] = {side: _Centroid() for side in SIDES}
        self._track_state: Dict[Any, _TrackState] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process_frame(
        self,
        frame_detections: pd.DataFrame,
        is_pitch_view: bool,
        timestamp_s: float,
        frame_duration_s: float,
    ) -> Dict[Any, dict]:
        """Process one frame's detections.

        Args:
            frame_detections: detection rows for this frame (role_detection,
                track_id, embeddings, color_features, bbox_pitch columns).
            is_pitch_view: PitchViewGate's decision for this frame -- when
                False, no classification or centroid update happens this
                frame (mid-stream cutaway handling, §1.7.2).
            timestamp_s: this frame's stream-relative timestamp, used to
                schedule/check unconfirmed-track reclassification retries.
                Real elapsed time across a gated gap is fine here -- a retry
                check just compares "has enough real time passed," gaps
                included, unlike bootstrap accumulation below.
            frame_duration_s: seconds this single frame represents (1/fps) --
                used ONLY to accumulate bootstrap gated-time, deliberately not
                derived from timestamp deltas (a delta across a skipped/
                ungated frame would wrongly count that gap as gated time).

        Returns:
            {track_id: {"team": str|None, "confidence": float|None, "locked": bool}}
            for every player/goalkeeper track_id present in this frame.
        """
        results: Dict[Any, dict] = {}
        if not is_pitch_view:
            return results

        role_col = "role_detection" if "role_detection" in frame_detections.columns else "role"
        candidates = frame_detections[
            frame_detections[role_col].isin(PITCH_ROLES) & frame_detections["track_id"].notna()
        ]
        if candidates.empty:
            return results

        if not self._bootstrapped:
            self._accumulate_bootstrap(candidates, frame_duration_s)
            self._try_finalize_bootstrap()

        for _, row in candidates.iterrows():
            tid = row["track_id"]
            state = self._track_state.get(tid)

            if not self._bootstrapped:
                # Held as "classifying" until centroids exist -- retroactively
                # evaluated on whatever frame is current once bootstrap finishes.
                if state is None:
                    state = _TrackState(next_retry_s=timestamp_s)
                    self._track_state[tid] = state
                results[tid] = {"team": state.team, "confidence": state.confidence, "locked": state.locked}
                continue

            if state is None:
                state = _TrackState(next_retry_s=timestamp_s)
                self._track_state[tid] = state
                self._attempt_classification(state, row, timestamp_s)
            elif not state.locked and timestamp_s >= state.next_retry_s:
                self._attempt_classification(state, row, timestamp_s)

            results[tid] = {"team": state.team, "confidence": state.confidence, "locked": state.locked}

        return results

    # ------------------------------------------------------------------ #
    # Bootstrap
    # ------------------------------------------------------------------ #

    def _accumulate_bootstrap(self, candidates: pd.DataFrame, frame_duration_s: float) -> None:
        self._bootstrap_gated_seconds += frame_duration_s
        for _, row in candidates.iterrows():
            bbox_pitch = row.get("bbox_pitch")
            x = bbox_pitch.get("x_bottom_middle") if bbox_pitch else None
            self._bootstrap_rows.append({
                "track_id": row["track_id"],
                "embeddings": row["embeddings"],
                "color_features": row["color_features"],
                "x": x,
            })

    def _try_finalize_bootstrap(self) -> None:
        if self._bootstrap_gated_seconds < self.cfg.bootstrap_window_s:
            return

        buffer_df = pd.DataFrame(self._bootstrap_rows)
        n_tracklets = buffer_df["track_id"].nunique() if not buffer_df.empty else 0
        if n_tracklets < self.cfg.bootstrap_min_tracklets:
            # Extend the window rather than force a split on insufficient
            # data (§1.7.3) -- keep accumulating, re-check next gated frame.
            log.debug(
                "team_live: bootstrap window elapsed (%.2fs) but only %d distinct "
                "tracklets seen (need %d) -- extending window",
                self._bootstrap_gated_seconds, n_tracklets, self.cfg.bootstrap_min_tracklets,
            )
            return

        tracklet_df = TeamClassifier._tracklet_features(buffer_df, pd.Series(True, index=buffer_df.index))
        X = TeamClassifier._combined_feature_matrix(tracklet_df, reid_weight=self.cfg.reid_weight)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
        tracklet_df = tracklet_df.copy()
        tracklet_df["cluster"] = kmeans.fit_predict(X)

        # Label clusters left/right by average pitch-x, identical to the
        # batch algorithm's side-labeling step (team.py's `process_sequence`).
        mean_x_by_track = buffer_df.dropna(subset=["x"]).groupby("track_id")["x"].mean()
        tracklet_df["avg_x"] = tracklet_df["track_id"].map(mean_x_by_track)

        cluster_avg_x = tracklet_df.groupby("cluster")["avg_x"].mean().sort_values()
        cluster_to_side = dict(zip(cluster_avg_x.index, SIDES))  # lowest avg_x -> "left"

        seed_weight = self.cfg.bootstrap_seed_weight
        for cluster_id, side in cluster_to_side.items():
            centroid_vec = kmeans.cluster_centers_[cluster_id]
            self._centroids[side] = _Centroid(W=seed_weight, S=seed_weight * centroid_vec)

        self._bootstrapped = True
        self._bootstrap_rows = []
        log.info(
            "team_live: bootstrap complete (%.2fs gated, %d tracklets) -- centroids seeded",
            self._bootstrap_gated_seconds, n_tracklets,
        )

    # ------------------------------------------------------------------ #
    # Per-track classification
    # ------------------------------------------------------------------ #

    def _attempt_classification(self, state: _TrackState, row: pd.Series, timestamp_s: float) -> None:
        feature = combine_color_reid(row["color_features"], row["embeddings"], self.cfg.reid_weight)

        distances = {}
        for side in SIDES:
            centroid = self._centroids[side].vector()
            if centroid is None:
                return  # centroids not ready -- shouldn't happen post-bootstrap, defensive only
            distances[side] = float(np.linalg.norm(feature - centroid))

        nearest_side = min(distances, key=distances.get)
        other_side = SIDES[1] if nearest_side == SIDES[0] else SIDES[0]
        d_near, d_other = distances[nearest_side], distances[other_side]
        # Symmetric relative margin: 0 when equidistant (totally ambiguous),
        # -> 1 as d_near -> 0 relative to d_other (unambiguous). NOTE: an
        # earlier version of this formula (`1 - d_near/(d_near+d_other)`)
        # simplifies to `d_other/(d_near+d_other)`, which -- since d_near is
        # by definition <= d_other -- can never fall below 0.5, silently
        # making low thresholds unreachable and defeating the unconfirmed/
        # retry path entirely. Caught by the synthetic ambiguous-track test.
        confidence = (d_other - d_near) / (d_other + d_near) if (d_near + d_other) > 0 else 0.0

        state.team = nearest_side
        state.confidence = confidence

        if confidence >= self.cfg.confidence_threshold:
            state.locked = True
        else:
            state.next_retry_s = timestamp_s + self.cfg.reclassify_interval_s

        self._centroids[nearest_side].update(feature, confidence)
