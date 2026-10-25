"""
athlete_roles.py — role + team classification for role-agnostic "athlete" detections.

A detector that only outputs `athlete` / `ball` (e.g. runs/yolo26s_athlete_ball)
loses the player / goalkeeper / referee distinction the rest of the pipeline relies on.
This module restores it, plus a team split, without a trained role model:

    player_team_a, player_team_b, goalkeeper_team_a, goalkeeper_team_b, referee, other

Method (per match):
  1. Colour: median Lab colour of the non-grass pixels in each box's torso region.
  2. `fit` on a bootstrap window: over-cluster the colours with KMeans, merge clusters that
     are the same kit under different lighting, and take the two kits that appear several
     times per frame as the two teams. Everything else is a "minor" colour (keepers,
     officials, staff).
  3. `predict` assigns each detection to its nearest fitted colour, then:
       team colour                          -> player_team_a / player_team_b
       minor colour, near a goal line       -> goalkeeper (team = the team defending that end)
       minor colour, anywhere else          -> referee
       outside the pitch by a margin        -> other   (overrides colour)

Inputs are colour descriptors, pitch positions (metres from the pitch centre, the same
convention as the calibration output) and frame ids. Detections whose pitch projection is
missing should be skipped by the caller.

Known limits: a keeper or referee kit close to a team kit is absorbed by that team; `other`
is position-only (on-pitch staff are not caught); `team_a` / `team_b` are arbitrary names
fixed at fit time (a = the kit that defecurnded the lower-x end then).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List, Optional, Sequence

import cv2
import numpy as np
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)

PLAYER_A, PLAYER_B = "player_team_a", "player_team_b"
GOALKEEPER_A, GOALKEEPER_B = "goalkeeper_team_a", "goalkeeper_team_b"
REFEREE, OTHER = "referee", "other"
LABELS = (PLAYER_A, PLAYER_B, GOALKEEPER_A, GOALKEEPER_B, REFEREE, OTHER)

_TORSO_X = (0.2, 0.8)
_TORSO_Y = (0.15, 0.50)


def torso_descriptor(frame_bgr: np.ndarray, ltrb: Sequence[float]) -> Optional[np.ndarray]:
    """Median Lab colour of the non-grass pixels in the upper-body region of a box.

    Returns None when the box is too small to sample (< 3px in either direction).
    """
    height, width = frame_bgr.shape[:2]
    left, top, right, bottom = ltrb
    w, h = right - left, bottom - top
    x0, x1 = int(max(0, left + _TORSO_X[0] * w)), int(min(width, left + _TORSO_X[1] * w))
    y0, y1 = int(max(0, top + _TORSO_Y[0] * h)), int(min(height, top + _TORSO_Y[1] * h))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    crop = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] > 40) & (hsv[..., 2] > 30)
    keep = ~grass
    if keep.sum() < 0.25 * keep.size:  # crop is mostly grass-coloured: fall back to all pixels
        keep = np.ones_like(keep)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[keep.reshape(-1)]
    return np.median(lab, axis=0)


class AthleteRoleTeamClassifier:
    def __init__(
        self,
        n_clusters: int = 6,
        merge_dist: float = 25.0,
        l_weight: float = 0.5,
        team_min_per_frame: float = 2.5,
        gk_zone_m: float = 20.0,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        offpitch_margin_x_m: float = 5.5,
        offpitch_margin_y_m: float = 6.0,
        random_state: int = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.merge_dist = merge_dist
        self.l_weight = l_weight
        self.team_min_per_frame = team_min_per_frame
        self.gk_zone_m = gk_zone_m
        self.half_length = pitch_length_m / 2
        self.half_width = pitch_width_m / 2
        self.offpitch_x = self.half_length + offpitch_margin_x_m
        self.offpitch_y = self.half_width + offpitch_margin_y_m
        self.random_state = random_state
        self._centers: dict[int, np.ndarray] = {}
        self._team_cluster: dict[str, int] = {}

    @property
    def fitted(self) -> bool:
        return bool(self._team_cluster)

    def _weight(self, colours: np.ndarray) -> np.ndarray:
        return np.asarray(colours, dtype=float) * np.array([self.l_weight, 1.0, 1.0])

    def fit(self, descriptors: np.ndarray, pitch_xy: np.ndarray, frame_ids: Sequence) -> "AthleteRoleTeamClassifier":
        colours = self._weight(descriptors)
        pitch_xy = np.asarray(pitch_xy, dtype=float)
        n_frames = len(set(frame_ids))
        k = min(self.n_clusters, len(colours))
        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(colours)
        labels = km.labels_.copy()
        centers = {c: km.cluster_centers_[c].copy() for c in range(k)}

        while True:
            ids = sorted(set(labels))
            closest = min(
                ((np.linalg.norm(centers[a] - centers[b]), a, b) for i, a in enumerate(ids) for b in ids[i + 1:]),
                default=None,
            )
            if closest is None or closest[0] >= self.merge_dist:
                break
            _, a, b = closest
            na, nb = (labels == a).sum(), (labels == b).sum()
            centers[a] = (centers[a] * na + centers[b] * nb) / (na + nb)
            del centers[b]
            labels[labels == b] = a

        sizes = Counter(labels)
        ranked = [c for c, _ in sizes.most_common()]
        teams = [c for c in ranked if sizes[c] / n_frames >= self.team_min_per_frame][:2]
        teams += [c for c in ranked if c not in teams][: 2 - len(teams)]
        if len(teams) < 2:
            raise ValueError("fit needs at least two distinct colour clusters to separate the teams")

        mean_x = {c: pitch_xy[labels == c, 0].mean() for c in teams}
        team_a, team_b = sorted(teams, key=lambda c: mean_x[c])
        self._centers = centers
        self._team_cluster = {"a": team_a, "b": team_b}
        log.info("AthleteRoleTeamClassifier fitted: %d colour clusters (%d minor), team sizes %s",
                 len(centers), len(centers) - 2, {t: int(sizes[c]) for t, c in self._team_cluster.items()})
        return self

    def predict(self, descriptors: np.ndarray, pitch_xy: np.ndarray,
                team_a_defends_negative_x: bool = True) -> List[str]:
        """Label each detection. Pass team_a_defends_negative_x=False after half-time so a
        goalkeeper is attributed to the right team."""
        if not self.fitted:
            raise RuntimeError("call fit() before predict()")
        ids = list(self._centers)
        centre_matrix = np.array([self._centers[c] for c in ids])
        nearest = [ids[i] for i in np.argmin(
            np.linalg.norm(self._weight(descriptors)[:, None, :] - centre_matrix[None], axis=2), axis=1)]
        team_of_cluster = {c: t for t, c in self._team_cluster.items()}

        out = []
        for cluster, (x, y) in zip(nearest, np.asarray(pitch_xy, dtype=float)):
            if abs(x) > self.offpitch_x or abs(y) > self.offpitch_y:
                out.append(OTHER)
            elif cluster in team_of_cluster:
                out.append(PLAYER_A if team_of_cluster[cluster] == "a" else PLAYER_B)
            elif self.half_length - abs(x) < self.gk_zone_m:
                at_negative_end = x < 0
                team_a = at_negative_end == team_a_defends_negative_x
                out.append(GOALKEEPER_A if team_a else GOALKEEPER_B)
            else:
                out.append(REFEREE)
        return out
