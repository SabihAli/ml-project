"""
live_roles.py -- roles and teams for the athlete/ball detector's role-agnostic `athlete` boxes,
live (per frame) and batch (per sequence).

Both wrap athlete_roles.AthleteRoleTeamClassifier, which fits kit colours on a window of frames
and labels each box player_team_a/b, goalkeeper_team_a/b, referee or other. The batch form is
what notebooks/path_b_eval.py measured (fit on the first `fit_frames` frames, predict every
frame, majority vote per tracklet); the live form does the same thing incrementally:

  * until fitted: buffer torso colours of boxes that have a pitch position; role "player",
    team None (the wire still gets positions every frame);
  * at `fit_frames` buffered frames: fit, predict the buffer, and seed each track's vote;
    a failed fit (one kit visible, cutaways) retries every `refit_every_frames` frames;
  * after: predict each box and add it to its track's running vote. A track's role and team
    are its vote's leaders; confidence is the leader's share of the votes.

Wire values: `role` is player | goalkeeper | referee | other | ball (the names
data_ingestion's CVPipelineSource maps); `team` is "left" (team_a, the kit defending the
lower-x end at fit time) or "right", the same labels the batch output always used.

Known limits (from athlete_roles.py): team_a / team_b are fixed at fit time, so a half-time
end swap is not followed (no half-time flip yet, IMPLEMENTATION_PLAN.md 4.2); a keeper or
referee kit close to a team kit is absorbed by that team.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from .athlete_roles import AthleteRoleTeamClassifier, torso_descriptor

log = logging.getLogger(__name__)

ATHLETE_CLASS, BALL_CLASS = 0, 1
TEAM_SIDE = {"a": "left", "b": "right"}


def split_label(label: str) -> tuple[str, Optional[str]]:
    """athlete_roles label -> (role, team side or None). 'goalkeeper_team_b' -> ('goalkeeper', 'right')."""
    role = label.split("_")[0]
    team = TEAM_SIDE[label[-1]] if label.startswith(("player_team", "goalkeeper_team")) else None
    return role, team


def _pitch_xy(df: pd.DataFrame) -> np.ndarray:
    return df[["x_bottom_middle", "y_bottom_middle"]].to_numpy(dtype=float)


def _usable(df: pd.DataFrame) -> pd.Series:
    """Athlete rows with a pitch position (the classifier needs it) -- descriptor checked later."""
    if "x_bottom_middle" not in df.columns:
        return pd.Series(False, index=df.index)
    return (df["class_id"] == ATHLETE_CLASS) & df["x_bottom_middle"].notna() & df["y_bottom_middle"].notna()


class LiveAthleteRoles:
    """Per-stream role/team state. Not thread-safe; one per stream, driven from its frame loop."""

    def __init__(
        self,
        athlete_roles_cfg: Optional[dict] = None,
        fit_frames: int = 250,
        refit_every_frames: int = 50,
        max_buffer_frames: int = 1500,
        min_athletes_for_pitch_view: int = 4,
    ) -> None:
        """
        Args:
            athlete_roles_cfg: AthleteRoleTeamClassifier keyword args (config `athlete_roles`).
            fit_frames: buffered frames (with at least one usable box) before the first fit.
                250 = 10 s at 25 fps, the window path_b_eval measured.
            refit_every_frames: after a failed fit, buffered frames between retries.
            max_buffer_frames: buffer cap; the oldest frames are dropped beyond it.
            min_athletes_for_pitch_view: `is_pitch_view` needs calibration plus this many
                athletes with a pitch position (cutaways and close-ups fail it).
        """
        self._clf_cfg = dict(athlete_roles_cfg or {})
        self.fit_frames = fit_frames
        self.refit_every_frames = refit_every_frames
        self.max_buffer_frames = max_buffer_frames
        self.min_athletes = min_athletes_for_pitch_view
        self.reset()

    def reset(self) -> None:
        self.classifier: Optional[AthleteRoleTeamClassifier] = None
        self._buffer: list[tuple[int, list, np.ndarray, np.ndarray]] = []   # (frame, track ids, desc, xy)
        self._next_fit_at = self.fit_frames
        self._votes: dict = defaultdict(Counter)
        self.last_fit_error: Optional[str] = None

    @property
    def fitted(self) -> bool:
        return self.classifier is not None

    def update(self, frame_idx: int, frame_bgr: np.ndarray, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Label one frame's detections in place of the old detector roles.

        Adds `role_detection`, `role_confidence`, `team`, `team_confidence`. Returns
        (df, is_pitch_view)."""
        df = df.copy()
        is_ball = df["class_id"] == BALL_CLASS
        usable = _usable(df)
        rows = df.index[usable]
        descs = [torso_descriptor(frame_bgr, df.at[i, "bbox_ltrb"]) for i in rows]
        keep = [i for i, d in zip(rows, descs) if d is not None]
        desc = np.array([d for d in descs if d is not None])
        xy = _pitch_xy(df.loc[keep]) if keep else np.zeros((0, 2))
        tracks = df.loc[keep, "track_id"].tolist() if keep else []
        is_pitch_view = len(keep) >= self.min_athletes

        if not self.fitted and keep:
            self._buffer.append((frame_idx, tracks, desc, xy))
            if len(self._buffer) > self.max_buffer_frames:
                self._buffer.pop(0)
            if len(self._buffer) >= self._next_fit_at:
                self._fit()

        labels_by_row: dict = {}
        if self.fitted and keep:
            for i, track, label in zip(keep, tracks, self.classifier.predict(desc, xy)):
                labels_by_row[i] = label
                if pd.notna(track):
                    self._votes[int(track)][label] += 1

        roles, role_conf, teams, team_conf = [], [], [], []
        for i in df.index:
            if is_ball.at[i]:
                roles.append("ball"); role_conf.append(float(df.at[i, "bbox_conf"]))
                teams.append(None); team_conf.append(None)
                continue
            role, rconf, team, tconf = self._track_role(df.at[i, "track_id"], labels_by_row.get(i))
            roles.append(role); role_conf.append(rconf); teams.append(team); team_conf.append(tconf)
        df["role_detection"] = roles
        df["role_confidence"] = role_conf
        df["team"] = teams
        df["team_confidence"] = team_conf
        return df, is_pitch_view

    def track_team(self, track) -> tuple[Optional[str], Optional[float]]:
        """A track's current (team, confidence) from its vote; (None, None) if it has none."""
        _role, _rconf, team, tconf = self._track_role(track, None)
        return team, tconf

    def _track_role(self, track, row_label: Optional[str]):
        """(role, role confidence, team, team confidence) from a track's vote; an untracked box
        uses its own label; nothing yet -> ('player', None, None, None)."""
        votes = self._votes.get(int(track)) if pd.notna(track) else None
        if not votes:
            if row_label is None:
                return "player", None, None, None
            role, team = split_label(row_label)
            return role, None, team, None
        by_role: Counter = Counter()
        for label, n in votes.items():
            by_role[split_label(label)[0]] += n
        total = sum(by_role.values())
        role, n_role = by_role.most_common(1)[0]
        sides = Counter({split_label(lb)[1]: n for lb, n in votes.items()
                         if split_label(lb)[0] == role and split_label(lb)[1]})
        if not sides:
            return role, n_role / total, None, None
        team, n_team = sides.most_common(1)[0]
        return role, n_role / total, team, n_team / sum(sides.values())

    def _fit(self) -> None:
        frames = [f for f, tracks, _d, _p in self._buffer for _ in tracks]
        desc = np.concatenate([d for _f, _t, d, _p in self._buffer])
        xy = np.concatenate([p for _f, _t, _d, p in self._buffer])
        try:
            clf = AthleteRoleTeamClassifier(**self._clf_cfg).fit(desc, xy, frames)
        except (ValueError, RuntimeError) as exc:
            self.last_fit_error = str(exc)
            self._next_fit_at = len(self._buffer) + self.refit_every_frames
            log.warning("live roles: fit on %d frames failed (%s); retrying after %d more",
                        len(self._buffer), exc, self.refit_every_frames)
            return
        self.classifier = clf
        for _f, tracks, d, p in self._buffer:
            for track, label in zip(tracks, clf.predict(d, p)):
                if pd.notna(track):
                    self._votes[int(track)][label] += 1
        log.info("live roles: fitted on %d frames (%d boxes)", len(self._buffer), len(desc))
        self._buffer = []


def assign_sequence_roles(df: pd.DataFrame, athlete_roles_cfg: Optional[dict], fit_frames: int = 250) -> dict:
    """Batch: fit on the first `fit_frames` frames (0 = all), predict every usable box, then a
    majority vote per tracklet -- notebooks/path_b_eval.py's assign_roles_and_teams on a
    DataFrame. Needs a `torso_desc` column. Sets `role_detection` (the track's role, so
    tracklet_agg's role vote agrees with it) and `team`, in place. Returns fit info."""
    athletes = df["class_id"] == ATHLETE_CLASS
    usable = _usable(df) & df["torso_desc"].notna()
    frames = sorted(df.loc[usable, "image_id"].unique())
    fit_ids = set(frames if fit_frames <= 0 else frames[:fit_frames])
    fit_mask = usable & df["image_id"].isin(fit_ids)

    info = {"fit_rows": int(fit_mask.sum()), "usable_rows": int(usable.sum()), "fit_ok": False, "fit_error": None}
    labels = pd.Series(None, index=df.index, dtype=object)
    try:
        clf = AthleteRoleTeamClassifier(**(athlete_roles_cfg or {})).fit(
            np.stack(df.loc[fit_mask, "torso_desc"].to_numpy()), _pitch_xy(df[fit_mask]),
            df.loc[fit_mask, "image_id"].tolist())
        labels[usable] = clf.predict(np.stack(df.loc[usable, "torso_desc"].to_numpy()), _pitch_xy(df[usable]))
        info["fit_ok"] = True
    except (ValueError, RuntimeError) as exc:
        info["fit_error"] = str(exc)
        log.warning("batch roles: fit failed (%s); every athlete stays 'player' with no team", exc)

    df["role_detection"] = np.where(athletes, "player", "ball")
    df["team"] = None
    for track, idx in df[athletes].groupby("track_id", dropna=False).groups.items():
        if pd.isna(track):            # untracked boxes keep their own label
            for i in idx:
                if labels[i] is not None:
                    df.at[i, "role_detection"], df.at[i, "team"] = split_label(labels[i])
            continue
        lab = [split_label(x) for x in labels[idx].dropna()]
        if not lab:
            continue
        role = Counter(r for r, _ in lab).most_common(1)[0][0]
        sides = Counter(t for r, t in lab if t and r == role)
        df.loc[idx, "role_detection"] = role
        if sides and role in ("player", "goalkeeper"):
            df.loc[idx, "team"] = sides.most_common(1)[0][0]
    info["role_counts"] = dict(Counter(df.loc[athletes, "role_detection"]))
    return info
