"""
ball_tracker.py -- one ball, many candidates: the tracker of docs/plans/BALL_IMPROVEMENT_PLAN.md
§3 (item 2). Pure numpy, no package imports, so the evaluators in scripts/ and the Kaggle kit can
import this file directly.

Live (`BallTracker`), per frame:
  1. `predict(frame_idx, affine)`: move the state by the camera motion (a 2x3 affine mapping the
     previous frame's pixels into this frame's), then a constant-velocity Kalman prediction.
     Returns the predicted position, which a caller may use to place the specialist's window.
  2. `update(candidates)`: candidates inside the gate radius r(g) = min(gate_a + gate_b*g, gate_max),
     g = frames since the last accepted measurement, compete on cost d/r - lam*conf; main beats
     specialist at equal cost. Accepted -> Kalman update. None -> coast on the prediction for up
     to max_coast_frames (source "predicted"), then drop the track.
     A strong candidate (conf >= restart_conf) outside the gate on restart_frames consecutive
     frames, each within r(1) of the previous, starts a new track (a cut or a long pass).
     With no track, the strongest candidate at conf >= init_conf starts one.

Batch (`best_path`): the whole sequence is known, so pick the ball per frame by dynamic
programming over every frame's candidates (a longest path through a DAG: emission reward per
candidate, transition penalty by distance against r(gap), a start penalty per new segment), then
`interpolate` fills gaps between consecutive picks up to max_interp_frames, compensating camera
motion. (The plan names RTS smoothing too; it is left out until the DP + interpolation result
shows positions, not picks, are what's wrong.)

Confidences: candidate.conf is the model's own. Specialist confidences are multiplied by
spec_scale before any comparison, so the two sources can be put on one scale (fit on `select`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BallCandidate:
    x: float
    y: float
    conf: float
    source: str = "main"           # "main" | "specialist"
    w: float = 0.0
    h: float = 0.0


@dataclass(frozen=True)
class BallEstimate:
    x: float
    y: float
    conf: float
    source: str                    # "main" | "specialist" | "predicted"
    track_id: int
    frames_since_seen: int


@dataclass
class TrackerConfig:
    gate_a: float = 40.0           # px: gate radius at gap 0 ...
    gate_b: float = 30.0           # ... plus this much per frame since the last measurement
    gate_max: float = 500.0
    lam: float = 0.5               # weight of confidence against normalised distance
    min_conf_main: float = 0.05    # candidate floors per source
    min_conf_spec: float = 0.30
    spec_scale: float = 1.0        # specialist confidence multiplier (put it on main's scale)
    init_conf: float = 0.15        # start a track (no track yet) from a candidate this strong
    restart_conf: float = 0.50     # start a new track from a strong candidate outside the gate ...
    restart_frames: int = 2        # ... seen on this many consecutive frames
    max_coast_frames: int = 10     # predicted output after the last measurement, then drop
    q_pos: float = 4.0             # process noise (px^2 per frame) on position ...
    q_vel: float = 9.0             # ... and on velocity
    r_main: float = 4.0            # measurement noise (px^2): detector box centres
    r_spec: float = 9.0            # specialist points
    vel_damping: float = 0.85      # velocity kept per coasted frame (a coast shouldn't fly off)

    def gate(self, gap: int) -> float:
        return min(self.gate_a + self.gate_b * max(gap, 0), self.gate_max)

    def eff_conf(self, c: BallCandidate) -> float:
        return c.conf * (self.spec_scale if c.source == "specialist" else 1.0)

    def floor(self, c: BallCandidate) -> float:
        return self.min_conf_spec if c.source == "specialist" else self.min_conf_main


def apply_affine(affine, x: float, y: float) -> tuple[float, float]:
    if affine is None:
        return x, y
    a = np.asarray(affine, float)
    return float(a[0, 0] * x + a[0, 1] * y + a[0, 2]), float(a[1, 0] * x + a[1, 1] * y + a[1, 2])


def _rotate(affine, vx: float, vy: float) -> tuple[float, float]:
    if affine is None:
        return vx, vy
    a = np.asarray(affine, float)
    return float(a[0, 0] * vx + a[0, 1] * vy), float(a[1, 0] * vx + a[1, 1] * vy)


class BallTracker:
    """Online single-ball tracker. Call predict() then update() once per frame, in frame order."""

    def __init__(self, cfg: Optional[TrackerConfig] = None) -> None:
        self.cfg = cfg or TrackerConfig()
        self.reset()

    def reset(self) -> None:
        self.x: Optional[np.ndarray] = None      # state (x, y, vx, vy)
        self.P: Optional[np.ndarray] = None
        self.track_id = 0
        self.last_meas: Optional[tuple[float, float, int]] = None   # (x, y, frame) of the last accepted
        self._frame: Optional[int] = None
        self._pending: list[tuple[int, float, float]] = []          # strong out-of-gate candidates
        self._coasting = False
        self.stats = {"main": 0, "specialist": 0, "predicted": 0, "restarts": 0, "drops": 0}

    @property
    def active(self) -> bool:
        return self.x is not None

    @property
    def gap(self) -> int:
        """Frames since the last accepted measurement (at the current frame)."""
        if self.last_meas is None or self._frame is None:
            return 10 ** 6
        return self._frame - self.last_meas[2]

    @property
    def position(self) -> Optional[tuple[float, float]]:
        return None if self.x is None else (float(self.x[0]), float(self.x[1]))

    def _start(self, c: BallCandidate, frame_idx: int) -> None:
        self.track_id += 1
        self.x = np.array([c.x, c.y, 0.0, 0.0])
        r = self.cfg.r_spec if c.source == "specialist" else self.cfg.r_main
        self.P = np.diag([r, r, 400.0, 400.0])     # velocity unknown at birth
        self.last_meas = (c.x, c.y, frame_idx)
        self._pending = []
        self._coasting = False

    def predict(self, frame_idx: int, affine=None) -> Optional[tuple[float, float]]:
        """Advance to frame_idx (camera motion, then constant velocity). Returns the prediction."""
        dt = 1 if self._frame is None else max(frame_idx - self._frame, 1)
        self._frame = frame_idx
        if self.last_meas is not None and affine is not None:
            lx, ly = apply_affine(affine, self.last_meas[0], self.last_meas[1])
            self.last_meas = (lx, ly, self.last_meas[2])
        self._pending = [(f, *apply_affine(affine, px, py)) for f, px, py in self._pending]
        if self.x is None:
            return None
        x, y = apply_affine(affine, self.x[0], self.x[1])
        vx, vy = _rotate(affine, self.x[2], self.x[3])
        if self._coasting:
            vx, vy = vx * self.cfg.vel_damping, vy * self.cfg.vel_damping
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.x = F @ np.array([x, y, vx, vy])
        Q = np.diag([self.cfg.q_pos, self.cfg.q_pos, self.cfg.q_vel, self.cfg.q_vel]) * dt
        self.P = F @ self.P @ F.T + Q
        return self.position

    def update(self, candidates: Sequence[BallCandidate]) -> Optional[BallEstimate]:
        """Choose this frame's ball (call after predict). Returns None when there is none."""
        cfg, frame = self.cfg, self._frame
        if frame is None:
            raise RuntimeError("call predict(frame_idx, affine) before update()")
        cands = [c for c in candidates if c.conf >= cfg.floor(c)]

        if self.x is None:
            if cands:
                best = max(cands, key=lambda c: (cfg.eff_conf(c), c.source == "main"))
                if cfg.eff_conf(best) >= cfg.init_conf:
                    self._start(best, frame)
                    self.stats[best.source] += 1
                    return self._estimate(best.conf, best.source, 0)
            return None

        gap = frame - self.last_meas[2]
        r = cfg.gate(gap)
        px, py = self.x[0], self.x[1]
        inside = []
        for c in cands:
            d = float(np.hypot(c.x - px, c.y - py))
            if d <= r:
                inside.append((d / r - cfg.lam * cfg.eff_conf(c), c.source != "main", c))
        if inside:
            _, _, c = min(inside, key=lambda t: (t[0], t[1]))
            self._kalman_update(c)
            self.last_meas = (c.x, c.y, frame)
            self._pending = []
            self._coasting = False
            self.stats[c.source] += 1
            return self._estimate(c.conf, c.source, 0)

        # Nothing in the gate: a run of strong candidates elsewhere restarts the track.
        strong = [c for c in cands if cfg.eff_conf(c) >= cfg.restart_conf]
        if strong:
            s = max(strong, key=cfg.eff_conf)
            if self._pending and self._pending[-1][0] == frame - 1 and \
                    np.hypot(s.x - self._pending[-1][1], s.y - self._pending[-1][2]) <= cfg.gate(1):
                self._pending.append((frame, s.x, s.y))
            else:
                self._pending = [(frame, s.x, s.y)]
            if len(self._pending) >= cfg.restart_frames:
                self._start(s, frame)
                self.stats["restarts"] += 1
                self.stats[s.source] += 1
                return self._estimate(s.conf, s.source, 0)
        else:
            self._pending = []

        if gap <= cfg.max_coast_frames:
            self._coasting = True
            self.stats["predicted"] += 1
            return self._estimate(0.0, "predicted", gap)
        self.x, self.P = None, None
        self.stats["drops"] += 1
        return None

    def _kalman_update(self, c: BallCandidate) -> None:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        r = self.cfg.r_spec if c.source == "specialist" else self.cfg.r_main
        S = H @ self.P @ H.T + np.eye(2) * r
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.array([c.x, c.y]) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    def _estimate(self, conf: float, source: str, gap: int) -> BallEstimate:
        return BallEstimate(float(self.x[0]), float(self.x[1]), float(conf), source, self.track_id, gap)


# ---------------------------------------------------------------------- batch

@dataclass
class PathConfig:
    """best_path's costs. Rewards and penalties are in confidence units."""
    c0: float = 0.10               # emission reward = eff_conf - c0 (weaker candidates cost)
    w_trans: float = 0.30          # transition penalty = w_trans * (d / r(gap))^2
    skip_pen: float = 0.02         # per frame skipped between two picks
    start_pen: float = 0.30        # per new segment
    max_skip: int = 25             # longest gap a segment may bridge
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


class _Chain:
    """Camera motion over a sequence: compose(s, t) maps frame s pixels to frame t pixels in O(1),
    from cumulative transforms (affines[k]: frame k-1 -> frame k; None = no motion)."""

    def __init__(self, affines: Optional[Sequence], T: int) -> None:
        self.cum = [np.eye(3)]
        for k in range(1, T):
            a = affines[k] if affines is not None and k < len(affines) else None
            step = np.eye(3) if a is None else np.vstack([np.asarray(a, float), [0, 0, 1]])
            self.cum.append(step @ self.cum[-1])
        self.inv = [np.linalg.inv(m) for m in self.cum]

    def compose(self, s: int, t: int) -> np.ndarray:
        return self.cum[t] @ self.inv[s]


def best_path(cands: Sequence[Sequence[BallCandidate]], affines: Optional[Sequence] = None,
              cfg: Optional[PathConfig] = None) -> list[Optional[BallCandidate]]:
    """cands[t]: candidates of frame t (frames 0..T-1, consecutive). affines[t]: frame t-1 -> t
    (None or missing = no motion). Returns the chosen candidate per frame, or None."""
    cfg = cfg or PathConfig()
    tc = cfg.tracker
    T = len(cands)
    chain = _Chain(affines, T)
    nodes = [[c for c in fr if c.conf >= tc.floor(c)] for fr in cands]
    score = [np.full(len(n), -np.inf) for n in nodes]
    back: list[list] = [[None] * len(n) for n in nodes]
    best_upto: list[tuple[float, Optional[tuple[int, int]]]] = [(0.0, None)] * T
    running: tuple[float, Optional[tuple[int, int]]] = (0.0, None)
    for t in range(T):
        for j, c in enumerate(nodes[t]):
            prev_total, prev_node = best_upto[t - 1] if t > 0 else (0.0, None)
            top, arg = prev_total - cfg.start_pen, prev_node       # a new segment after the best so far
            for s in range(max(0, t - cfg.max_skip), t):
                if not nodes[s]:
                    continue
                m = chain.compose(s, t)
                g = t - s
                r = tc.gate(g - 1)
                for i, p in enumerate(nodes[s]):
                    if score[s][i] == -np.inf:
                        continue
                    qx = m[0, 0] * p.x + m[0, 1] * p.y + m[0, 2]
                    qy = m[1, 0] * p.x + m[1, 1] * p.y + m[1, 2]
                    d = float(np.hypot(c.x - qx, c.y - qy))
                    if d > r:
                        continue
                    v = score[s][i] - cfg.w_trans * (d / r) ** 2 - cfg.skip_pen * (g - 1)
                    if v > top:
                        top, arg = v, (s, i)
            score[t][j] = top + tc.eff_conf(c) - cfg.c0
            back[t][j] = arg
            if score[t][j] > running[0]:
                running = (score[t][j], (t, j))
        best_upto[t] = running

    out: list[Optional[BallCandidate]] = [None] * T
    node = running[1]
    while node is not None:
        t, j = node
        out[t] = nodes[t][j]
        node = back[t][j]
    return out


def interpolate(picks: Sequence[Optional[BallCandidate]], affines: Optional[Sequence] = None,
                max_interp_frames: int = 25) -> list[Optional[tuple[float, float]]]:
    """Fill gaps of at most max_interp_frames between consecutive picks, linearly in the earlier
    pick's camera frame, mapped forward by the camera motion. Returns per-frame (x, y) or None;
    picked frames keep their candidate's position."""
    T = len(picks)
    chain = _Chain(affines, T)
    out: list[Optional[tuple[float, float]]] = [(p.x, p.y) if p is not None else None for p in picks]
    idx = [t for t, p in enumerate(picks) if p is not None]
    for s, e in zip(idx, idx[1:]):
        g = e - s
        if g <= 1 or g - 1 > max_interp_frames:
            continue
        inv = chain.compose(e, s)
        ex, ey = picks[e].x, picks[e].y
        bx = inv[0, 0] * ex + inv[0, 1] * ey + inv[0, 2]
        by = inv[1, 0] * ex + inv[1, 1] * ey + inv[1, 2]
        ax_, ay_ = picks[s].x, picks[s].y
        for t in range(s + 1, e):
            f = (t - s) / g
            x, y = ax_ + f * (bx - ax_), ay_ + f * (by - ay_)
            m = chain.compose(s, t)
            out[t] =(m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2])
    return out
