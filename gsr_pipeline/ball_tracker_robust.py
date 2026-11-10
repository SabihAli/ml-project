"""
ball_tracker_robust.py -- the Phase 4 ball tracker of docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md
(§3.7): one ball, robust to occlusion, unpredictable motion and camera changes. Pure numpy, no
package imports, so the evaluators in scripts/ and the Kaggle kit import this file directly.

Per frame, in order:
  1. predict(t, affine, athletes)
       * camera: the state (position, velocity, covariance) is moved by the 2x3 camera motion of
         frame t-1 -> t. affine None (motion not estimable: a cut, a close-up, a fast pan) inflates
         the position covariance by cam_unknown_px^2 and opens a "camera break" window in which
         one strong detection is enough to restart the track;
       * motion: an IMM over two constant-velocity models, "calm" (sigma_calm px/frame^2) and
         "manoeuvre" (sigma_man), mixed with a Markov switch (p_stay_*). Kicks, deflections and
         bounces move the weight to the manoeuvre model and widen the gate on their own;
       * occlusion: while the ball is unmeasured and the prediction lies inside an athlete box
         (expanded by occ_margin), the track is attached to that athlete and follows the box
         (ball offset kept in box units) instead of extrapolating a straight line.
  2. match(cands) -> index or None, without changing the state: gated by the Mahalanobis
     distance of the innovation (chi2 gate_chi2, at least min_gate_px, at most max_gate_px);
     cost d2/gate_chi2 - lam*conf, the detector preferred at equal cost.
  3. search() -> (x, y, sigma_px) or None: where a caller should place the specialist's window
     (the prediction, not the last sighting), while the track lives and gap <= max_search_frames.
  4. update(cands) -> BallEstimate or None: the matched candidate updates the filter; otherwise
     the track coasts (source "predicted") for max_coast_frames, or max_occluded_frames while
     attached to an athlete, then it is dropped. Strong candidates outside the gate on
     restart_frames consecutive frames (1 inside a camera break) start a new track; with no
     track, the strongest candidate at conf >= init_conf starts one.

Candidates: (x, y, conf, source) with source "main" | "specialist". Athletes: (cx, cy, w, h).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


@dataclass(frozen=True)
class RobustTrackerConfig:
    sigma_calm: float = 1.5          # px/frame^2 acceleration noise, calm model
    sigma_man: float = 12.0          # manoeuvre model (kicks, deflections)
    p_stay_calm: float = 0.95
    p_stay_man: float = 0.80
    r_main: float = 3.0              # measurement std (px): detector box centre
    r_spec: float = 5.0              # specialist point
    gate_chi2: float = 13.8          # 99.9% for 2 dof
    min_gate_px: float = 25.0
    max_gate_px: float = 400.0
    lam: float = 0.5
    init_conf: float = 0.15
    restart_conf: float = 0.5
    restart_frames: int = 2
    cam_unknown_px: float = 80.0
    cam_break_frames: int = 25       # frames after a camera break with the 1-frame restart
    cut_restart_conf: float = 0.3
    max_coast_frames: int = 10
    occlusion: bool = True
    occ_margin: float = 0.15         # athlete box expansion (fraction of w, h)
    max_occluded_frames: int = 50
    occ_q_px: float = 6.0            # position std growth per occluded frame
    vel_damping: float = 0.9         # velocity kept per coasted (free) frame
    max_search_frames: int = 30
    camera: bool = True              # use the camera motion at all (ablation)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BallEstimate:
    x: float
    y: float
    conf: float
    source: str                      # "main" | "specialist" | "predicted"
    track_id: int
    frames_since_seen: int
    occluded: bool = False


def _q(sigma: float, dt: float) -> np.ndarray:
    g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    return g @ g.T * sigma * sigma


def _F(dt: float) -> np.ndarray:
    return np.array([[1.0, 0, dt, 0], [0, 1.0, 0, dt], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])


class RobustBallTracker:
    """Online single-ball tracker. Per frame: predict(), then match()/search() as needed, then update()."""

    def __init__(self, cfg: Optional[RobustTrackerConfig] = None) -> None:
        self.cfg = cfg or RobustTrackerConfig()
        c = self.cfg
        self.PI = np.array([[c.p_stay_calm, 1 - c.p_stay_calm], [1 - c.p_stay_man, c.p_stay_man]])
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.xs: Optional[list] = None   # per-model state (4,)
        self.Ps: Optional[list] = None
        self.mu = np.array([0.9, 0.1])
        self.track_id = 0
        self.frame: Optional[int] = None
        self.last_meas_frame: Optional[int] = None
        self.cam_break_until = -1
        self._pending: list = []         # (frame, x, y) strong out-of-gate candidates
        self.occ: Optional[dict] = None  # {"box": (cx, cy, w, h), "off": (ox, oy)} offset in box units
        self.stats = {"main": 0, "specialist": 0, "predicted": 0, "occluded": 0, "restarts": 0,
                      "drops": 0, "cam_breaks": 0}

    @property
    def active(self) -> bool:
        return self.xs is not None

    @property
    def gap(self) -> int:
        if self.last_meas_frame is None or self.frame is None:
            return 10 ** 6
        return self.frame - self.last_meas_frame

    def _combined(self):
        x = sum(m * xi for m, xi in zip(self.mu, self.xs))
        P = sum(m * (Pi + np.outer(xi - x, xi - x)) for m, xi, Pi in zip(self.mu, self.xs, self.Ps))
        return x, P

    @property
    def position(self) -> Optional[tuple]:
        if not self.active:
            return None
        x, _ = self._combined()
        return float(x[0]), float(x[1])

    def _start(self, x: float, y: float, frame: int, source: str) -> None:
        self.track_id += 1
        r = self.cfg.r_spec if source == "specialist" else self.cfg.r_main
        P = np.diag([r * r, r * r, 400.0, 400.0])     # velocity unknown at birth
        self.xs = [np.array([x, y, 0.0, 0.0]), np.array([x, y, 0.0, 0.0])]
        self.Ps = [P.copy(), P.copy()]
        self.mu = np.array([0.5, 0.5])
        self.last_meas_frame = frame
        self._pending = []
        self.occ = None

    # ------------------------------------------------------------------ predict
    def predict(self, frame: int, affine=None, athletes: Sequence = ()) -> Optional[tuple]:
        c = self.cfg
        dt = 1.0 if self.frame is None else float(max(frame - self.frame, 1))
        first = self.frame is None
        self.frame = frame
        A = np.asarray(affine, float) if (affine is not None and c.camera) else None
        if affine is None and c.camera and not first:
            self.cam_break_until = frame + c.cam_break_frames
            self.stats["cam_breaks"] += 1
        if A is not None:
            self._pending = [(f, *self._warp(A, px, py)) for f, px, py in self._pending]
        if not self.active:
            return None
        if A is not None:
            L = A[:, :2]
            J = np.zeros((4, 4))
            J[:2, :2] = L
            J[2:, 2:] = L
            for i in range(2):
                p = self.xs[i]
                px, py = self._warp(A, p[0], p[1])
                v = L @ p[2:]
                self.xs[i] = np.array([px, py, v[0], v[1]])
                self.Ps[i] = J @ self.Ps[i] @ J.T
            if self.occ is not None:
                bx, by, bw, bh = self.occ["box"]
                nx, ny = self._warp(A, bx, by)
                s = math.sqrt(abs(np.linalg.det(L)))
                self.occ["box"] = (nx, ny, bw * s, bh * s)
        elif affine is None and c.camera and not first:
            for i in range(2):
                self.Ps[i] = self.Ps[i] + np.diag([c.cam_unknown_px ** 2] * 2 + [100.0, 100.0])
        # IMM mixing, then each model's prediction
        cbar = self.PI.T @ self.mu
        mix = (self.PI * self.mu[:, None]) / np.maximum(cbar[None, :], 1e-12)
        x0 = [sum(mix[i, j] * self.xs[i] for i in range(2)) for j in range(2)]
        P0 = [sum(mix[i, j] * (self.Ps[i] + np.outer(self.xs[i] - x0[j], self.xs[i] - x0[j])) for i in range(2))
              for j in range(2)]
        F = _F(dt)
        coasting = self.gap >= 1
        for j, sig in enumerate((c.sigma_calm, c.sigma_man)):
            xj = x0[j].copy()
            if coasting and self.occ is None:
                xj[2:] *= c.vel_damping
            self.xs[j] = F @ xj
            self.Ps[j] = F @ P0[j] @ F.T + _q(sig, dt)
        self.mu = cbar
        if c.occlusion and self.gap >= 1:
            self._follow_athlete(athletes)
        return self.position

    @staticmethod
    def _warp(A, x, y):
        return float(A[0, 0] * x + A[0, 1] * y + A[0, 2]), float(A[1, 0] * x + A[1, 1] * y + A[1, 2])

    def _inside(self, x, y, box) -> bool:
        cx, cy, w, h = box[:4]
        m = self.cfg.occ_margin
        return abs(x - cx) <= w * (0.5 + m) and abs(y - cy) <= h * (0.5 + m)

    def _follow_athlete(self, athletes) -> None:
        """Attach the unmeasured ball to the athlete box it is hidden in, and move it with that box."""
        c = self.cfg
        boxes = [tuple(a[:4]) for a in athletes]
        if self.occ is None:
            x, y = self.position
            inside = [b for b in boxes if self._inside(x, y, b)]
            if not inside:
                return
            b = min(inside, key=lambda b: math.hypot(b[0] - x, b[1] - y))
            self.occ = {"box": b, "off": ((x - b[0]) / max(b[2], 1.0), (y - b[1]) / max(b[3], 1.0))}
            self.stats["occluded"] += 1
            return
        if not boxes:
            return
        ox, oy, ow, oh = self.occ["box"]
        b = min(boxes, key=lambda b: math.hypot(b[0] - ox, b[1] - oy))
        if math.hypot(b[0] - ox, b[1] - oy) > max(ow, oh):
            self.occ = None          # the athlete was lost: back to free coasting
            return
        nx, ny = b[0] + self.occ["off"][0] * b[2], b[1] + self.occ["off"][1] * b[3]
        vx, vy = b[0] - ox, b[1] - oy
        for j in range(2):
            self.xs[j] = np.array([nx, ny, vx, vy])
            self.Ps[j][:2, :2] += np.eye(2) * c.occ_q_px ** 2
        self.occ["box"] = b

    # ------------------------------------------------------------------ association
    def _gate(self, cands):
        """-> [(cost, not_main, i)] for the candidates inside the gate."""
        c = self.cfg
        if not self.active:
            return []
        x, P = self._combined()
        out = []
        for i, k in enumerate(cands):
            r = c.r_spec if k[3] == "specialist" else c.r_main
            S = H @ P @ H.T + np.eye(2) * r * r
            nu = np.array([k[0] - x[0], k[1] - x[1]])
            d2 = float(nu @ np.linalg.solve(S, nu))
            dist = math.hypot(nu[0], nu[1])
            if dist > c.max_gate_px or (d2 > c.gate_chi2 and dist > c.min_gate_px):
                continue
            out.append((min(d2, c.gate_chi2) / c.gate_chi2 - c.lam * k[2], k[3] != "main", i))
        return out

    def match(self, cands: Sequence[tuple]) -> Optional[int]:
        g = self._gate(cands)
        return min(g)[2] if g else None

    def search(self) -> Optional[tuple]:
        if not self.active or self.gap > self.cfg.max_search_frames:
            return None
        x, P = self._combined()
        return float(x[0]), float(x[1]), float(math.sqrt(max(P[0, 0], P[1, 1])))

    # ------------------------------------------------------------------ update
    def update(self, cands: Sequence[tuple]) -> Optional[BallEstimate]:
        c, frame = self.cfg, self.frame
        if frame is None:
            raise RuntimeError("call predict() before update()")
        if not self.active:
            pool = [k for k in cands if k[2] >= c.init_conf]
            if not pool:
                return None
            k = max(pool, key=lambda k: (k[2], k[3] == "main"))
            self._start(k[0], k[1], frame, k[3])
            self.stats[k[3]] += 1
            return self._est(k[2], k[3])
        i = self.match(cands)
        if i is not None:
            k = cands[i]
            self._kalman(k)
            self.last_meas_frame = frame
            self._pending = []
            self.occ = None
            self.stats[k[3]] += 1
            return self._est(k[2], k[3])
        # strong detections outside the gate restart the track (a cut, a long pass)
        in_break = frame <= self.cam_break_until
        need = 1 if in_break else c.restart_frames
        floor = c.cut_restart_conf if in_break else c.restart_conf
        strong = [k for k in cands if k[3] == "main" and k[2] >= floor]
        if strong:
            s = max(strong, key=lambda k: k[2])
            if self._pending and self._pending[-1][0] == frame - 1 and \
                    math.hypot(s[0] - self._pending[-1][1], s[1] - self._pending[-1][2]) <= c.min_gate_px * 2:
                self._pending.append((frame, s[0], s[1]))
            else:
                self._pending = [(frame, s[0], s[1])]
            if len(self._pending) >= need:
                self._start(s[0], s[1], frame, "main")
                self.stats["restarts"] += 1
                self.stats["main"] += 1
                return self._est(s[2], "main")
        else:
            self._pending = []
        limit = c.max_occluded_frames if self.occ is not None else c.max_coast_frames
        if self.gap <= limit:
            self.stats["predicted"] += 1
            return self._est(0.0, "predicted")
        self.xs, self.Ps, self.occ = None, None, None
        self.stats["drops"] += 1
        return None

    def _kalman(self, k) -> None:
        r = self.cfg.r_spec if k[3] == "specialist" else self.cfg.r_main
        R = np.eye(2) * r * r
        z = np.array([k[0], k[1]])
        lik = np.zeros(2)
        for j in range(2):
            S = H @ self.Ps[j] @ H.T + R
            nu = z - H @ self.xs[j]
            K = self.Ps[j] @ H.T @ np.linalg.inv(S)
            self.xs[j] = self.xs[j] + K @ nu
            self.Ps[j] = (np.eye(4) - K @ H) @ self.Ps[j]
            d2 = float(nu @ np.linalg.solve(S, nu))
            lik[j] = math.exp(-0.5 * min(d2, 700.0)) / (2 * math.pi * math.sqrt(max(np.linalg.det(S), 1e-12)))
        mu = self.mu * lik
        self.mu = mu / mu.sum() if mu.sum() > 1e-300 else np.array([0.5, 0.5])

    def _est(self, conf: float, source: str) -> BallEstimate:
        x, y = self.position
        return BallEstimate(x, y, float(conf), source, self.track_id, max(self.gap, 0), self.occ is not None)
