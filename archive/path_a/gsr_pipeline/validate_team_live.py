"""
validate_team_live.py — standalone, re-runnable checks for PitchViewGate and
LiveTeamAssigner (team_live.py). Same style/rationale as
validate_frame_source.py: not a test framework, a runnable script that prints
PASS/FAIL per scenario and exits non-zero on failure.

No real live-stream infra exists yet (process_stream isn't built), so these
checks use synthetic per-frame detection data with controllable appearance
features -- the same approach used to validate possession.py before any real
integration existed.

Usage:
    python -m gsr_pipeline.validate_team_live
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from gsr_pipeline.pitch_view_gate import PitchViewGate
from gsr_pipeline.team_live import LiveTeamAssigner

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    results.append(condition)


def feat(seed: int, dim: int) -> np.ndarray:
    return np.random.RandomState(seed).rand(dim)


def det(track_id, role, x, emb_seed, color_seed):
    return dict(
        track_id=track_id, role_detection=role,
        embeddings=feat(emb_seed, 512), color_features=feat(color_seed, 32),
        bbox_pitch={"x_bottom_middle": x, "y_bottom_middle": 0.0},
    )


def non_pitch_frame():
    return pd.DataFrame([dict(track_id=None, role_detection="referee", embeddings=None, color_features=None, bbox_pitch=None)])


def bootstrap_assigner(gate, cfg=None, dt=1 / 25.0):
    """Run a fresh LiveTeamAssigner through a standard bootstrap sequence:
    3s of non-pitch frames, then enough gated frames of 8 distinct,
    appearance-separated players to complete bootstrap. Returns (assigner, t).
    """
    assigner = LiveTeamAssigner(cfg or {"bootstrap_window_s": 1.0, "bootstrap_min_tracklets": 6})
    t = 0.0
    for _ in range(75):  # 3s of non-pitch frames
        fd = non_pitch_frame()
        assigner.process_frame(fd, gate.is_pitch_view(fd), t, dt)
        t += dt

    team_a, team_b = [1, 2, 3, 4], [5, 6, 7, 8]
    for _ in range(int(1.0 / dt) + 5):
        rows = [det(tid, "player", -10 - tid, 100, 1) for tid in team_a] + \
               [det(tid, "player", 10 + tid, 200, 2) for tid in team_b]
        fd = pd.DataFrame(rows)
        assigner.process_frame(fd, gate.is_pitch_view(fd), t, dt)
        t += dt
    return assigner, t


def main():
    gate = PitchViewGate({"min_player_detections": 4})
    dt = 1 / 25.0

    print("== PitchViewGate ==")
    check("rejects a non-pitch frame (no calibration, no players)", not gate.is_pitch_view(non_pitch_frame()))
    pitch_fd = pd.DataFrame([det(i, "player", i, i, i) for i in range(4)])
    check("accepts a frame with enough calibrated player detections", gate.is_pitch_view(pitch_fd))
    sparse_fd = pd.DataFrame([det(1, "player", 1, 1, 1)])
    check("rejects a frame with too few player detections", not gate.is_pitch_view(sparse_fd))

    print("== Bootstrap ==")
    assigner, t = bootstrap_assigner(gate)
    check(
        "non-pitch frames don't count toward bootstrap gated time",
        assigner._bootstrap_gated_seconds >= 1.0,
        f"gated_seconds={assigner._bootstrap_gated_seconds:.2f} (non-pitch frames excluded entirely)",
    )
    check("bootstrap completes once enough gated diverse tracklets are seen", assigner._bootstrapped)

    sparse_assigner = LiveTeamAssigner({"bootstrap_window_s": 1.0, "bootstrap_min_tracklets": 6})
    t2 = 0.0
    for _ in range(int(2.0 / dt)):  # 2s, but only 3 distinct tracklets -- below bootstrap_min_tracklets
        rows = [det(tid, "player", -10 - tid, 100 + tid, 1 + tid) for tid in [1, 2, 3]]
        fd = pd.DataFrame(rows)
        sparse_assigner.process_frame(fd, gate.is_pitch_view(fd), t2, dt)
        t2 += dt
    check(
        "bootstrap window extends rather than forcing a fit on too little diversity",
        not sparse_assigner._bootstrapped,
        "only 3 distinct tracklets seen, need 6",
    )

    print("== Per-track classification ==")
    res_a = assigner.process_frame(pd.DataFrame([det(9, "player", -30, 100, 1)]), True, t, dt)
    t += dt
    res_b = assigner.process_frame(pd.DataFrame([det(10, "player", 30, 200, 2)]), True, t, dt)
    check("clearly-separated new track locks on its FIRST frame", res_a[9]["locked"] and res_b[10]["locked"])
    check("two appearance-opposite tracks get opposite team labels", res_a[9]["team"] != res_b[10]["team"])

    mid_emb = (feat(100, 512) + feat(200, 512)) / 2
    mid_color = (feat(1, 32) + feat(2, 32)) / 2
    ambiguous_row = dict(track_id=99, role_detection="player", embeddings=mid_emb, color_features=mid_color,
                          bbox_pitch={"x_bottom_middle": 0, "y_bottom_middle": 0.0})
    t += dt
    res_amb = assigner.process_frame(pd.DataFrame([ambiguous_row]), True, t, dt)
    check(
        "an exactly-ambiguous track (midpoint appearance) does NOT lock at the default threshold",
        not res_amb[99]["locked"] and res_amb[99]["confidence"] < 0.05,
        f"confidence={res_amb[99]['confidence']:.4f} (threshold={assigner.cfg.confidence_threshold})",
    )

    next_retry = assigner._track_state[99].next_retry_s
    res_too_early = assigner.process_frame(pd.DataFrame([ambiguous_row]), True, t + 0.1, dt)
    check("unconfirmed track is NOT retried before its scheduled retry time", not res_too_early[99]["locked"])

    print("== Adaptive centroids ==")
    before = assigner._centroids[res_a[9]["team"]].vector().copy()
    strong = dict(track_id=2001, role_detection="player", embeddings=feat(100, 512), color_features=feat(1, 32),
                  bbox_pitch={"x_bottom_middle": -20, "y_bottom_middle": 0.0})
    assigner.process_frame(pd.DataFrame([strong]), True, next_retry + 0.01, dt)
    after = assigner._centroids[res_a[9]["team"]].vector()
    check("centroid measurably shifts after a confident classification", float(np.linalg.norm(after - before)) > 0)

    print()
    passed = sum(results)
    print(f"{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
