"""BallTracker (online) and best_path / interpolate (batch), pure logic on hand-built candidates."""

import numpy as np
import pytest

from gsr_pipeline.ball_tracker import (BallCandidate as C, BallTracker, PathConfig, TrackerConfig,
                                       best_path, interpolate)


def run(tracker, frames, affines=None):
    out = []
    for t, cands in enumerate(frames):
        tracker.predict(t, None if affines is None else affines[t])
        out.append(tracker.update(cands))
    return out


def test_no_track_needs_init_conf_then_follows_the_ball():
    tr = BallTracker(TrackerConfig(init_conf=0.15))
    out = run(tr, [[C(100, 100, 0.10)], [C(110, 100, 0.40)], [C(120, 100, 0.06)]])
    assert out[0] is None                                  # too weak to start a track
    assert out[1].source == "main" and out[1].track_id == 1
    assert out[2].source == "main"                         # weak, but inside the gate: accepted
    assert out[2].x == pytest.approx(120, abs=3)


def test_gate_rejects_far_candidates_and_coasts_then_drops():
    cfg = TrackerConfig(gate_a=20, gate_b=10, max_coast_frames=2, restart_conf=0.99)
    tr = BallTracker(cfg)
    out = run(tr, [[C(100, 100, 0.9)], [C(400, 400, 0.3)], [], [], []])
    assert out[1].source == "predicted" and out[1].frames_since_seen == 1
    assert out[2].source == "predicted" and out[2].frames_since_seen == 2
    assert out[3] is None and not tr.active                # coasted max_coast_frames, then dropped
    assert tr.stats["drops"] == 1


def test_gate_grows_with_the_gap():
    cfg = TrackerConfig(gate_a=20, gate_b=30, max_coast_frames=5, restart_conf=0.99)
    tr = BallTracker(cfg)
    out = run(tr, [[C(100, 100, 0.9)], [], [C(175, 100, 0.3)]])
    assert out[1].source == "predicted"
    assert out[2].source == "main"                         # r(2) = 80 >= 75


def test_cost_prefers_near_and_confident_and_main_on_ties():
    tr = BallTracker(TrackerConfig(lam=0.5))
    run(tr, [[C(100, 100, 0.9)]])
    tr.predict(1)
    est = tr.update([C(130, 100, 0.9), C(102, 100, 0.3)])
    assert est.x == pytest.approx(102, abs=3)              # near beats confident-but-far here
    tr2 = BallTracker(TrackerConfig())
    run(tr2, [[C(100, 100, 0.9)]])
    tr2.predict(1)
    assert tr2.update([C(105, 100, 0.5, "specialist"), C(105, 100, 0.5, "main")]).source == "main"


def test_floors_per_source_and_spec_scale():
    cfg = TrackerConfig(min_conf_spec=0.4, spec_scale=0.5, init_conf=0.3)
    assert run(BallTracker(cfg), [[C(100, 100, 0.35, "specialist")]])[0] is None   # under its floor
    assert run(BallTracker(cfg), [[C(100, 100, 0.5, "specialist")]])[0] is None    # 0.5 * 0.5 < init
    assert run(BallTracker(cfg), [[C(100, 100, 0.7, "specialist")]])[0].source == "specialist"


def test_restart_needs_consecutive_strong_frames_outside_the_gate():
    cfg = TrackerConfig(gate_a=20, gate_b=5, restart_conf=0.5, restart_frames=2, max_coast_frames=10)
    tr = BallTracker(cfg)
    out = run(tr, [[C(100, 100, 0.9)], [C(900, 500, 0.8)], [C(905, 502, 0.8)]])
    assert out[1].source == "predicted"                    # one strong far frame isn't enough
    assert out[2].source == "main" and out[2].track_id == 2 and tr.stats["restarts"] == 1
    tr = BallTracker(cfg)
    out = run(tr, [[C(100, 100, 0.9)], [C(900, 500, 0.8)], [], [C(905, 502, 0.8)]])
    assert out[3].source == "predicted"                    # not consecutive: no restart


def test_velocity_is_learned_and_camera_motion_shifts_the_state():
    tr = BallTracker(TrackerConfig(r_main=1.0))
    run(tr, [[C(100, 100, 0.9)], [C(110, 100, 0.9)], [C(120, 100, 0.9)], [C(130, 100, 0.9)]])
    assert tr.predict(4)[0] == pytest.approx(140, abs=4)   # constant velocity ~10 px/frame
    shift = np.array([[1, 0, -50.0], [0, 1, 0]])           # camera pans: everything moves 50 px left
    p2 = BallTracker(TrackerConfig())
    run(p2, [[C(100, 100, 0.9)]])
    assert p2.predict(1, shift)[0] == pytest.approx(50, abs=1)


def test_predict_before_update_is_required():
    with pytest.raises(RuntimeError):
        BallTracker().update([])


def test_best_path_skips_an_outlier_and_bridges_a_gap():
    frames = [[C(100 + 10 * t, 100, 0.8)] for t in range(10)]
    frames[4] = [C(1500, 900, 0.3)]                         # off-path false positive
    frames[5] = []                                          # missed
    picks = best_path(frames, cfg=PathConfig(start_pen=0.5))
    assert picks[4] is None and picks[5] is None
    assert all(picks[t] is not None for t in (0, 1, 2, 3, 6, 7, 8, 9))
    filled = interpolate(picks, max_interp_frames=5)
    assert filled[4][0] == pytest.approx(140, abs=1) and filled[5][0] == pytest.approx(150, abs=1)


def test_best_path_prefers_the_consistent_candidate_among_many():
    frames = [[C(100 + 10 * t, 100, 0.4), C(800, 50 + 37 * t, 0.45)] for t in range(8)]
    picks = best_path(frames)
    assert all(p is not None and p.y == 100 for p in picks)   # the smooth track, not the jumpy one


def test_interpolate_respects_max_gap_and_camera_motion():
    picks = [C(0, 0, 0.9)] + [None] * 3 + [C(40, 0, 0.9)]
    assert interpolate(picks, max_interp_frames=2)[2] is None
    shift = [None] + [np.array([[1, 0, 10.0], [0, 1, 0]])] * 4    # camera adds +10 px per frame
    # In frame-0 coordinates the end is at 40 - 40 = 0: the ball didn't move, the camera did.
    filled = interpolate(picks, shift, max_interp_frames=5)
    assert [round(p[0]) for p in filled] == [0, 10, 20, 30, 40]
