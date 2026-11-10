"""RobustBallTracker (gsr_pipeline/ball_tracker_robust.py): camera compensation, occlusion
following, cut restart, manoeuvre handling, coasting limits. No model, no GPU."""

import numpy as np

from gsr_pipeline.ball_tracker_robust import RobustBallTracker, RobustTrackerConfig

SHIFT = np.array([[1.0, 0, 40.0], [0, 1.0, 0]])     # camera pans: pixels move +40 px in x
STILL = np.eye(3)[:2]


def main(x, y, conf=0.8):
    return (x, y, conf, "main")


def run_line(tr, n=10, x0=500.0, vx=8.0, y=500.0):
    for t in range(n):
        tr.predict(t, None if t == 0 else STILL)
        tr.update([main(x0 + vx * t, y)])
    return x0 + vx * (n - 1)


def test_starts_and_follows_constant_motion():
    tr = RobustBallTracker()
    last = run_line(tr)
    x, y = tr.predict(10, STILL)
    assert abs(x - (last + 8.0)) < 3 and abs(y - 500) < 2


def test_camera_motion_moves_the_prediction():
    tr = RobustBallTracker()
    last = run_line(tr, vx=0.0)
    x, _ = tr.predict(10, SHIFT)
    assert abs(x - (last + 40.0)) < 2


def test_far_detection_rejected_then_restart_after_two_frames():
    tr = RobustBallTracker()
    run_line(tr, vx=0.0)
    tid = tr.track_id
    tr.predict(10, STILL)
    assert tr.update([main(1500, 200)]).source == "predicted"      # outside the gate: coast
    tr.predict(11, STILL)
    e = tr.update([main(1502, 200)])                                # 2nd consecutive strong frame
    assert e.source == "main" and tr.track_id == tid + 1 and abs(e.x - 1502) < 1


def test_camera_break_allows_one_frame_restart():
    tr = RobustBallTracker()
    run_line(tr, vx=0.0)
    tid = tr.track_id
    tr.predict(10, None)                                            # cut: no motion estimate
    e = tr.update([main(1500, 200, 0.4)])
    assert e.source == "main" and tr.track_id == tid + 1


def test_occluded_ball_follows_the_athlete():
    tr = RobustBallTracker(RobustTrackerConfig(max_coast_frames=2))
    run_line(tr, vx=0.0, x0=600.0, y=600.0)
    for k in range(1, 21):                                          # athlete walks right, ball hidden
        athlete = (600.0 + 5 * k, 590.0, 40.0, 90.0)
        tr.predict(9 + k, STILL, [athlete])
        e = tr.update([])
        assert e is not None and e.source == "predicted"            # past max_coast: occlusion keeps it
    assert e.occluded and abs(e.x - (600 + 100)) < 10


def test_free_coast_drops_after_limit():
    tr = RobustBallTracker(RobustTrackerConfig(max_coast_frames=3, occlusion=False))
    run_line(tr, vx=0.0)
    outs = []
    for k in range(1, 6):
        tr.predict(9 + k, STILL)
        outs.append(tr.update([]))
    assert [o is not None for o in outs] == [True, True, True, False, False]


def test_kick_is_reacquired_in_the_gate():
    tr = RobustBallTracker()
    run_line(tr, n=12, vx=2.0)                                      # slow roll ... then a kick
    x = 500 + 2 * 11
    hits = 0
    for k in range(1, 6):
        x += 25.0
        tr.predict(11 + k, STILL)
        e = tr.update([main(x, 500 - 6 * k)])
        hits += e.source == "main" and tr.track_id == 1
    assert hits >= 4                                                # the manoeuvre model keeps the track
