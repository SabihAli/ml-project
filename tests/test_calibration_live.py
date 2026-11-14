"""Camera-motion estimation and homography propagation (plan section 7), on synthetic
frames with known camera motion. No weights, no GPU: NBJW is replaced by an oracle that
returns the true homography, so the tests isolate the propagation logic."""

import cv2
import numpy as np
import pytest

from gsr_pipeline.calibration_live import H_FRESH, H_PROPAGATED, LiveCalibrator
from gsr_pipeline.camera_motion import CameraMotionEstimator

W, H_PX = 960, 540


def _texture(seed=0, size=(1400, 2400)):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (size[0] // 8, size[1] // 8, 3), dtype=np.uint8),
                     (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    for _ in range(300):    # corners for the tracker
        x, y = rng.integers(0, size[1]), rng.integers(0, size[0])
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 9)), colour, -1)
    return img


def _camera(t):
    """World (texture) px -> image px at frame t: a slow pan with a little zoom."""
    s = 1.0 + 0.004 * t
    tx, ty = -300 - 6.0 * t, -250 - 1.5 * t
    return np.array([[s, 0, tx * s], [0, s, ty * s], [0, 0, 1.0]])


def _frames(n):
    tex = _texture()
    return [cv2.warpPerspective(tex, _camera(t), (W, H_PX)) for t in range(n)]


# A fixed, arbitrary world-px -> pitch-m map, so the true image->pitch H at t is
# WORLD_TO_PITCH . camera(t)^-1.
WORLD_TO_PITCH = np.array([[0.05, 0.002, -60.0], [0.001, 0.06, -40.0], [0.0, 1e-5, 1.0]])


def _true_h(t):
    return WORLD_TO_PITCH @ np.linalg.inv(_camera(t))


def _pitch_err(h_a, h_b):
    xs, ys = np.meshgrid(np.linspace(0, W - 1, 12), np.linspace(H_PX / 3, H_PX - 1, 8))
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    pa, pb = h_a @ pts, h_b @ pts
    return float(np.abs(pa[:2] / pa[2] - pb[:2] / pb[2]).max())


def test_motion_estimate_matches_true_camera_motion():
    frames = _frames(3)
    est = CameraMotionEstimator(work_width=480)
    est.update(frames[0])
    m = est.update(frames[1])
    true_m = _camera(1) @ np.linalg.inv(_camera(0))
    assert m.M is not None and not m.scene_cut
    assert m.inlier_ratio > 0.8
    corners = np.array([[0, 0, 1], [W, 0, 1], [0, H_PX, 1], [W, H_PX, 1]], float).T
    a, b = m.M @ corners, true_m @ corners
    assert np.abs(a[:2] / a[2] - b[:2] / b[2]).max() < 1.0      # < 1 px
    assert m.scale == pytest.approx(1.004, abs=2e-3)


def test_scene_cut_detected():
    est = CameraMotionEstimator(work_width=480)
    est.update(_frames(1)[0])
    other = np.full((H_PX, W, 3), 255, np.uint8)
    other[:, : W // 2] = 0
    assert est.update(other).scene_cut


def _oracle_calibrator(**kw):
    """LiveCalibrator whose NBJW returns the true H of the frame it was handed."""
    calib = LiveCalibrator(kp_detector=None, image_width=W, image_height=H_PX, **kw)
    calls = []
    by_id = {}

    def run_nbjw(frame_bgr, *_):
        t = by_id[frame_bgr.tobytes()[:4096]]
        calls.append(t)
        return _true_h(t), {}, {}, {}

    calib._run_nbjw = run_nbjw
    return calib, calls, by_id


def test_inline_every_k_propagates_within_tolerance():
    n, k = 21, 5
    frames = _frames(n)
    calib, calls, by_id = _oracle_calibrator(calib_every_k=k, mode="inline", motion={"work_width": 480})
    worst = 0.0
    for t, frame in enumerate(frames):
        by_id[frame.tobytes()[:4096]] = t
        res = calib.update(t, frame)
        assert res.H is not None
        assert res.h_source == (H_FRESH if t % k == 0 else H_PROPAGATED), (t, res.trigger)
        assert res.h_age_frames == t % k
        worst = max(worst, _pitch_err(res.H, _true_h(t)))
    assert calls == list(range(0, n, k))
    assert worst < 0.25         # metres, after 4 propagated frames


def test_k1_inline_runs_nbjw_every_frame_without_motion():
    calib, calls, by_id = _oracle_calibrator(calib_every_k=1, mode="inline")
    for t, frame in enumerate(_frames(4)):
        by_id[frame.tobytes()[:4096]] = t
        assert calib.update(t, frame).h_source == H_FRESH
    assert calls == [0, 1, 2, 3] and calib.motion is None


def test_failed_nbjw_falls_back_to_propagation():
    frames = _frames(8)
    calib, _, by_id = _oracle_calibrator(calib_every_k=3, mode="inline", motion={"work_width": 480})
    good = calib._run_nbjw
    for t, frame in enumerate(frames):
        by_id[frame.tobytes()[:4096]] = t
        calib._run_nbjw = (lambda f: (None, {}, {}, {})) if t == 3 else good
        res = calib.update(t, frame)
        if t == 3:      # NBJW failed: the frame-0 anchor is carried forward by motion
            assert res.h_source == H_PROPAGATED and res.h_age_frames == 3
            assert _pitch_err(res.H, _true_h(t)) < 0.25


@pytest.mark.parametrize("interval, max_calls", [(1, 20), (3, 8)])
def test_failing_nbjw_on_textureless_shot_is_rate_limited(interval, max_calls):
    """A close-up/replay: motion is not estimable (no corners) and NBJW finds no pitch. Without
    the limit (interval=1, the old behaviour) NBJW ran on every frame."""
    flat = np.full((H_PX, W, 3), 90, np.uint8)
    frames = [_frames(1)[0]] + [flat] * 19
    calib, calls, by_id = _oracle_calibrator(calib_every_k=5, mode="inline", motion={"work_width": 480},
                                             min_trigger_interval_frames=interval)
    good = calib._run_nbjw
    for t, frame in enumerate(frames):
        by_id[frame.tobytes()[:4096]] = t
        calib._run_nbjw = good if t == 0 else (lambda f, t=t: (calls.append(t), (None, {}, {}, {}))[1])
        res = calib.update(t, frame)
        assert res.H is not None            # the last good H is still served
    assert calls[:2] == [0, 1]              # first, then the scene cut (never limited)
    assert len(calls) == max_calls, calls
    if interval > 1:
        assert all(b - a >= interval for a, b in zip(calls[1:], calls[2:])), calls


def test_async_never_blocks_after_bootstrap_and_reanchors():
    frames = _frames(16)
    calib, calls, by_id = _oracle_calibrator(calib_every_k=4, mode="async", motion={"work_width": 480})
    try:
        sources = []
        for t, frame in enumerate(frames):
            by_id[frame.tobytes()[:4096]] = t
            res = calib.update(t, frame)
            assert res.H is not None
            sources.append(res.h_source)
            assert _pitch_err(res.H, _true_h(t)) < 0.3
        assert sources[0] == H_FRESH
        assert len(calls) >= 3          # re-anchored repeatedly in the background
    finally:
        calib.close()


class _NetStub:
    def __init__(self, name, log, calib):
        self.name, self.log, self.calib = name, log, calib

    def infer(self, x):
        self.log.append((self.name, self.calib._tick))
        return x


class _GatedKpStub:
    """Has the attributes the async gated path uses; networks record the frame tick they ran on."""

    def __init__(self, log, calib):
        self.net_kp, self.net_l = _NetStub("kp", log, calib), _NetStub("l", log, calib)

    def preprocess(self, frame):
        return frame

    def detect(self, frame):                             # the ungated bootstrap run
        self.net_kp.log.append(("bootstrap", self.net_kp.calib._tick))
        return {}, {}

    def decode(self, hm_kp, hm_l):
        return {}


def test_async_gating_runs_one_network_per_frame_tick(monkeypatch):
    """Async NBJW launches its two networks on consecutive update() ticks, the first one on the
    submitting frame's tick, and close() never deadlocks on a worker waiting for a tick."""
    import time

    import torch
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: type("S", (), {"synchronize": lambda self: None})())
    log = []
    calib = LiveCalibrator(kp_detector=None, image_width=W, image_height=H_PX, calib_every_k=5, mode="async",
                           motion={"work_width": 480})
    calib._cuda_stream = None
    calib.kp_detector = _GatedKpStub(log, calib)
    calib.homography.compute = lambda kps: (np.eye(3), {})
    frames = _frames(12)
    for t, frame in enumerate(frames):
        calib.update(t, frame)
        time.sleep(0.02)                                 # let the worker reach its next wait
    calib.close()
    # Bootstrap (frame 0) is ungated; afterwards every run's kp/l land on ticks n, n+1.
    assert log[0][0] == "bootstrap"
    gated = log[1:]
    assert len(gated) >= 4, log
    for (a, ta), (b, tb) in zip(gated[::2], gated[1::2]):
        assert (a, b) == ("kp", "l") and tb == ta + 1, log
    assert calib._closing


def test_async_close_releases_a_worker_waiting_for_a_tick():
    import threading
    calib = LiveCalibrator(kp_detector=None, image_width=W, image_height=H_PX, calib_every_k=5, mode="async")
    done = threading.Event()
    threading.Thread(target=lambda: (calib._wait_tick(calib._tick + 5), done.set()), daemon=True).start()
    calib.close()
    assert done.wait(2.0)
