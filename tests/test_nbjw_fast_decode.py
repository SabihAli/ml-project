"""nbjw_fast_decode must reproduce the vendored NBJW decode exactly (OPTIMIZATION_PLAN.md
Phase 1). Synthetic heatmaps with NBJW's real shapes; quantised values force ties between
neighbouring pixels, the case where a pooling change would show first."""

import numpy as np
import pytest
import torch

from gsr_pipeline import nbjw_fast_decode as fast
from gsr_pipeline.calibrate import KP_THRESHOLD, LINE_THRESHOLD, NBJW_INPUT_HW
from gsr_pipeline.nbjw_calib.utils import utils_heatmap as ref

HM = (135, 240)


def _heatmaps(seed, channels, quantise):
    rng = np.random.default_rng(seed)
    hm = rng.random((1, channels, *HM)).astype(np.float32) * 0.3
    for c in range(channels):         # a few gaussian peaks per channel, some at the border
        for _ in range(rng.integers(0, 4)):
            y, x = rng.integers(0, HM[0]), rng.integers(0, HM[1])
            yy, xx = np.ogrid[:HM[0], :HM[1]]
            hm[0, c] += rng.random() * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 8.0).astype(np.float32)
    if quantise:
        hm = np.round(hm * 20) / 20
    return torch.from_numpy(hm.astype(np.float32))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("quantise", [False, True])
def test_decode_matches_reference(seed, quantise):
    hm_kp = _heatmaps(seed, 58, quantise)
    hm_l = _heatmaps(seed + 100, 24, quantise)
    h, w = NBJW_INPUT_HW

    r_kp = ref.coords_to_dict(ref.get_keypoints_from_heatmap_batch_maxpool(hm_kp[:, :-1]), threshold=KP_THRESHOLD)
    r_l = ref.coords_to_dict(ref.get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1]), threshold=LINE_THRESHOLD)
    f_kp = fast.coords_to_dict(fast.peaks_kp(hm_kp[:, :-1]), threshold=KP_THRESHOLD)
    f_l = fast.coords_to_dict(fast.peaks_lines(hm_l[:, :-1]), threshold=LINE_THRESHOLD)
    assert f_kp == r_kp
    assert f_l == r_l

    r_done = ref.complete_keypoints(r_kp, r_l, w=w, h=h, normalize=True)
    f_done = fast.complete_keypoints(f_kp, f_l, w=w, h=h, normalize=True)
    assert list(f_done[0]) == list(r_done[0])       # same keys, same order
    assert f_done == r_done


def test_completion_matches_reference_on_dense_lines():
    """Every line present: every intersection and auxiliary-pair branch runs."""
    rng = np.random.default_rng(7)
    h, w = NBJW_INPUT_HW
    for _ in range(50):
        lines = {k: {"x_1": float(rng.integers(0, w)), "y_1": float(rng.integers(0, h)), "p_1": 0.9,
                     "x_2": float(rng.integers(0, w)), "y_2": float(rng.integers(0, h)), "p_2": 0.9}
                 for k in range(1, 24)}
        kps = {k: {"x": float(rng.integers(0, w)), "y": float(rng.integers(0, h)), "p": 0.5}
               for k in rng.choice(np.arange(1, 58), size=10, replace=False).tolist()}
        assert fast.complete_keypoints([kps], [lines], w, h, True) == \
            ref.complete_keypoints([kps], [lines], w, h, True)


def test_separable_pool_is_exact():
    x = _heatmaps(3, 8, quantise=True)
    for k, pad in ((31, 0), (21, 10)):
        assert torch.equal(fast._separable_max_pool(x, k, pad),
                           torch.nn.functional.max_pool2d(x, k, stride=1, padding=pad))
