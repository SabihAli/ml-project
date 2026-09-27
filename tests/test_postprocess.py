"""
Letterbox and post-processing tests (IMPLEMENTATION_PLAN.md sections 3.3, 12).

Post-processing is where a Path B bug would be quietest: wrong box arithmetic still yields
plausible-looking boxes in roughly the right places, and NMS mistakes only show up as an
occasional duplicate or a missing ball. So the geometry is checked against hand-computed
values rather than against a golden file that could bake in the same error.

No weights, no GPU, no data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gsr_pipeline.runtime.postprocess import (
    PAD_VALUE,
    letterbox_params,
    postprocess,
    preprocess,
)

BROADCAST = (1080, 1920)
MODEL_HW = (736, 1280)


def _raw(boxes_xywh, scores_per_class):
    """Build a (1, 4 + nc, anchors) head output from explicit boxes and class scores."""
    n = len(boxes_xywh)
    nc = len(scores_per_class[0])
    out = torch.zeros((1, 4 + nc, n), dtype=torch.float32)
    out[0, :4, :] = torch.tensor(boxes_xywh, dtype=torch.float32).T
    out[0, 4:, :] = torch.tensor(scores_per_class, dtype=torch.float32).T
    return out


# ------------------------------------------------------------------------- letterbox


def test_letterbox_1080p_into_the_rectangular_input():
    """1920x1080 -> 1280x736: the frame becomes 1280x720, leaving 8px top and bottom.

    This is the whole point of the rectangular engine (plan 3.4) -- a 1280x1280 square
    would pad 280px per side instead, ~44% wasted compute.
    """
    p = letterbox_params(1080, 1920, *MODEL_HW)
    assert p.scale == pytest.approx(1280 / 1920)
    assert round(1080 * p.scale) == 720
    assert p.pad_y == pytest.approx(8.0)
    assert p.pad_x == pytest.approx(0.0)


def test_letterbox_square_input_pads_the_height():
    p = letterbox_params(1080, 1920, 1280, 1280)
    assert p.pad_x == pytest.approx(0.0)
    assert p.pad_y == pytest.approx(280.0)


def test_preprocess_shape_padding_and_channel_order():
    frame = np.zeros((*BROADCAST, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # pure blue in BGR
    x, p = preprocess(frame, *MODEL_HW)

    assert x.shape == (1, 3, 736, 1280)
    # BGR -> RGB: blue must land in channel 2, not channel 0.
    assert x[0, 2, 400, 640].item() == pytest.approx(1.0)
    assert x[0, 0, 400, 640].item() == pytest.approx(0.0)
    # The padding rows carry Ultralytics' 114 grey, not black -- black would look like a
    # dark region to the model rather than like absent image.
    assert x[0, :, 0, 640].mean().item() == pytest.approx(PAD_VALUE, abs=1e-3)
    assert x[0, :, 735, 640].mean().item() == pytest.approx(PAD_VALUE, abs=1e-3)


def test_preprocess_writes_into_a_provided_buffer():
    """The live path reuses one input buffer per stream (plan principle 4)."""
    frame = np.zeros((*BROADCAST, 3), dtype=np.uint8)
    buf = torch.zeros((1, 3, 736, 1280), dtype=torch.float32)
    x, _ = preprocess(frame, *MODEL_HW, out=buf)
    assert x.data_ptr() == buf.data_ptr()


# ----------------------------------------------------------------------- postprocess


def test_boxes_map_back_to_frame_pixels():
    """A box at the centre of the model input must land at the centre of the frame."""
    p = letterbox_params(1080, 1920, *MODEL_HW)
    raw = _raw([(640.0, 368.0, 64.0, 128.0)], [(0.9, 0.0)])
    boxes, scores, classes = postprocess(raw, p, {0: 0.25, 1: 0.15}, iou=0.7, max_det=300)

    assert len(boxes) == 1
    l, t, r, b = boxes[0].tolist()
    assert (l + r) / 2 == pytest.approx(960.0, abs=0.5)   # 1920 / 2
    assert (t + b) / 2 == pytest.approx(540.0, abs=0.5)   # 1080 / 2
    # 64px wide at scale 0.6667 -> 96px in the frame.
    assert r - l == pytest.approx(96.0, abs=0.5)
    assert scores[0].item() == pytest.approx(0.9)
    assert classes[0].item() == 0


def test_boxes_are_clamped_to_the_frame():
    p = letterbox_params(1080, 1920, *MODEL_HW)
    raw = _raw([(10.0, 10.0, 200.0, 200.0)], [(0.9, 0.0)])
    boxes, _, _ = postprocess(raw, p, {0: 0.25, 1: 0.15}, iou=0.7, max_det=300)
    l, t, _, _ = boxes[0].tolist()
    assert l >= 0.0 and t >= 0.0


def test_per_class_thresholds_are_applied_independently():
    """The ball's floor is deliberately lower than the athlete's. A single shared threshold
    would drop exactly the detections Path B exists to recover (blockers doc 2)."""
    p = letterbox_params(1080, 1920, *MODEL_HW)
    raw = _raw(
        [(100.0, 100.0, 20.0, 20.0), (300.0, 300.0, 20.0, 20.0)],
        [(0.20, 0.0), (0.0, 0.20)],  # athlete @0.20 (below 0.25), ball @0.20 (above 0.15)
    )
    boxes, _, classes = postprocess(raw, p, {0: 0.25, 1: 0.15}, iou=0.7, max_det=300)
    assert len(boxes) == 1
    assert classes[0].item() == 1


def test_classes_absent_from_the_threshold_map_are_dropped():
    p = letterbox_params(1080, 1920, *MODEL_HW)
    raw = _raw([(100.0, 100.0, 20.0, 20.0)], [(0.0, 0.99)])
    boxes, _, _ = postprocess(raw, p, {0: 0.25}, iou=0.7, max_det=300)
    assert len(boxes) == 0


def test_nms_suppresses_duplicates_of_the_same_class():
    p = letterbox_params(1080, 1920, *MODEL_HW)
    raw = _raw(
        [(200.0, 200.0, 60.0, 120.0), (203.0, 202.0, 60.0, 120.0)],  # near-identical
        [(0.9, 0.0), (0.8, 0.0)],
    )
    boxes, scores, _ = postprocess(raw, p, {0: 0.25, 1: 0.15}, iou=0.7, max_det=300)
    assert len(boxes) == 1
    assert scores[0].item() == pytest.approx(0.9)  # the higher-scoring one survives


def test_nms_is_class_aware_so_a_ball_on_a_player_survives():
    """Class-agnostic NMS would delete the ball whenever it overlaps a player -- i.e.
    exactly when the ball matters most."""
    p = letterbox_params(1080, 1920, *MODEL_HW)
    raw = _raw(
        [(200.0, 200.0, 60.0, 120.0), (200.0, 200.0, 58.0, 118.0)],
        [(0.9, 0.0), (0.0, 0.5)],  # an athlete and a ball at the same place
    )
    boxes, _, classes = postprocess(raw, p, {0: 0.25, 1: 0.15}, iou=0.7, max_det=300)
    assert sorted(classes.tolist()) == [0, 1]


def test_max_det_caps_the_output():
    p = letterbox_params(1080, 1920, *MODEL_HW)
    n = 50
    raw = _raw(
        [(50.0 * i, 50.0 * i, 10.0, 10.0) for i in range(1, n + 1)],
        [(0.9, 0.0)] * n,
    )
    boxes, _, _ = postprocess(raw, p, {0: 0.25, 1: 0.15}, iou=0.7, max_det=5)
    assert len(boxes) == 5


def test_no_detections_returns_correctly_shaped_empties():
    """Downstream stages index these arrays, so an empty frame must not be a special case."""
    p = letterbox_params(1080, 1920, *MODEL_HW)
    raw = _raw([(100.0, 100.0, 20.0, 20.0)], [(0.01, 0.01)])
    boxes, scores, classes = postprocess(raw, p, {0: 0.25, 1: 0.15}, iou=0.7, max_det=300)
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert classes.shape == (0,)


def test_rejects_a_wrongly_shaped_head_output():
    p = letterbox_params(1080, 1920, *MODEL_HW)
    with pytest.raises(ValueError, match="expected"):
        postprocess(torch.zeros((6, 100)), p, {0: 0.25}, iou=0.7, max_det=300)
