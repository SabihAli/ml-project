"""BallFuser's state machine and the specialist artifact contract, with a fake specialist (no
model, no GPU). The fuser must behave like the evaluator's fused chain
(scripts/eval/ball_det/eval_ball_temporal.py `simulate`), since that is what gate 3 measured."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gsr_pipeline.ball_specialist import BallFuser, SpecialistArtifact, clamp_window

W, H = 1920, 1080


class FakeSpecialist:
    """Returns queued (x_in_crop, y_in_crop, conf) answers and records every call."""

    crop = 640

    def __init__(self, answers=(), threshold=0.7):
        self.artifact = SimpleNamespace(threshold=threshold)
        self.answers = list(answers)
        self.calls = []

    def detect(self, frames, x0, y0):
        self.calls.append((frames, x0, y0))
        x, y, conf = self.answers.pop(0) if self.answers else (0.0, 0.0, 0.0)
        return x + x0, y + y0, conf


def frame(i):
    return np.full((H, W, 3), i % 256, np.uint8)


def balls(*boxes):
    """Main-detector ball rows from (l, t, r, b, conf)."""
    return pd.DataFrame([{"bbox_ltrb": (l, t, r, b), "bbox_conf": c} for l, t, r, b, c in boxes],
                        columns=["bbox_ltrb", "bbox_conf"])


NONE = balls()


def test_clamp_window_shifts_into_the_frame():
    assert clamp_window(960, 540, W, H, 640) == (640, 220)
    assert clamp_window(10, 10, W, H, 640) == (0, 0)                 # top-left corner
    assert clamp_window(1915, 1075, W, H, 640) == (W - 640, H - 640)  # bottom-right corner


def test_main_ball_wins_and_specialist_does_not_run():
    spec = FakeSpecialist()
    fuser = BallFuser(spec)
    out = fuser.update(0, frame(0), balls((100, 100, 110, 110, 0.3), (500, 500, 512, 512, 0.9)))
    assert out["ball_source"] == "main" and out["bbox_conf"] == 0.9   # top confidence
    assert out["bbox_ltrb"] == (500.0, 500.0, 512.0, 512.0)
    assert spec.calls == []


def test_no_sighting_yet_means_no_specialist_call():
    spec = FakeSpecialist()
    assert BallFuser(spec).update(0, frame(0), NONE) is None
    assert spec.calls == []


def test_miss_runs_specialist_on_window_at_last_sighting_with_three_frames():
    spec = FakeSpecialist([(320.0, 320.0, 0.9)])
    fuser = BallFuser(spec)
    f0, f1, f2 = frame(0), frame(1), frame(2)
    fuser.update(0, f0, balls((950, 530, 970, 550, 0.8)))            # centre (960, 540)
    fuser.update(1, f1, balls((950, 530, 970, 550, 0.8)))
    out = fuser.update(2, f2, NONE)
    frames, x0, y0 = spec.calls[0]
    assert (x0, y0) == clamp_window(960, 540, W, H, 640)
    assert frames[0] is f0 and frames[1] is f1 and frames[2] is f2  # t-2, t-1, t
    assert out["ball_source"] == "specialist" and out["frames_since_seen"] == 1
    assert out["bbox_ltwh"][2:] == (20.0, 20.0)                      # keeps the main box size
    cx = out["bbox_ltrb"][0] + 10
    assert cx == pytest.approx(x0 + 320.0)


def test_missing_previous_frames_repeat_frame_t():
    spec = FakeSpecialist([(0, 0, 0.1)])
    fuser = BallFuser(spec)
    f5 = frame(5)
    fuser.update(5, f5, balls((950, 530, 970, 550, 0.8)))
    f7 = frame(7)
    fuser.update(7, f7, NONE)                                         # frame 6 never arrived
    frames, *_ = spec.calls[0]
    assert frames[0] is f5                                            # t-2 = 5 is in the ring
    assert frames[1] is f7 and frames[2] is f7                        # t-1 = 6 is not -> t


def test_below_threshold_is_no_ball_and_keeps_the_old_sighting():
    spec = FakeSpecialist([(0, 0, 0.69), (0, 0, 0.1)])
    fuser = BallFuser(spec)                                           # sidecar threshold 0.7
    fuser.update(0, frame(0), balls((950, 530, 970, 550, 0.8)))
    assert fuser.update(1, frame(1), NONE) is None
    fuser.update(2, frame(2), NONE)
    assert spec.calls[1][1:] == spec.calls[0][1:]                     # same window: not re-centred


def test_specialist_hit_recentres_the_window_and_resets_the_gap():
    # 1st call: ball at crop (600, 300) -> new sighting far right of the old one.
    spec = FakeSpecialist([(600.0, 300.0, 0.9), (0, 0, 0.1)])
    fuser = BallFuser(spec)
    fuser.update(0, frame(0), balls((950, 530, 970, 550, 0.8)))
    hit = fuser.update(1, frame(1), NONE)
    fuser.update(2, frame(2), NONE)
    cx = hit["bbox_ltrb"][0] + 10
    cy = hit["bbox_ltrb"][1] + 10
    assert spec.calls[1][1:] == clamp_window(cx, cy, W, H, 640)
    assert fuser.stats == {"main": 1, "specialist_calls": 2, "specialist_hits": 1}


def test_specialist_stops_after_max_miss_frames():
    spec = FakeSpecialist()
    fuser = BallFuser(spec, max_miss_frames=3)
    fuser.update(0, frame(0), balls((950, 530, 970, 550, 0.8)))
    for t in range(1, 7):
        fuser.update(t, frame(t), NONE)
    assert len(spec.calls) == 3                                       # gaps 1, 2, 3 only


def test_explicit_threshold_overrides_sidecar_and_none_specialist_is_main_only():
    spec = FakeSpecialist([(0, 0, 0.5)])
    fuser = BallFuser(spec, threshold=0.4)
    fuser.update(0, frame(0), balls((950, 530, 970, 550, 0.8)))
    assert fuser.update(1, frame(1), NONE)["ball_source"] == "specialist"

    main_only = BallFuser(None)
    main_only.update(0, frame(0), balls((950, 530, 970, 550, 0.8)))
    assert main_only.update(1, frame(1), NONE) is None


def _write_artifact(tmp_path, **overrides):
    onnx = tmp_path / "spec.onnx"
    onnx.write_bytes(b"not really onnx")
    from gsr_pipeline.runtime.backend import sha256_file
    meta = {"model": "ball_temporal_heatmap", "input": {"shape": [1, 9, 640, 640]},
            "crop": 640, "stride": 2, "threshold": 0.7, "onnx_sha256": sha256_file(onnx)}
    meta.update(overrides)
    (tmp_path / "spec.onnx.json").write_text(json.dumps(meta))
    return onnx


def test_artifact_loads_and_names_its_engine(tmp_path):
    art = SpecialistArtifact.load(_write_artifact(tmp_path))
    assert (art.crop, art.stride, art.threshold) == (640, 2, 0.7)
    assert art.engine_path.name == "spec_fp16.engine"


@pytest.mark.parametrize("override, message", [
    ({"model": "something_else"}, "describes"),
    ({"input": {"shape": [1, 5, 640, 640]}}, "input shape"),
    ({"onnx_sha256": "0" * 64}, "checksum"),
])
def test_artifact_refusals(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        SpecialistArtifact.load(_write_artifact(tmp_path, **override))


def test_artifact_without_sidecar_is_refused(tmp_path):
    onnx = tmp_path / "spec.onnx"
    onnx.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="sidecar"):
        SpecialistArtifact.load(onnx)
