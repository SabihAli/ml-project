"""ByteTracker must step on frames with no detections (OPTIMIZATION_PLAN.md 0.3): skipping
them froze lost tracks, so a gap never expired them."""

import pandas as pd
import pytest

pytest.importorskip("ultralytics")

from gsr_pipeline.track import ByteTracker  # noqa: E402

COLUMNS = ["bbox_ltwh", "bbox_ltrb", "bbox_conf", "class_id", "conf", "cls_id"]


def _df(boxes):
    return pd.DataFrame({
        "bbox_ltwh": [(l, t, r - l, b - t) for l, t, r, b in boxes], "bbox_ltrb": [tuple(b) for b in boxes],
        "bbox_conf": [0.9] * len(boxes), "class_id": [0] * len(boxes),
        "conf": [0.9] * len(boxes), "cls_id": [0] * len(boxes)})


def test_lost_tracks_expire_across_empty_frames():
    tracker = ByteTracker(track_buffer=3)
    for _ in range(3):
        out = tracker.update(_df([[100, 100, 150, 200]]), (1080, 1920))
    assert out["track_id"].notna().all()

    empty = pd.DataFrame(columns=COLUMNS)
    for _ in range(6):
        out = tracker.update(empty, (1080, 1920))
        assert out.empty and "track_id" in out
    assert not tracker._tracker.lost_stracks          # aged out, not frozen
    assert tracker._tracker.removed_stracks
