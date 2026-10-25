"""live_roles: label mapping, the live fit/vote, and the batch fit/vote, on synthetic kits
(painted boxes with known pitch positions; no model)."""

import numpy as np
import pandas as pd

from gsr_pipeline.live_roles import LiveAthleteRoles, assign_sequence_roles, split_label

W, H = 1920, 1080
BGR = {"red": (40, 40, 220), "blue": (220, 60, 40), "yellow": (40, 220, 230), "magenta": (200, 40, 200)}

# track id -> (kit, pitch x, pitch y). Red defends the lower-x end (its keeper is at x = -50).
PEOPLE = {
    **{i: ("red", -25 + 4 * i, -10 + 4 * i) for i in range(5)},
    **{10 + i: ("blue", 8 + 4 * i, -10 + 4 * i) for i in range(5)},
    20: ("yellow", 0.0, 0.0),        # referee: minor colour, mid-pitch
    21: ("magenta", -50.0, 2.0),     # keeper: minor colour, near the -x goal line
}


def make_frame(t, rng, tracks=PEOPLE, ball=True):
    img = np.zeros((H, W, 3), np.uint8)
    rows = []
    for k, (tid, (kit, px, py)) in enumerate(tracks.items()):
        l, top = 60 + 140 * k, 300
        colour = np.clip(np.array(BGR[kit]) + rng.integers(-6, 7, 3), 0, 255)
        img[top:top + 120, l:l + 50] = colour
        rows.append({"bbox_ltrb": (l, top, l + 50, top + 120), "bbox_ltwh": (l, top, 50, 120),
                     "bbox_conf": 0.9, "class_id": 0, "track_id": tid, "image_id": t,
                     "x_bottom_middle": px, "y_bottom_middle": py})
    if ball:
        rows.append({"bbox_ltrb": (900, 800, 912, 812), "bbox_ltwh": (900, 800, 12, 12), "bbox_conf": 0.6,
                     "class_id": 1, "track_id": 90_000, "image_id": t,
                     "x_bottom_middle": 1.0, "y_bottom_middle": 1.0})
    return img, pd.DataFrame(rows)


def expected(tid):
    return {"red": ("player", "left"), "blue": ("player", "right"), "yellow": ("referee", None),
            "magenta": ("goalkeeper", "left")}[PEOPLE[tid][0]]


def test_split_label():
    assert split_label("player_team_a") == ("player", "left")
    assert split_label("goalkeeper_team_b") == ("goalkeeper", "right")
    assert split_label("referee") == ("referee", None)
    assert split_label("other") == ("other", None)


def test_live_roles_before_fit_are_player_without_team_then_fit_and_vote():
    rng = np.random.default_rng(0)
    roles = LiveAthleteRoles(fit_frames=5)
    for t in range(4):
        img, df = make_frame(t, rng)
        out, pitch_view = roles.update(t, img, df)
        assert pitch_view
        athletes = out[out["class_id"] == 0]
        assert set(athletes["role_detection"]) == {"player"} and athletes["team"].isna().all()
        assert out.loc[out["class_id"] == 1, "role_detection"].tolist() == ["ball"]
    assert not roles.fitted

    for t in range(4, 8):
        img, df = make_frame(t, rng)
        out, _ = roles.update(t, img, df)
    assert roles.fitted
    for _, r in out[out["class_id"] == 0].iterrows():
        team = None if pd.isna(r["team"]) else r["team"]          # iterrows turns None into NaN
        assert (r["role_detection"], team) == expected(int(r["track_id"])), int(r["track_id"])
        assert r["role_confidence"] == 1.0
    assert roles.track_team(3) == ("left", 1.0)
    assert roles.track_team(999) == (None, None)


def test_failed_fit_retries_later():
    rng = np.random.default_rng(1)
    one_kit = {i: ("red", -20 + 4 * i, 0.0) for i in range(6)}           # one team only
    roles = LiveAthleteRoles(fit_frames=3, refit_every_frames=2)
    for t in range(3):
        roles.update(t, *make_frame(t, rng, one_kit, ball=False))
    assert not roles.fitted and roles.last_fit_error
    for t in range(3, 5):                                                  # both kits from now on
        roles.update(t, *make_frame(t, rng))
    assert roles.fitted


def test_rows_without_pitch_position_are_not_pitch_view_and_not_buffered():
    rng = np.random.default_rng(2)
    roles = LiveAthleteRoles(fit_frames=2)
    img, df = make_frame(0, rng)
    df[["x_bottom_middle", "y_bottom_middle"]] = np.nan                     # calibration failed
    out, pitch_view = roles.update(0, img, df)
    assert not pitch_view and not roles._buffer
    assert set(out.loc[out["class_id"] == 0, "role_detection"]) == {"player"}


def test_batch_assign_votes_per_tracklet():
    from gsr_pipeline.athlete_roles import torso_descriptor

    rng = np.random.default_rng(3)
    parts = []
    for t in range(10):
        img, df = make_frame(t, rng)
        df["torso_desc"] = [torso_descriptor(img, b) if c == 0 else None
                            for b, c in zip(df["bbox_ltrb"], df["class_id"])]
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    info = assign_sequence_roles(full, {}, fit_frames=5)
    assert info["fit_ok"]
    for tid in PEOPLE:
        rows = full[full["track_id"] == tid]
        assert set(zip(rows["role_detection"], rows["team"])) == {expected(tid)}, tid
    assert set(full.loc[full["class_id"] == 1, "role_detection"]) == {"ball"}
