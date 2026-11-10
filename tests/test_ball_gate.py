"""The detector-ball plausibility gate (gsr_pipeline/ball_gate.py) and its wiring into BallFuser.
The same BallGate drives scripts/pipeline_improvement_1/phase1/phase1_sweep.py, so these pin the
rule the sweep measured."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gsr_pipeline.ball_gate import BallGate, GateParams  # noqa: E402
from gsr_pipeline.ball_specialist import BallFuser  # noqa: E402
from test_ball_specialist import FakeSpecialist, balls, frame  # noqa: E402

ON = GateParams(enabled=True, r0=30.0, speed=50.0, conf_hi=1.01, max_rejects=2)


def test_disabled_gate_accepts_top_candidate():
    g = BallGate(GateParams())
    assert g.pick([(0, 0, 0.9), (1000, 1000, 0.2)], 5, (1000, 1000, 4)) == 0


def test_no_prior_or_old_prior_accepts_top():
    g = BallGate(ON, max_miss=15)
    assert g.pick([(0, 0, 0.9)], 5, None) == 0
    assert g.pick([(0, 0, 0.9)], 30, (1000, 1000, 10)) == 0         # last sighting 20 frames old


def test_far_candidate_rejected_second_candidate_accepted():
    g = BallGate(ON)
    last = (500, 500, 9)
    assert g.pick([(1500, 500, 0.9)], 10, last) is None              # 1000 px > 30 + 50
    assert g.pick([(1500, 500, 0.9), (540, 500, 0.3)], 10, last) == 1


def test_allowance_grows_with_gap():
    g = BallGate(ON)
    assert g.pick([(620, 500, 0.9)], 12, (500, 500, 10)) == 0        # 120 <= 30 + 50 * 2


def test_high_confidence_bypasses_distance():
    g = BallGate(GateParams(enabled=True, conf_hi=0.8))
    assert g.pick([(1500, 500, 0.85)], 10, (500, 500, 9)) == 0


def test_forced_accept_after_max_rejects():
    g = BallGate(ON)
    far = [(1500, 500, 0.9)]
    assert g.pick(far, 10, (500, 500, 9)) is None
    assert g.pick(far, 11, (500, 500, 9)) is None
    assert g.pick(far, 12, (500, 500, 9)) == 0                       # third reject > max_rejects=2


def test_fuser_gate_rejection_runs_specialist_at_last_sighting():
    spec = FakeSpecialist([(320.0, 320.0, 0.9)])
    fuser = BallFuser(spec, gate=ON)
    fuser.update(0, frame(0), balls((950, 530, 970, 550, 0.8)))      # centre (960, 540)
    out = fuser.update(1, frame(1), balls((1800, 100, 1810, 110, 0.9)))  # implausible jump
    assert out["ball_source"] == "specialist"
    assert len(spec.calls) == 1 and fuser.stats["gate_rejects"] == 1


def test_fuser_without_gate_is_unchanged():
    spec = FakeSpecialist()
    fuser = BallFuser(spec)
    fuser.update(0, frame(0), balls((950, 530, 970, 550, 0.8)))
    out = fuser.update(1, frame(1), balls((1800, 100, 1810, 110, 0.9)))
    assert out["ball_source"] == "main" and spec.calls == []
