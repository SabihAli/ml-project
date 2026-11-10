"""
ball_gate.py -- plausibility check on the main detector's ball candidates
(docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md §3.1, Phase 1 task 1).

Without the gate, BallFuser accepts the detector's top ball unconditionally. Phase 0 found that
a wrong detector ball is the largest miss cause (10.2% of test ball frames), and that a wrong
sighting then misplaces the specialist window (another 7.7%). The gate accepts the first
candidate (highest confidence first) that is plausible given the last sighting:

  * there is no last sighting, or it is more than `max_miss` frames old  -> accept the top one;
  * conf >= conf_hi                                                      -> accept;
  * distance to the last sighting <= r0 + speed * gap                    -> accept.

If no candidate passes, the frame is treated as a detector miss (the specialist runs at the
last sighting). After `max_rejects` consecutive rejected frames the top candidate is accepted
anyway, so a real fast change of position (long pass, cut) cannot lock the ball out.

Pure Python, no dependencies: imported by gsr_pipeline/ball_specialist.py (runtime) and
scripts/pipeline_improvement_1/phase1/phase1_sweep.py (evaluation), so both run the same rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Sequence


@dataclass(frozen=True)
class GateParams:
    enabled: bool = False      # False = accept the top candidate (the pre-Phase-1 behaviour)
    r0: float = 30.0           # px allowed at any gap (detector jitter, box-centre noise)
    speed: float = 60.0        # px per frame of gap (ball p99 ~53 px at 1 frame, 1920x1080)
    conf_hi: float = 1.01      # a candidate at or above this is accepted regardless (>1 = never)
    max_rejects: int = 10      # consecutive rejected frames before the top candidate is forced

    def as_dict(self) -> dict:
        return asdict(self)

    def tag(self) -> str:
        if not self.enabled:
            return "off"
        return f"r{self.r0:g}_v{self.speed:g}_c{self.conf_hi:g}_k{self.max_rejects}"


class BallGate:
    """Per-stream state (the consecutive-reject counter). Create one per stream or call reset."""

    def __init__(self, params: Optional[GateParams] = None, max_miss: int = 15) -> None:
        self.p = params or GateParams()
        self.max_miss = max_miss
        self.reset()

    def reset(self) -> None:
        self.rejects = 0

    def pick(self, cands: Sequence[tuple], t: int, last: Optional[tuple]) -> Optional[int]:
        """cands: (x, y, conf, ...) sorted by conf, highest first, already at the output conf.
        last: (x, y, frame) of the last sighting or None. Returns the accepted index or None."""
        if not cands:
            return None
        p = self.p
        if not p.enabled or last is None or t - last[2] > self.max_miss:
            self.rejects = 0
            return 0
        allowed = p.r0 + p.speed * max(t - last[2], 1)
        for i, c in enumerate(cands):
            if c[2] >= p.conf_hi or math.hypot(c[0] - last[0], c[1] - last[1]) <= allowed:
                self.rejects = 0
                return i
        self.rejects += 1
        if self.rejects > p.max_rejects:
            self.rejects = 0
            return 0
        return None
