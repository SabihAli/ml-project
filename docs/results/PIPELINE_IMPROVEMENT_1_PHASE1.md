# Pipeline Improvement Plan 1 — Phase 1 results (2026-11-10)

Plan: `docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md` Phase 1, tasks 1-2. Run on Kaggle T4 (`kaggle-t4`)
with `notebooks/pipeline_improvement_1/phase1_gate_threshold.ipynb`
(`scripts/pipeline_improvement_1/phase1/phase1_sweep.py`). Raw outputs (gitignored):
`data/outputs/pipeline_improvement_1/phase1/`. Baseline: `PIPELINE_IMPROVEMENT_1_PHASE0.md`.

## What changed

- **Plausibility gate** (`gsr_pipeline/ball_gate.py`, used by `BallFuser.update` and by the sweep). The
  detector's candidates are tried in confidence order; the first one within `r0 + speed × gap` px of
  the last sighting, or at conf >= `conf_hi`, is accepted. None accepted = treated as a detector miss,
  so c5c runs at the last sighting. After `max_rejects` consecutive rejections the top ball is forced.
- **c5c threshold** 0.70 -> 0.45.
- Both chosen on `select` (precision floor 87.58% = baseline − 1 pp) and frozen for `test`. Now the
  defaults in `gsr_pipeline/configs/default.yaml` (`ball_specialist.threshold`, `ball_specialist.gate`).

## Test (15 sequences, 10,178 ball frames, 15 px)

| config | gate | thr | recall | @10 px | @5 px | precision | w/o SNGS-050 | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | off | 0.70 | 69.44% | 65.85% | 56.46% | 86.49% | 68.80% | — |
| threshold only | off | 0.40 | 73.04% | 69.05% | 58.88% | 86.02% | 72.32% | keep |
| gate only | r30 v80 c0.6 k15 | 0.70 | 72.30% | 68.46% | 58.72% | 90.38% | 71.53% | keep |
| **gate + threshold** | **r30 v50 c0.6 k15** | **0.45** | **76.29%** | **72.09%** | **61.44%** | **90.21%** | **75.56%** | **keep, chosen** |

`select` for the chosen setting: 76.80% / 90.93%, so test tracks select closely.

## Miss breakdown, before -> after (test, % of ball frames)

| bucket | Phase 0 | Phase 1 |
|---|---|---|
| B7 main_false | 10.2 | 6.6 |
| B3 false_sighting | 7.7 | 5.5 |
| B5 below_threshold | 7.4 | 7.0 |
| B1 long_gap | 4.7 | 3.6 |
| B6 spec_wrong_place | 0.2 | 0.7 |
| B4 outside_window | 0.0 | 0.1 |
| B2 no_prior | 0.2 | 0.2 |
| **total misses** | 30.6 | 23.7 |

- Recall by size (fused): < 10 px 52.0 -> 59.8%, 10-15 px 69.2 -> 77.3%, 15-20 px 73.8 -> 78.8%, 20+ px 74.7 -> 81.4%.
- Specialist: 2,986 -> 3,485 calls (+17%); recall per call 39.7 -> 55.2%; ball inside the window 89.3 -> 96.4%.
- Gate cost: 340 top balls rejected on test, 24 of them on the ball.
- Continuity: miss runs 508 -> 441, p95 22 -> 17 frames; longest miss unchanged (248 frames, 9.9 s).

## Limits

- Replay with the Ultralytics checkpoint (imgsz 1280), not the runtime ONNX/TRT detector; the gate
  must be re-measured on the runtime detector (carried to Phase 5).
- conf_hi 0.6 and max_rejects 15 are at the edge of the searched grid; a wider grid may add a little.
- The chosen threshold (0.45) costs ~0 precision only together with the gate; do not lower the
  threshold without it (threshold-only loses 0.5 pp precision at 0.40).

## Next

Largest remaining buckets: B5 7.0% (Phase 3 specialist), B7 6.6% (Phase 2 detector), B3 5.5%,
B1 3.6% (Phase 4 tracker, also the 9.9 s misses). Milestone 2 (~80-83%) needs Phase 3 and/or 4.
