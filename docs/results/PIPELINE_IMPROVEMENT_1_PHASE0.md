# Pipeline Improvement Plan 1 — Phase 0 results (2026-11-09)

Plan: `docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md` Phase 0. Run on Kaggle T4 (`kaggle-t4`) with
`notebooks/pipeline_improvement_1/phase0_baseline.ipynb`. Raw outputs (gitignored):
`data/outputs/pipeline_improvement_1/phase0/` (full report: `reports/phase0/report_phase0.md`).

## Setup

| item | value |
|---|---|
| main detector | `runs/yolo26s_athlete_ball/best.pt` (sha256 `cdc88829…`), Ultralytics, imgsz 1280, FP16, ball conf >= 0.15 |
| specialist | `runs/ball_temporal/c5c/best.pt` (sha256 `d5f399c9…`), threshold 0.70, 640 window at the last sighting, up to 15 frames |
| fusion | `diagnose_ball_misses.py` replay = `BallFuser.update` (main top ball accepted unconditionally, else c5c) |
| splits | `gsr_io.discover`: train 57, pool 33, select 10, test 15 |
| match | GT within 15 px unless stated |
| GPU time | cache 15.9 min, motion 3.9 min, diagnose 3.9 min |

The main detector here is the Ultralytics checkpoint, not the runtime ONNX/TRT engine; this is how
the documented baseline was produced.

## Baseline reproduced

| | test (15 seq, 10,178 ball frames) | select (10 seq, 7,074) | documented (PERF_LOG 2026-10-16) |
|---|---|---|---|
| main alone recall / precision | 59.3% / 85.3% | 64.0% / 88.3% | 59.36% / 85.29% |
| **fused recall / precision** | **69.4% / 86.5%** | 71.0% / 88.6% | 69.47% / 86.51% |
| recall @5 / @10 px (fused) | 56.5% / 65.8% | 59.5% / 67.7% | — |

The test numbers match the documented baseline.

## Miss breakdown (% of ball frames)

| bucket | test | select | what it is |
|---|---|---|---|
| B7 main_false | **10.2%** | 8.5% | main detector output accepted, but not on the ball |
| B3 false_sighting | **7.7%** | 6.4% | specialist ran from a wrong last sighting |
| B5 below_threshold | **7.4%** | 8.6% | ball inside the window, c5c peak < 0.70 |
| B1 long_gap | 4.7% | 3.5% | last sighting > 15 frames old, specialist off |
| B2 no_prior | 0.2% | 1.7% | no sighting yet |
| B6 spec_wrong_place | 0.2% | 0.3% | c5c fired >= 0.70 in the wrong place |
| B4 outside_window | 0.0% | 0.0% | correct last sighting, ball outside the 640 window |
| B0 replay_check | 0 | 0 | consistency check passed |
| **total misses** | 30.6% | 29.0% | |

Ceilings (recall if the bucket were fully fixed, test): B7 79.7%, B3 77.2%, B5 76.9%, B1 74.2%,
B3+B4+B7 87.4%. Every miss except the occluded-proxy ones: 83.5%.

These are upper bounds. Fixing B3/B7 by *rejecting* a wrong ball only turns a wrong output into a
specialist call; at the measured in-window specialist recall (44% test) the expected gain is about
half the ceiling.

## Other findings

- **Specialist:** 2,986 calls on test, ball inside the window on 89.3% of them, recall 39.7% per call
  (44.4% when contained).
- **Small balls:** < 10 px fused recall 52.0% test (main 33.7%); 20+ px 74.7%. c5c adds most for
  small balls.
- **Continuity (test):** 508 miss runs, p50 2 frames, p95 22, longest 248 frames (9.9 s).
- **Occlusion:** 16.5% of test ball frames are misses flagged by the occlusion proxy (person box
  overlap), 11.1% on select. It is a proxy, not a visibility label; the audit sheets
  (`reports/diagnosis/audit/`, `human_cause` column) must be labelled before it is trusted.
- **"Outside window" reconciled:** the older diagnosis reports 157 select / 118 test
  `outside_window` misses. All of them came from a wrong last sighting (B3). With a correct
  sighting, the window missed the ball 1 (select) / 2 (test) times, so the window shift (§3.2)
  has nothing to fix.

## Split integrity

- No sequence name in two splits. No near-duplicate frames (dHash Hamming <= 4, every 25th frame).
- Same game, overlapping clip time: test SNGS-040 / select SNGS-051, test SNGS-050 / pool SNGS-039,
  select SNGS-084 / pool SNGS-092. c5c trains on `pool`, so SNGS-050 is a possible leak.
- Effect: test recall 69.44% → 68.80% without SNGS-050 (77.6% on its own), 70.71% without 040 and
  050. The leak is worth at most ~0.6 pp; report test both ways from now on.

## Consequences for the plan

1. Phase 1 starts with the plausibility check (B3 + B7 = 17.9% of test ball frames).
2. Phase 1 tasks 2-3 (camera-motion and velocity shift of the window) are dropped: B4 ≈ 0.
3. Sweep the c5c threshold on `select` before any retraining (B5 7.4%, in-window recall 44%).
4. B1 (4.7%) is smaller than B3/B5/B7; the tracker (Phase 4) stays after Phase 1 and the sweep.
5. B2 is ~0, so Phase 2 (main-detector small-ball recall) acts through B7 and the < 10 px row,
   not through "no prior" misses.
6. 90% recall on all ball frames is out of reach without recovering hidden balls: fixing every
   non-occluded-proxy miss gives 83.5%. §6 is revised accordingly.
