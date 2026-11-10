# Pipeline Improvement Plan 1 — Phase 4 results: ball tracker (closed, reverted)

**Closed 2026-11-11. Verdict: REVERT.** No tracker passed the kill criterion (+0.5 pp recall,
≤ 1 pp precision loss) against the Phase 1 system. Phase 1 (plausibility gate + c5c threshold 0.45)
stays live; nothing from this phase is wired into `BallFuser` or `pipeline.py`.

Hardware `kaggle-t4` (T4 x2). Splits as Phase 0: knobs chosen on `select` (10 sequences), frozen
for `test` (15). Same scorer for every system (recall @15 px on measured outputs).

| run | notebook | script | raw outputs |
|---|---|---|---|
| 4a: own tracker | `notebooks/pipeline_improvement_1/phase4_tracker.ipynb` | `scripts/pipeline_improvement_1/phase4/phase4_sweep.py` | `data/outputs/pipeline_improvement_1/phase4/` |
| 4b: SORT family (boxmot 26.0.0) | `notebooks/pipeline_improvement_1/phase4b_sort.ipynb` | `scripts/pipeline_improvement_1/phase4/sort_compare.py` | `data/outputs/pipeline_improvement_1/phase4/sort/` |

## Systems

- **Phase 1** (baseline): detector top ball accepted when within r0 + speed·gap of the last sighting or conf ≥ 0.6; else c5c on a 640 window at the last sighting (gap ≤ 15).
- **Robust tracker** (`gsr_pipeline/ball_tracker_robust.py`): IMM over a calm and a manoeuvre constant-velocity model, Mahalanobis gate, camera-motion compensation (break = one-frame restart), occlusion model following the athlete box covering the ball, c5c window centred on the prediction. 12 knobs, coordinate descent (49 runs).
- **ByteTrack, OC-SORT, BoT-SORT** (boxmot, no re-id): detector balls plus the c5c peak as boxes; one sticky ball track output per frame. Box size, IoU vs centroid association, track lifetime, min hits and thresholds searched per tracker. BoT-SORT searched without camera compensation, run on test with and without it.

## `test` results

| system | recall @15 | precision | longest miss | best on `select` |
|---|---|---|---|---|
| **Phase 1** | **76.29%** | **90.21%** | 248 | **76.80%** |
| robust tracker | 73.88% | 88.84% | 310 | 76.56% |
| robust tracker + predicted outputs | 78.07% | 80.23% | — | — |
| ByteTrack | 73.86% | 90.18% | 252 | 73.96% |
| OC-SORT | 69.57% | 89.20% | 252 | 70.46% |
| BoT-SORT | 68.23% | 90.40% | 586 | 70.51% |
| BoT-SORT + camera compensation | 67.90% | 90.34% | 586 | — |

No system beat Phase 1 even on `select`, so this is not a test-split artefact.

### Robust tracker ablations (`test`)

| variant | recall | precision |
|---|---|---|
| full | 73.88% | 88.84% |
| without occlusion model | 74.68% | 88.87% |
| without camera compensation | 74.40% | 88.70% |
| one motion model | 70.66% | 89.64% |

### Recall by condition (`test`, measured outputs)

Conditions from labels: occluded = GT ball inside a GT person box; abrupt = second difference of the GT track > 6 px. No post-cut frames occurred (no frame without a motion estimate).

| system | occluded (n=3331) | abrupt (n=1925) | other (n=4922) |
|---|---|---|---|
| Phase 1 | 60.28% | 77.82% | 86.53% |
| robust tracker | 59.11% | 72.05% | 84.60% |
| ByteTrack | 58.39% | 75.53% | 83.67% |
| OC-SORT | 58.66% | 61.92% | 79.95% |
| BoT-SORT | 57.28% | 65.56% | 76.68% |

## Findings

1. **Association is not the bottleneck.** Every Kalman-plus-matching tracker, our own and the off-the-shelf ones, lands at or below ~74%. The remaining misses are detection errors: after the tracker the buckets were B3 false_sighting 8.1%, B7 main_false 7.9%, B5 below_threshold 6.2%, B1 long_gap only 2.4% of ball frames.
2. **Phase 1's gate is the better filter.** A distance radius that grows with the gap rejects more false detections than a covariance gate. The robust tracker raised false sightings from 5.5% to 8.1% of ball frames.
3. **Constant velocity fails on kicks.** Abrupt-motion recall falls to 62–66% for OC-SORT and BoT-SORT and to 62% for the single-model ablation. The two-model IMM is the one tracker component that earned its place (+3.2 pp vs one model).
4. **Camera compensation did not help** at 25 fps on these clips (robust tracker −0.5 pp, BoT-SORT −0.3 pp with it). The occlusion model also cost 0.8 pp.
5. **Predicted outputs** lift occluded-frame recall from 59% to 68% but cost ~10 pp precision; not worth outputting at this precision.

## Consequences for the plan

- Phase 4's exit condition was not met. The tracker code stays as a documented record (module, tests, both evaluation scripts) for a later retry once detections are cleaner; it is not imported by the runtime.
- Next: Phase 3 (specialist: B5 + localisation), then Phase 2 (detector: B7, small balls). Phase 3 can reuse the Phase 4 caches with athlete boxes.
- If the tracker is revisited: keep the Phase 1 gate as the acceptance rule and add only the IMM manoeuvre model; drop occlusion following and camera compensation.
