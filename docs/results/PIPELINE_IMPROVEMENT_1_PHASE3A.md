# Pipeline Improvement Plan 1 — Phase 3a results: replay, mining, multi-peak

**2026-11-12.** Hardware `kaggle-t4` (T4 x2). Notebook `notebooks/pipeline_improvement_1/phase3a_mining.ipynb`
(`scripts/pipeline_improvement_1/phase3/phase3a_mine.py`). Raw: `data/outputs/pipeline_improvement_1/phase3a/`.

## Data

- SoccerNet-Tracking `train` is GSR `train` (same 57 clips, game ids 4/6/9). Tracking `test` (49 clips, games
  7/8/11) is GSR `test`, which c5c's `pool` already held (82 clips). The Kaggle GSR copy lacks it; converted
  with `convert_sntracking.py` (labels + symlinked frames), `pool` is 82 clips again.
- Confirmed by the replay: on those clips c5c's below-threshold rate is 0.4% (held-out: 7%), i.e. it trained on them.
- Split integrity with the new clips: no new overlaps, 0 near-duplicate frames between {select, test} and {train, pool}.
- SoccerNet has no further labelled broadcast ball data: every labelled clip is in our splits.

## Phase 1 (live system) replayed per split

| split | clips | ball frames | recall @15 | precision | B3 | B5 | B6 | B7 | B1 |
|---|---|---|---|---|---|---|---|---|---|
| train | 57 | 40,234 | 93.63% | 95.91% | 0.58% | 1.55% | 0.39% | 3.47% | 0.18% |
| pool (GSR valid) | 33 | 23,187 | 93.74% | 95.65% | 0.57% | 0.67% | 0.17% | 3.95% | 0.13% |
| pool (GSR test) | 49 | 33,804 | 92.21% | 94.45% | 1.23% | 0.43% | 0.16% | 5.11% | 0.55% |
| select | 10 | 7,074 | 76.80% | 90.93% | 4.38% | 7.42% | 0.95% | 5.81% | 2.97% |
| test | 15 | 10,178 | 76.29% | 90.21% | 5.50% | 7.00% | 0.68% | 6.61% | 3.57% |

Select/test reproduce Phase 1 exactly. **Seen clips 92–94%, unseen 76–77%**: the gap is generalisation.
pool's GSR clips come from the same three games as test, so it is clip memorisation, not unseen venues.

## Mined set (`mined.csv`, cap 150 per clip and kind)

| source | hard_pos | neg_spec | neg_main |
|---|---|---|---|
| train | 946 | 160 | 1,588 |
| pool (GSR valid) | 265 | 296 | 2,400 |
| pool (GSR test) | 274 | 200 | 3,568 |

Only 1,485 hard positives: c5c handles its own training clips, so mining them with c5c finds little.
Realistic hard examples need cross-fitting (mine each fold with a model that did not train on it).

## Multi-peak selection — REVERT

Highest c5c peak inside the Phase 1 gate radius instead of the top peak. On `select` lower peak thresholds add up
to +1.2 pp recall (77.98% at 0.2) but precision falls to 88.4–89.6%, under the 89.93% floor. The chosen setting
(0.45 / 1.01, effectively Phase 1) gives 76.14% / 90.27% on `test` vs 76.29% / 90.21%. Not kept.

## Next

Teacher–student (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md Phase 3 status): profile live-path latency on the
RTX 3090 first (`scripts/pipeline_improvement_1/phase3/profile_latency.py`), then a large offline teacher
pseudo-labels SoccerNet full matches, distilled into a temporal (5-channel) main detector and a retrained c5c,
with cross-fitted mining. Runtime cost must stay inside the §1.4 budget.
