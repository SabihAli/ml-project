# Ball detection: from ~60% to above 90% recall

How the ball system's recall on the held-out `test` split (15 GSR sequences, a ball counts as found within 15 px)
grew over Pipeline Improvement 1. Every row is measured on the same frames with the same scorer. Thresholds were
chosen on `select` and then frozen for `test`.

| step | what changed | test recall | precision |
|---|---|---|---|
| 0 | 2-class detector (YOLO26s, 736x1280) alone | 59.4% | 85.3% |
| 1 | + c5c, the temporal ball specialist (3 frames, 640 window at the last sighting) for frames the detector misses | 69.5% | 86.5% |
| 2 | + plausibility gate (a ball must be near the last sighting or confident), specialist threshold 0.45 | **76.3%** | 90.2% |
| 3 | offline ball pass: 1920 detector with motion (frames t-1, t, t+1) | 83.2%\* | 90.1%\* |
| 4 | + trajectory pass (temporal support, gap fill up to 10 frames) | 88.1% | 93.9% |
| 5 | + second model (5-frame heatmap net), candidates fused 0.7 / 0.3 | 88.4% | 93.8% |
| 6 | + global trajectory decode over the whole clip | **89.4%** | 93.5% |
| 6 | the same, scored lag-tolerant (see below) | **92.9%** | |

\* step 3 alone was only scored on `select`.

## What did the work

- **A second look where the detector misses (steps 1–2).** Most misses are small, blurred or partly hidden balls.
  c5c looks again at a small window around where the ball was last seen, using three frames, which recovers about
  10 points. The gate then rejects far-off false balls, which let the threshold drop and recovered another 7 points
  at higher precision.
- **Resolution and motion (step 3).** At 1920 a ball is about 2.25 times as many pixels as at 1280. Adding the
  neighbouring frames as grey channels lets the model see the ball move against the background.
- **Using the whole clip (steps 4–6).** Each frame keeps its top 5 candidates, and a decode picks the most plausible
  ball path through them. That fills short gaps and drops one-frame false balls. Two different models catch different
  misses, so fusing them helps. Doing the decode over the whole clip instead of locally added the last point.
- **Fixing the measurement (lag-tolerant).** An audit of near misses found that GSR's ball labels trail fast balls,
  by +0.2 frames on average. About 24 of 40 sampled "misses" had the ball on target and the label off it. Allowing for
  that lag gives 92.9%.

## Where each number applies

- Steps 0–2 are the **live pipeline** (`process_stream`): **76.3% / 90.2%** is what runs live today.
- Steps 3–6 are the **offline ball pass**. It looks at future frames and the whole clip and runs at about
  28 fps at 1920, so it does not fit the live latency budget. It is used to label training data
  (`data/pseudo_labels/phase3b_teacher`, plus GSR train and pool) for the live models.
- Phase 3c distils it into the live models. The goal is to move the live number from 76.3% toward the offline one.

Sources: `docs/results/PIPELINE_IMPROVEMENT_1_PHASE1.md`, `docs/results/PIPELINE_IMPROVEMENT_1_PHASE3B_TEACHER.md`,
`docs/architecture/BALL_TEMPORAL_ARCH.md`, `docs/path_b/PERF_LOG.md`.
