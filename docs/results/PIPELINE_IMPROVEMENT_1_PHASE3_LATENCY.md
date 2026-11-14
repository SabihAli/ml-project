# Pipeline improvement 1, Phase 3: live-path latency (2026-11-14)

Next steps 1-3 of `docs/plans/HANDOFF_PIPELINE_IMPROVEMENT_1.md`. Hardware `rtx-3090`; NBJW backend auto
(TensorRT). SNGS-021, 600 frames after 50 warm-up, frames preloaded (`profile_latency.py`).

## Summary

| ms | baseline (inline) | + async | + streams, c5c GPU prep | + NBJW gated to frame ticks | + motion 480 px (final) |
|---|---|---|---|---|---|
| detector p95 | 10.1 | 37.1 | 29.8 | 14.9 | 14.3 |
| c5c per call, p50 | 10.1 | 9.5 | 3.5 | 3.5 | 3.1 |
| calibrator p50 / p95 | 7.7 / 45.8 | 7.5 / 9.0 | 7.4 / 8.8 | 7.7 / 10.8 | 5.1 / 6.7 |
| **e2e p50 / p95** | **24.2 / 63.5** | **24.7 / 56.3** | **23.8 / 46.1** | **24.7 / 34.5** | **21.6 / 28.7** |
| fps (sequential) | 31.3 | 32.7 | 35.5 | 38.3 | 44.0 |

e2e p95 is inside the 40 ms ceiling, 3.7 ms over the 25 ms target. Raw: `data/outputs/pipeline_improvement_1/phase3/latency/`
(`inline/`, `async/`, `async_streams/` (holds the last of the intermediate runs), `final/`). The 140-175 ms max in every
run is `roles`, the one-off kit-colour fit at frame 250. Detector, c5c and NBJW outputs are identical with and without
the new streams (100 / 20 / 10 real inputs); c5c preprocessing is identical by test; only async and motion at 480 px
change H, and both are gated below.

# Step 1: async calibration

**Decision:** `calibration.mode: async` is the live-path default. Batch `process_sequence` always calibrates
inline (`GSRPipeline._new_calibrator(batch=True)`), because an async result lands when the background thread
finishes, which would make `predictions.json` and GS-HOTA vary from run to run.

## Accuracy (step 1)

`python -m gsr_pipeline.evaluate_calib_propagation --ks 1 5 --mode async --delays 1 2 3 5`. New in this step:
`--mode async` replaces the background worker with a deterministic one whose result lands D frames after
submission, so the gate does not depend on machine timing. Measured D: NBJW 36 ms at a ~24 ms frame loop is
2 frames.

Pitch error of GT athletes vs GT pitch positions, inline K=5, and async K=5 minus inline K=5 (mean / p95, m):

| seq | inline K=5 | D=1 | D=2 | D=3 | D=5 |
|---|---|---|---|---|---|
| SNGS-021 | 0.99 / 3.12 | -0.00 / -0.00 | +0.00 / -0.00 | +0.00 / -0.00 | +0.01 / +0.02 |
| SNGS-026 | 1.08 / 3.34 | +0.01 / +0.03 | +0.01 / +0.05 | +0.02 / +0.06 | +0.04 / +0.12 |
| SNGS-031 | 3.69 / 17.30 | +0.00 / -0.02 | +0.01 / -0.04 | +0.01 / -0.09 | +0.09 / -0.18 |
| SNGS-036 | 1.09 / 2.44 | -0.00 / -0.01 | -0.00 / -0.01 | -0.00 / -0.01 | +0.00 / +0.00 |
| SNGS-041 | 0.82 / 1.72 | +0.00 / +0.00 | +0.00 / +0.01 | +0.00 / +0.01 | +0.01 / +0.03 |
| SNGS-046 | 0.89 / 2.11 | +0.01 / +0.03 | +0.01 / +0.04 | +0.02 / +0.07 | +0.04 / +0.10 |
| SNGS-051 | 1.00 / 4.02 | +0.00 / -0.01 | +0.00 / -0.01 | +0.00 / -0.01 | +0.01 / +0.02 |
| SNGS-056 | 0.69 / 1.08 | +0.00 / -0.00 | +0.00 / +0.00 | +0.00 / +0.00 | +0.01 / +0.00 |
| SNGS-081 | 0.84 / 2.32 | +0.01 / +0.00 | +0.01 / +0.02 | -0.08 / -0.03 | +0.03 / +0.11 |
| SNGS-091 | 1.50 / 5.65 | +0.00 / -0.01 | +0.00 / -0.05 | +0.01 / -0.07 | +0.01 / -0.10 |

Worst case +0.09 m mean / +0.12 m p95 at D=5, against a 0.5 m tolerance. At D=2 the error is unchanged
(<= +0.01 m mean). Raw: `data/outputs/pipeline_improvement_1/phase3/calib_async/`.

## Latency (step 1, before steps 2-3)

`profile_latency.py --calib_mode inline|async`, SNGS-021, 600 frames. In async mode the profiler skips its
device-wide `torch.cuda.synchronize()`, which would otherwise wait for the background NBJW as well.

| ms | inline | async |
|---|---|---|
| calibrator incl. NBJW, p50 / p95 | 7.69 / 45.81 | 7.46 / 8.96 |
| detector, p50 / p95 | 7.95 / 10.14 | 8.12 / **37.13** |
| e2e p50 / p95 / max | 24.19 / 63.52 / 169.5 | 24.69 / **56.27** / 162.4 |

NBJW is off the frame loop, but e2e p95 only drops 63.5 -> 56.3 ms. The detector p95 rises to 37 ms because
the detector, c5c and NBJW TensorRT engines all run on the default CUDA stream (`TrtRunner.infer` uses the
current stream; TRT warns "Using default stream in enqueueV3()"), so a detector call queues behind NBJW
kernels already in flight. Async only pays off once NBJW runs on its own non-default stream, which is next
step 3. The 143-170 ms max is `roles`, the one-off kit-colour fit at frame 250.

## Also in step 1

- `gsr_pipeline/validate_process_stream.py`: the `ball_stats` check now includes `gate_rejects` (added in
  Phase 1, check not updated). 11/11 with async calibration, 200 frames of SNGS-021 on CUDA.
- `pytest tests/`: 104 passed.

# Step 3: CUDA streams and c5c preprocessing on the GPU

- `runtime/trt_runner.py`: each `TrtRunner` enqueues on its own CUDA stream (priority: frame engines -1, NBJW 0),
  ordered both ways against the caller's stream. In async calibration the background NBJW thread also runs its torch
  ops on a private stream (`calibration_live.py`).
- `ball_specialist.py` `_preprocess`: the three uint8 BGR crops go through one pinned buffer; flip, stacking and the
  float cast run on the GPU. c5c per call 8.6 -> 2.7 ms isolated (CPU preprocessing was 4.2 ms, the float H2D copy
  1.5 ms, the network 1.6 ms). `tests/test_ball_specialist_preprocess.py` checks it equals the old path.

Streams and priorities alone left detector p95 at ~30 ms. A contention test (detector timed while a thread loops one
NBJW part) showed why: the NBJW GPU forward (two HRNets, ~23 ms) slows the detector 7 -> 48 ms p50, its CPU decode +
homography only to 16 ms. TensorRT kernels are large and not preempted, so priorities barely help.

**Fix: gate NBJW to the frame loop.** The async worker launches one NBJW network per frame, when `calibrator.update`
runs. That is after the detector and c5c, so each ~11 ms network lands in the GPU's idle gap while the CPU does
calibration, tracking and roles. A run takes 2 frames (accuracy at D=2 unchanged, step 1). The bootstrap run is not
gated (it is waited on); `close()` releases a waiting worker. Tests: `test_async_gating_runs_one_network_per_frame_tick`,
`test_async_close_releases_a_worker_waiting_for_a_tick`.

# Step 2: camera motion

In the live loop camera motion took 7.2 ms vs 3.9 ms isolated (contention with the other threads; the 1080p resize
alone 1.6-2.1 ms vs 0.8 ms isolated). OpenCV thread count (24 vs 4) changed nothing. Cheaper downscales
(INTER_LINEAR, stride-2) differ from the current image by up to 82 grey levels and were not taken.

`calibration.motion.work_width` 640 -> 480: motion 7.2 -> 4.8 ms in the loop. Accuracy vs GT, width 480 minus 640 on
the same 10 sequences: within +-0.02 m mean and p95, inline K=5 and async D=2, no frame without H
(`data/outputs/pipeline_improvement_1/phase3/calib_motion480/`). Still over the 2 ms target: the rest is the resize and
Shi-Tomasi corners. Next options: reuse tracked corners between frames instead of re-detecting, or take the downscaled
grey frame from the detector's GPU preprocessing.

# Checks

`pytest tests/` 114 passed. `validate_process_stream` 11/11 (async, CUDA, 200 frames). Batch `gsr_pipeline.run` on
SNGS-021 (300 frames, CUDA) completes and `gsr_pipeline.evaluate` scores it.
