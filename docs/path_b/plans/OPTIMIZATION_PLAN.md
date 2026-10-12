# Path B latency optimization plan

Status: Phases 0 and 1 built 2026-10-07 (see §7); Phases 2–6 not started.
Scope: the Path B live pipeline: detector, tracker, LiveCalibrator (NBJW + camera motion),
projection, role descriptors, and the evaluation runner `notebooks/path_b_eval.py`.

**Goal:** minimize per-frame latency (mean and tail) **without losing accuracy**. Every change
falls into one of two classes:

- **Exact.** Outputs are bit-identical, or identical up to float rounding that disappears in
  later rounding (for example keypoints rounded to whole pixels). These are verified with a
  parity test against the current code. No GS-HOTA run is needed to accept them, though the
  Kaggle smoke test re-confirms.
- **Gated.** Outputs change slightly (FP16 inference, async timing). These are accepted only
  if they pass an explicit accuracy gate, listed per phase.

---

## 1. Baseline

Source: `docs/results/PATH_B_EVAL_T4_2026-10-05.md`. Tesla T4, 3 GSR valid sequences × 100 frames, K=5, inline mode,
NBJW on TensorRT FP16, detector on **ORT GPU**. Everything runs back to back on one thread.

| Stage | Mean ms | p50 | p95 | Share | Notes |
|---|---|---|---|---|---|
| frame load (`cv2.imread`) | 10.7 | 10.8 | 12.2 | 12% | production uses NVDEC |
| detector end-to-end | 24.3 | 24.2 | 25.4 | 28% | preprocess 5.0 / inference 17.6 / NMS + D2H 1.9 |
| tracker (ByteTrack) | 6.9 | 6.3 | 8.5 | 8% | one 183 ms outlier |
| calibration (LiveCalibrator) | 37.5 | 15.9 | 128.8 | 44% | ~16 ms motion on propagated frames; ~115–130 ms on NBJW frames |
| pitch projection | 1.3 | 1.3 | 1.6 | 1% | |
| descriptors + records | 5.2 | 5.4 | 6.9 | 6% | |
| **whole pipeline** | **85.9** | 64.7 | **178.1** | | **11.65 fps** |

Breakdown of an NBJW (fresh) frame, from the profiler, GPU synchronised:

| Step | ms |
|---|---|
| image prep (H2D + on-device resize) | 2.6 |
| keypoint HRNet (TRT FP16) | 32.9 |
| line HRNet (TRT FP16) | 37.6 |
| **heatmap decode + keypoint completion** | **39.4** |
| homography RANSAC | 0.8 |
| camera-param voting (*profiler only; off in the pipeline*) | 19.0 |

Accuracy reference for this plan (same smoke config): GS-HOTA persons **58.95**, with_ball
**57.75**, DetA 52.89, AssA 71.78. The detector parity gate passes on ORT GPU.

**Two structural facts drive most of the plan:**

1. Everything is serial. The CPU idles while the GPU runs the detector and NBJW, and the GPU
   idles while the CPU decodes JPEGs, tracks, estimates camera motion and computes descriptors.
2. The tail (p95 178 ms) comes entirely from the 1-in-5 frames that run NBJW inline.

---

## 2. Target

| Milestone | Mean ms | p95 ms | Accuracy |
|---|---|---|---|
| Baseline | 85.9 | 178 | reference |
| After Phases 0–3 (exact) | ~45–50 | ~120 | identical |
| After Phase 4 (detector TRT, gated) | ~37–42 | ~110 | parity gate |
| After Phase 5 (async calibration, gated) | ~35 | ~40–45 | GS-HOTA within 0.3, pitch gate |
| Phase 6 (NBJW engine investigation) | open | open | parity |

These figures are estimates from the smoke-test numbers. Each phase re-measures them on
Kaggle and updates the table.

---

## 3. Phases

### Phase 0: measurement, harness fixes, correctness bugs

The rest of the plan relies on the measurements being right, and two of these bugs are latency
hazards in their own right.

**0.1 Fix the latency report** (`notebooks/path_b_eval.py`)
- Print the resolved `calib_every_k` and `mode`. The report currently prints `config`.
- `profile_components` always runs `heuristic_voting`. Gate it on `compute_cam_params` (or
  report it on a separate "not on path" line) so the breakdown sum (132.5 ms) matches real fresh
  frames (~113 ms).
- Update the stale scope note ("calibration is per-frame NBJW") and the module docstring.
- Add per-`h_source` latency rows (fresh vs propagated). The mean hides the bimodal shape.
- Add a finer split inside heatmap decode (max-pool / topk+D2H / `coords_to_dict` /
  `complete_keypoints`), so Phase 1 is measured rather than inferred.

**0.2 Bug B: NBJW storm on low-texture shots** (`gsr_pipeline/calibration_live.py`)
- How it happens: motion fails (`est.M is None`, e.g. fewer than `min_tracked` corners on a
  close-up), so `_motion_ok = False` and the trigger is `no_motion`. NBJW runs and finds no H.
  `_fallback` takes the reuse path, which sets `_motion_ok = True`. The next frame fails again
  and the cycle repeats: **NBJW runs on every frame at ~130 ms** during close-ups and replays.
- Fix: add a `min_trigger_interval_frames` (for example 3) for event triggers (`no_motion`,
  `low_inliers`, `chain_broken`). `scene_cut` and `first` bypass it. While cooling down, emit
  `reused`.
- Test: a synthetic sequence where `motion.update` always returns `M=None` and NBJW always
  returns `H=None`. Assert that NBJW runs ≤ ceil(N / interval) times.

**0.3 Bug C: tracker does not age on empty frames** (`gsr_pipeline/track.py:115`)
- `update` returns before calling the BYTETracker when there are no detections, so lost tracks
  never age through empty frames. That changes association and track IDs after a gap.
- Fix: always call the underlying tracker with an empty input.
- ⚠ This changes outputs (it fixes them). Keep it as a separate commit and record the
  GS-HOTA delta so it is not confused with Phase 1–3 "exact" changes.

**0.4 Parity harness** (new: `scripts/parity_pipeline.py`, or a mode in `path_b_eval.py`)
- Run the pipeline twice (reference code path and optimized code path) over N frames and
  compare, per frame: detection boxes, track IDs, NBJW keypoint dicts, H (max abs element
  difference after normalisation), `h_source`, and `bbox_pitch`.
- Must run on CPU locally (ORT CPU / torch) and on Kaggle GPU.
- Every "exact" change in Phases 1–3 must pass this with zero differences (or differences
  below 1e-6 in H).

**Open questions, Phase 0**
- What should the trigger cooldown be? Too long delays recovery after a genuine cut back to the
  wide shot. Proposal: 3 frames, scene_cut exempt. Needs a long clip containing replays to tune.
- Should 0.3 land before or after the baseline re-measurement? Proposal: after Phase 1–3 parity
  is proven, as its own measured step.
- Is a CPU parity run good enough for GPU code paths (streams, threads)? No: thread and stream
  changes must also pass parity on Kaggle.

**Skills:** `ecc:benchmark-methodology`, `ecc:benchmark` (measuring correctly, warm-up,
percentiles); `ecc:python-testing`, `ecc:tdd-workflow`, `tdd-guide` (the storm test and the
parity harness); `ecc:ai-regression-testing`, `ecc:eval-harness` (a regression gate on model
outputs); `ecc:verification-loop`.

---

### Phase 1: NBJW decode (exact). Fresh frames ~115 → ~80 ms

File: `gsr_pipeline/nbjw_calib/utils/utils_heatmap.py`. Add fast versions next to the
originals. Do not edit the originals, so parity can call both.

**1.1 Separable max-pool** (`get_keypoints_from_heatmap_batch_maxpool[_l]`)
- Today: `max_pool2d(kernel=31, stride=1)` over 57×135×240 (keypoints, after padding with 1.0)
  and `kernel=21` over 23×135×240 (lines). That is ~1.8 G and ~0.33 G comparisons per frame.
- Change: `max_pool2d(kernel=(1,k))` followed by `max_pool2d(kernel=(k,1))`. A max over a
  square window equals the max over rows of column maxima, exactly, because max only selects
  values and never rounds. That is ~15× fewer comparisons.
- Padding must be reproduced exactly:
  - keypoints: pad with constant 1.0 on all sides first (as now), then both passes with
    `padding=0`;
  - lines: implicit −inf padding, i.e. `padding=(0,pad)` then `(pad,0)`.
- `local_maxima = pooled == heatmap` stays an exact equality test, because the pooled values are
  copies of heatmap values.

**1.2 Remove the Python list round-trip after topk**
- Today: indices and scores go to numpy, then Python loops build nested lists, then
  `torch.tensor(...)`, then `coords_to_dict` calls `.item()` per element.
- Change: build the `(B, C, K, 3)` array with numpy vector operations and make `coords_to_dict`
  read from numpy. One D2H copy of the topk results, no per-element calls.
- Exact: the same values in the same order.

**1.3 Two-point line fit in `complete_keypoints`**
- Today: `scipy.stats.linregress` runs on 2-point lines up to about 45 times per frame (~50–100 µs
  each, plus the scipy overhead).
- Change: closed form `slope = (y2-y1)/(x2-x1)` and `intercept = y1 - slope*x1`, keeping the
  `+1e-7` nudge on the last x and the `+1e-7` in the denominator exactly as written.
- Near-exact: linregress computes the slope via covariance, so the last bits can differ, and the
  result is then `round(·, 0)`. Parity must show identical keypoint dicts; a pixel flip at a .5
  boundary is the only theoretical difference. If parity shows any, keep linregress.
- Also hoist `lines_list.index(...)` out of the loops (a constant table).

**1.4 Do not "fix" the upstream off-by-one**
- `for key in range(1, len(keypoint_aux_pair_list))` skips the last auxiliary pair. It is a bug
  upstream, but the NBJW thresholds and our accuracy reference include it. Leave it and add a
  comment. Changing it is an accuracy experiment, not an optimization.

**Acceptance:** parity harness shows identical keypoints, lines and H on ≥300 frames across 3
sequences; the decode sub-stage drops from 39 ms to a measured value (expected < 8 ms).

**Open questions, Phase 1**
- How much of the 39 ms is the max-pool and how much is Python? Phase 0.1's sub-split answers
  this. If the max-pool is small, 1.2/1.3 become the main lever and the estimate shifts.
- Do we vendor-patch `nbjw_calib` or add `gsr_pipeline/nbjw_fast_decode.py`? Proposal: a new
  module in `gsr_pipeline/`, and `calibrate.py` imports it, so the vendored package stays
  pristine for parity.
- TensorRT ties in local-maxima: with FP16 heatmaps, equal neighbouring values are more likely.
  Separable pooling preserves the tie behaviour exactly (it is the same equality test), so there
  is nothing to decide, but the parity run should include TRT heatmaps, not just torch.

**Skills:** `ecc:pytorch-patterns`; `senior-computer-vision`; `ecc:python-patterns`;
`ecc:benchmark-optimization-loop` (measure, change, re-measure); `ecc:python-testing`.

---

### Phase 2: overlap CPU work with GPU work (exact). ~−25 to −35 ms mean

Today the main thread runs every stage in order. Most CPU work here is OpenCV / numpy, which
releases the GIL, so threads give real parallelism.

**2.1 Camera motion in parallel with the detector** (`camera_motion.py`, `calibration_live.py`)
- Key dependency fact: `update(frame_t, boxes_t)` uses the **previous** frame's mask for corner
  selection. `boxes_t` is only needed to build the mask stored for frame t+1. So motion for frame
  t does not depend on the detector's output for frame t.
- Split the API:
  - `estimate(frame_t) -> MotionEstimate`: grey/resize/hist, corners (prev mask), LK, RANSAC;
  - `commit_mask(boxes_t)`: builds and stores the mask for the next call.
- Run `estimate` on a single worker thread, submitted before `detector.detect(frame_t)`. Join
  after the detector returns, then call `commit_mask` and run the calibrator's trigger logic.
- `LiveCalibrator.update` gains a variant that takes a precomputed `MotionEstimate`.
- Saves ~16 ms on propagated frames (80% of frames), as long as the detector takes ≥16 ms.
- Exact: the same inputs and operations; only the timing changes.

**2.2 Frame prefetch**
- A bounded queue (depth 2–3) filled by a decode thread (`cv2.imread` releases the GIL).
  Frame load drops off the critical path: ~−10.7 ms.
- In production the source is NVDEC / `FrameSource`; the same producer/consumer shape applies,
  so build it as a small reusable `PrefetchingFrameSource` wrapper.

**2.3 Pipeline the post-GPU CPU stages**
- Stage split per frame: [A] detect (GPU) → [B] track → calibrate trigger/propagate → project
  → describe/records (CPU).
- Run [B] for frame t on a worker while [A] runs for frame t+1. ByteTrack is sequential in t,
  which is fine because B stays on one thread in frame order.
- Ordering constraint: calibration's NBJW decision for frame t+1 needs the motion chain up to t,
  which B produces. Keep motion estimation and the trigger decision inline (2.1 already hides the
  motion cost) and pipeline only track/project/describe. Alternatively pipeline the whole B; but
  then an inline NBJW run in B would stall the next A only if it shares the GPU stream (see
  Phase 5).
- Saves up to ~13 ms (track + project + describe) when the detector is the longer leg.

**2.4 Small exact items**
- Detector preprocess: reuse a preallocated uint8 canvas (only the pad rows need refilling, and
  only once if the frame size is constant) instead of `np.full` per frame; `cv2.resize(..., dst=view)`
  writes straight into it.
- Pinned host staging buffer for the detector canvas upload and the NBJW frame upload, so
  `non_blocking=True` actually overlaps.
- `ByteTracker.update`: move the `TrackerInput` class to module level; skip
  `detections_df.copy()` when the caller does not reuse the frame.
- Descriptor loop: replace `iterrows` with array iteration over `bbox_ltrb`, and convert the
  torso crop to HSV/Lab once per crop.
- Expected total: ~2–4 ms.

**Acceptance:** parity harness identical (detections, IDs, H, pitch, descriptors) on Kaggle GPU;
mean latency measured with the pipelined clock (see open question).

**Open questions, Phase 2**
- **How do we measure latency once stages overlap?** The sum of stage times no longer equals frame
  latency. Proposal: report (a) throughput (frames / wall time), (b) per-frame end-to-end latency
  = time from frame available to records emitted, and (c) the per-stage busy times. Phase 0.1
  should add the timestamps.
- Threads or processes? Threads, as long as the work stays in GIL-releasing C (OpenCV, numpy,
  torch). Check this: py-spy or a GIL profiler on the worker during the Kaggle run. pandas-heavy
  code (tracker, records) holds the GIL; if it contends, convert those paths to numpy first.
- How many OpenCV threads? `cv2.setNumThreads` interacts with our worker threads on a 2-vCPU
  Kaggle box (and with torch's intra-op threads). Needs a sweep: {0, 1, 2}.
- Does the live `GSRPipeline.process_stream` get the same restructuring, or only the eval runner?
  Proposal: build it in `gsr_pipeline/` (a `PipelinedRunner`) used by both. Otherwise the eval
  measures a pipeline that production does not run.
- Is the latency per frame or per output? With pipelining the latency of frame t includes queue
  wait; the live WS feed cares about end-to-end lag, not throughput. Define this before tuning.

**Skills:** `ecc:parallel-execution-optimizer`; `ecc:latency-critical-systems`;
`ecc:data-throughput-accelerator`; `ecc:python-patterns`; `senior-ml-engineer`;
`ecc:benchmark-methodology` (measuring pipelined latency). Optional external:
`nvidia/skills@deepstream-dev` (2.4K installs; NVDEC / GPU video pipelines, relevant to 2.2 in
production). Install: `npx skills add nvidia/skills@deepstream-dev`.

---

### Phase 3: GPU streams and the async calibration prerequisites (exact by itself)

**3.1 Bug A: shared default stream**
- `TrtRunner.infer` uses `torch.cuda.current_stream()`. On a worker thread that is the same
  legacy default stream the detector uses. In async calibration mode, ~70 ms of NBJW kernels
  queue in front of the next detector call, so the "non-blocking" worker still stalls the frame
  loop.
- `OrtRunner._infer_bound` calls `torch.cuda.current_stream(...).synchronize()`, which on a
  shared stream makes the two threads wait for each other's work.
- Fix:
  - give the calibrator its own `torch.cuda.Stream(priority=low)`, and the detector a
    `Stream(priority=high)`; run each runner under `with torch.cuda.stream(s)`;
  - NBJW preprocess must use the same stream (H2D + resize);
  - ORT: pass `user_compute_stream` (the torch stream's handle) in the CUDA EP provider options,
    so ORT runs on our stream. Then the two syncs become an event wait / one sync on that stream.
    Re-run the ORT GPU parity check afterwards: this is the code that had the race.
- With inline mode this changes nothing (one thread). It is the prerequisite for Phase 5.

**3.2 Share the frame upload (optional)**
- The detector uploads a 736×1280 canvas and NBJW uploads the full 1920×1080 frame. On NBJW frames,
  upload the frame once to a device buffer and have both preprocessors read from it. The detector
  keeps its CPU-resize path (the parity-exact one); only the NBJW side consumes the device frame.
  ~1–2 ms on fresh frames only. Low priority.

**Acceptance:** parity identical inline; with an artificial async NBJW load, the detector's
inference time stays within ~10% of its solo time (T4 timeslices, so it will not be zero).

**Open questions, Phase 3**
- Do T4 stream priorities preempt in practice? They change block scheduling, not running kernels;
  a long HRNet kernel can still delay the detector. Measure the detector's p95 with NBJW
  running in the background.
- Does the onnxruntime-gpu 1.22 Python API accept `user_compute_stream` with IOBinding? It
  should; verify on Kaggle.
- Does TensorRT need one execution context per thread? Each runner is used by one thread only, so
  no; document that constraint.

**Skills:** `ecc:pytorch-patterns`; `ecc:latency-critical-systems`; `senior-ml-engineer`;
`ecc:pytorch-build-resolver` (agent) for CUDA/stream errors; `ecc:python-testing`.

---

### Phase 4: detector on TensorRT FP16 (gated). ~−8 to −10 ms

- The smoke test ran the detector on **ORT GPU** (17.6 ms inference) because no detector engine
  was built on Kaggle. `select_backend` already supports TensorRT FP16 with sidecar checks.
- Build: `scripts/build_trt_engine.py` on the T4 (same `tensorrt-cu12<11` pin used for NBJW).
- **Gate:** the existing parity gate (`scripts/parity_gate.py`, box for box vs the Ultralytics .pt)
  plus per-class detection metrics within noise of ORT FP32 (athlete F1 96.30, ball F1 63.27 on
  the smoke set), then GS-HOTA within 0.3.
- If FP16 fails the gate on the ball (small objects are the usual FP16 casualty): try keeping the
  detection head in FP32 (TensorRT per-layer precision constraints) before giving up.

**Open questions, Phase 4**
- What box tolerance should the parity gate use for FP16? It was written for FP32 tiers (min-IoU
  0.99998). FP16 needs a looser bar; propose min-IoU ≥ 0.99 and |Δconf| ≤ 0.01, plus the
  detection-metric gate as the real decider.
- Ball conf threshold 0.15 sits on a steep part of the curve; FP16 conf shifts near it flip
  detections. Should the gate include a ball-only recall check? Proposal: yes.
- Engines are GPU-specific. Production GPU (RTX 30-series per plan 1.4) needs its own build and
  gate run; Kaggle results do not transfer.

**Skills:** `senior-computer-vision`; `senior-ml-engineer`; `ecc:mle-workflow`;
`ecc:ai-regression-testing`; `ecc:eval-harness`.

---

### Phase 5: async calibration (gated). p95 ~120 → ~40–45 ms

- `LiveCalibrator(mode="async")` already exists: NBJW runs on a worker, and results are re-anchored
  through the motion ring. It needs Phase 3 (streams) to deliver, and Phase 1 to keep the worker
  under one K-interval (fresh ~80 ms < 5 frames at 25 fps = 200 ms of wall time, so it keeps up).
- Accuracy change: the triggering frame (and the 1–3 frames until the result lands) gets a
  propagated H instead of a fresh one. The anchor itself is unchanged; it lands late and is carried
  forward by motion, which the K=5 study shows costs ~0.18 m vs per-frame NBJW.
- **Gate:**
  1. `evaluate_calib_propagation.py` extended with an async mode: pitch error vs GT within 0.5 m
     of inline K=5, on ≥5 valid sequences.
  2. GS-HOTA within 0.3 of the Phase 4 result.
  3. Count of `h_source` per kind, plus how often a result was dropped because the chain broke
     (log it; today it is a debug line).
- GIL check: the NBJW worker's decode (after Phase 1) and RANSAC are short; confirm with a
  profile that the main thread is not starved.

**Open questions, Phase 5**
- Is the async staleness acceptable for events? Possession uses pitch positions; the 1–3 frames of
  propagated H are the same quality as the other 80% of frames, so probably yes. Confirm with the
  possession demo on one clip.
- What happens after a scene cut in async mode? Frames between the cut and the first result have no
  valid H (the chain is broken). Inline would block for one NBJW run instead. Options: block on
  `scene_cut` only (hybrid mode), or emit `none` for ~3 frames. Proposal: hybrid. Inline on
  `first` / `scene_cut`, async otherwise. Needs a decision.
- Does a GPU-bound worker hurt the detector on a T4 enough to cancel the win? Phase 3's
  measurement answers this; on an RTX 30-series it matters less.

**Skills:** `ecc:latency-critical-systems`; `ecc:parallel-execution-optimizer`;
`senior-computer-vision`; `ecc:eval-harness`; `ecc:benchmark-methodology`.

---

### Phase 6: NBJW network speed (investigation). Open-ended

- 32.9 ms (keypoints) + 37.6 ms (lines) at 540×960 FP16 on a T4 is slower than the FLOP count
  suggests for HRNet-W48 (~165 GFLOPs at this resolution, ideally ~10 ms on a T4).
- Investigate:
  1. `trtexec --loadEngine=... --dumpProfile --separateProfileRun` to see per-layer times and
     precisions. Are layers stuck in FP32 (resize/upsample ops, or layers the ONNX exporter wrapped
     oddly)?
  2. Do the upsampling ops in the HRNet fusion layers export as `Resize` with
     `align_corners`-style modes that TRT runs slowly?
  3. Builder options: `builder_optimization_level=5` is already set; try
     `--best`-equivalent tactic sources and a larger workspace.
  4. Run the two nets on two streams at once. Only helps if neither saturates the GPU; the profile
     answers that.
- Exactness: any change to the engine build is gated like the FP16 engine itself (pitch error vs
  torch FP32, 0.35 m mean today). INT8 is **out of scope** (an accuracy cost by definition).

**Open questions, Phase 6**
- Is the FP16 deviation (0.35 m mean vs torch FP32) concentrated in specific layers? The same
  profile can point to FP32 fallbacks for the sensitive layers, which might improve accuracy *and*
  keep speed.
- Is a lower NBJW input resolution acceptable? No, not under this plan's rule: it is an
  accuracy trade, so it stays out.

**Skills:** `senior-computer-vision`; `senior-ml-engineer`; `ecc:pytorch-patterns`;
`ecc:benchmark-optimization-loop`; `ecc:deep-research` (TensorRT HRNet performance notes).

---

## 4. Build order

| # | Item | Class | Depends on | Why this position |
|---|---|---|---|---|
| 1 | 0.1 report fixes + decode sub-split + per-source rows | tooling | – | every later step is measured with it |
| 2 | 0.4 parity harness | tooling | – | gates every exact change |
| 3 | 0.2 trigger cooldown (bug B) | fix | 2 | latency hazard on full matches |
| 4 | 1.1–1.3 fast NBJW decode | exact | 1, 2 | largest single exact win, small and self-contained |
| 5 | 2.4 small exact items | exact | 2 | cheap, no structural change |
| 6 | 2.1 motion ‖ detector | exact | 2, API split | biggest exact win on 80% of frames |
| 7 | 2.2 frame prefetch | exact | – | independent, simple |
| 8 | **Kaggle checkpoint 1**: parity + smoke test | measure | 4–7 | confirm exactness on GPU and the new baseline |
| 9 | 3.1 streams (bug A) + ORT `user_compute_stream` | exact | 8 | prerequisite for 2.3 at full effect and for 5 |
| 10 | 2.3 pipelined post-GPU stages (`PipelinedRunner`) | exact | 6, 9 | the most structural change, so after the simpler wins |
| 11 | 0.3 tracker empty-frame fix (bug C) | fix, output-changing | 8 | separately measured so it doesn't muddy the exact deltas |
| 12 | **Kaggle checkpoint 2** | measure | 9–11 | |
| 13 | 4 detector TRT FP16 | gated | 12 | needs a clean exact baseline to compare against |
| 14 | 5 async / hybrid calibration | gated | 9, 4 | needs streams and the fast decode |
| 15 | **Kaggle checkpoint 3**: full valid set | measure | 13–14 | the 3×100 smoke test is too small for gated decisions |
| 16 | 6 NBJW engine investigation | investigation | 15 | open-ended; doesn't block anything |
| 17 | 3.2 shared frame upload | exact | 9 | small win, do last |

Each item is one commit with its own test. Exact items record "parity: identical (N frames)" in the
commit message; gated items record the gate numbers.

---

## 5. Cross-cutting rules

- **Never edit the reference path in place.** Fast paths are separate functions or modules that the
  parity harness can compare against the originals (vendored `nbjw_calib` stays untouched).
- **Config switches for every gated change** (`detector_b.backend`, `calibration.mode`), so a
  regression is a config rollback, not a revert.
- **Re-measure on the target GPU.** The T4 numbers guide the order; the production RTX 30-series
  will shift the balance (faster GPU means CPU stages dominate more, which makes Phase 2 more
  valuable).
- Keep `docs/path_b/plans/IMPLEMENTATION_PLAN.md` §7 and `AGENTS.md` in sync as items land.

## 6. Skills reference

Installed and directly relevant:

| Skill | Use in this plan |
|---|---|
| `ecc:benchmark-methodology`, `ecc:benchmark` | warm-up, percentiles, pipelined-latency definition (Phases 0, 2, 5) |
| `ecc:benchmark-optimization-loop` | measure → change → re-measure cadence (Phases 1, 6) |
| `ecc:latency-critical-systems` | tail latency, async design, stream priorities (Phases 2, 3, 5) |
| `ecc:parallel-execution-optimizer`, `ecc:data-throughput-accelerator` | threading and producer/consumer pipelines (Phase 2) |
| `ecc:pytorch-patterns` | CUDA streams, pinned memory, separable pooling (Phases 1, 3, 6) |
| `senior-computer-vision`, `senior-ml-engineer`, `ecc:mle-workflow` | CV pipeline design, TensorRT deployment, gates (Phases 1, 4, 5, 6) |
| `ecc:python-testing`, `ecc:tdd-workflow`, `tdd-guide` | storm test, parity harness, unit tests (all phases) |
| `ecc:ai-regression-testing`, `ecc:eval-harness` | output-regression gates for exact and gated changes (Phases 0, 4, 5) |
| `ecc:verification-loop`, `code-review` | pre-commit verification and review of each item |
| `ecc:python-patterns` | numpy vectorisation of the decode and descriptor loops (Phases 1, 2) |
| `ecc:deep-research` | TensorRT/HRNet performance research (Phase 6) |
| agent `ecc:pytorch-build-resolver` | CUDA / TensorRT runtime errors on Kaggle |
| agent `ecc:performance-optimizer` | a second pair of eyes on profiling results |

From the open skills ecosystem (`npx skills find`, checked 2026-10-05). Not installed:

| Skill | Installs | Relevance |
|---|---|---|
| `nvidia/skills@deepstream-dev` | 2.4K | NVDEC / GPU video-pipeline design for the production version of 2.2 |
| `mindrally/skills@computer-vision-opencv` | 3.7K | general OpenCV patterns (camera motion, Phase 2.1); low priority |

Searches for "tensorrt", "pytorch performance", "cuda" and "python profiling" returned nothing
else relevant with meaningful adoption.

## 7. Build status

**Phase 0** (2026-10-07)
- 0.1 `notebooks/path_b_eval.py`:
  - the report prints the resolved `calib_every_k`, `mode`, `min_trigger_interval_frames` and
    `fast_decode`;
  - the calibrate row is split into NBJW frames and other frames;
  - the profiler splits decode into peaks / dicts / completion, runs voting only when
    `compute_cam_params` is on, and times the unused decode implementation as a non-summed
    comparison row;
  - the scope note and module docstring are updated.
- 0.2 `LiveCalibrator(min_trigger_interval_frames=3)` (`calibration.min_trigger_interval_frames`).
  It rate-limits `no_motion`, `low_inliers`, `chain_broken`, `zoom` and `max_age`; `first`,
  `scene_cut` and `every_k` are exempt. Test `test_failing_nbjw_on_textureless_shot_is_rate_limited`:
  20 NBJW runs in 20 frames before, 8 after.
  - `max_age` was a second storm path, missed in the original analysis: a failing NBJW never
    refreshes the anchor frame, so the age check fired on every frame.
  - Output-changing in rare cases (an event trigger within 3 frames of the last run now waits).
- 0.3 `ByteTracker.update` steps the tracker on empty frames (`tests/test_track.py`).
  Output-changing; measure its GS-HOTA delta on the next Kaggle run.
- 0.4 `scripts/parity_pipeline.py --check nbjw_decode`.

**Phase 1** (2026-10-07)
- `gsr_pipeline/nbjw_fast_decode.py`, used by `NBJWKeypointDetector.decode` when
  `calibration.fast_decode: true` (default).
- `tests/test_nbjw_fast_decode.py`: identical keypoints, lines and completion vs the vendored
  decode on synthetic heatmaps (including quantised ties and border peaks) and on 50 dense-line
  completions. The closed-form fit matched `linregress` exactly in all cases.
- Real frames, CPU (`parity_pipeline.py`, ORT CPU heatmaps, SNGS-022 + SNGS-033, 20 frames):
  **identical** keypoints, lines and H (worst |dH| 0). Decode on CPU: reference 12.2 s/frame,
  fast 1.6 s/frame.
  - This also explains the old ORT-CPU "13 s/frame NBJW" figure: it was almost all the
    31×31 max-pool on CPU, not the networks.
  - 1.6 s is still slow for the CPU tier (torch's CPU stride-1 max-pool is naive). A
    `scipy.ndimage.maximum_filter` or cumulative-max variant would help that tier only. Not
    pursued: the GPU tiers are the target.
- Pending: the T4 parity + timing run (Kaggle checkpoint 1).
