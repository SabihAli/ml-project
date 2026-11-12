# PIPELINE IMPROVEMENT PLAN 1

## Ball Detection: Recall, Robustness, and Real-Time Performance

**Status:** In progress. Phase 0 closed 2026-11-09, Phase 1 closed 2026-11-10, Phase 4 closed 2026-11-11 (reverted: no tracker beat Phase 1); Phase 3 in progress (3a done 2026-11-12: mining + multi-peak, multi-peak reverted; next: latency profile on the RTX 3090, then teacher-student), then 2 -> 5 -> 6 (§5).\
**Primary objective:** Reach **90%+ ball-detection recall** while
retaining real-time performance within the existing system's end-to-end
latency budget.\
**Current documented baseline:** YOLO26s + c5c specialist achieves
approximately **69.5% recall at 86.5% precision** on the documented
held-out evaluation. **After Phase 1: 76.3% recall at 90.2% precision.** The architecture documents originally targeted 80%
recall, so 90%+ is a substantial new target.

> This is a proposed improvement plan. It is not a claim that the
> proposed changes have already been implemented or benchmarked.
>
> **Revision 2026-11-09:** §2 and §3 were corrected against
> `gsr_pipeline/ball_specialist.py`. The first draft described a Kalman
> ball tracker and CUDA-stream overlap that do not exist yet. The 90%
> target is now conditional on the Phase 0 miss breakdown (§5, §6).

------------------------------------------------------------------------

## 1. Executive summary

The current system is a main-detector + tracker + crop-based specialist
pipeline. Its largest weakness is not necessarily the capacity of an
individual neural network. It is the way the components interact when
the ball becomes difficult to see or moves unexpectedly.

The main detector misses a substantial fraction of small balls. There
is no ball tracker yet. On a miss, the specialist searches a fixed
window centred on the last sighting, with no motion or camera-motion
compensation, and only for up to 15 frames. Any main-detector ball,
including a false one, is accepted and moves that window. If the ball
has left the window, recovery is impossible regardless of how capable
the specialist is. Past 15 frames, nothing searches at all.

The specialist's **conditional activation policy and its
trigger/verification logic should be treated as one design problem**,
not as separate improvements. The system must decide not only when the
specialist is needed, but also when a detector result is suspicious and
should be verified. In particular, a confident but incorrect
main-detector result can currently be more dangerous than a clean miss
if it prevents specialist activation.

The plan therefore prioritizes: 1. Diagnose exactly where recall is
being lost. 2. Make candidate generation and specialist activation
robust to detector and tracker failure. 3. Improve the main detector's
small-object performance. 4. Improve the specialist's temporal
reasoning, training distribution, and output representation. 5.
Strengthen tracking, uncertainty handling, and reacquisition. 6.
Optimize and validate the complete pipeline on the actual target GPU.

A 90%+ result should be treated as a system-level objective. Improving
detector recall alone will not guarantee 90% end-to-end tracking recall.

------------------------------------------------------------------------

## 2. Current system and documented baseline

### 2.1 Main detector

-   Main model: **YOLO26s**.
-   Input resolution: **1280 × 736**.
-   Original broadcast resolution: **1920 × 1080**.
-   The ball is often very small:
    -   10--20 px: documented recall of approximately **56.4%**.
    -   20+ px: documented recall of **100%**.
-   Documented held-out performance:
    -   Main detector alone: **59.4% recall**, **85.3% precision**.
    -   YOLO26s + c5c: **69.5% recall**, **86.5% precision**.
-   Training-clip recall is approximately **84.3%**, compared with
    **59.4%** on held-out clips. This is a substantial generalization
    gap that needs investigation.
-   The detector uses feature strides 8, 16, and 32 in the documented
    configuration.

### 2.2 c5c specialist

-   c5c is a ResNet-18 encoder + U-Net decoder.
-   The exported ONNX takes raw RGB crops of t−2, t−1, t stacked to
    `(1, 9, 640, 640)`, with preprocessing in the graph, and produces a
    320 × 320 heatmap with a subpixel offset. The architecture docs
    describe a five-channel temporal representation. Confirm that the
    in-graph preprocessing reduces the 9 raw channels to those five
    before running any temporal-input experiment (§3.5).
-   It predicts a single point.
-   Its training set contains approximately **51.9k samples**.
-   Frames missed by the main detector were weighted more heavily (×3).
-   c5c recovers approximately **54.5% of the balls missed by the main
    detector**, contributing about **+10.1 percentage points** to
    overall recall.
-   The five-channel setup performed better than the earlier
    nine-channel raw-frame-stack setup in the documented experiments
    (58.5% vs 52.1%), while the nine-channel version was approximately
    1.6× more expensive.
-   The current temporal input uses frames t, t−1, and t−2 at 25 FPS,
    giving only about **120 ms** of history.
-   Specialist performance falls sharply as the gap from the last
    sighting increases:
    -   1 frame: approximately 66%.
    -   2 frames: approximately 18%.
    -   3--5 frames: approximately 9%.
    -   6--10 frames: approximately 4%.
    -   11--15 frames: approximately 1%.
-   **This curve is conditional and should not be read as "the model
    needs more history."** A specialist hit resets the gap to 0, so a
    gap-2 frame exists only when the specialist already failed at gap
    1. Higher-gap buckets therefore hold progressively harder,
    pre-selected cases. With a 640 px window, a ball moving ~20 px/frame
    is unlikely to leave the window within 2 frames, so crop exit
    probably does not explain the drop either. Crop containment by gap
    (§3.2) is what separates these explanations.

### 2.3 Ball fusion and specialist invocation (as implemented)

Source: `gsr_pipeline/ball_specialist.py`, `BallFuser.update`.

-   **There is no ball tracker yet.** The Kalman `BallTracker` from
    `BALL_TEMPORAL_PLAN.md` §7 is not built. The only state is the last
    sighting (centre, frame index, box size).
-   If the main detector returns any ball at its `ball_conf`, the
    **top-confidence ball is accepted unconditionally** and becomes the
    last sighting. There is no trajectory or plausibility check.
-   Otherwise, if the last sighting is ≤ `max_miss_frames` (15) old, the
    specialist runs on a 640 × 640 window **centred on the last
    sighting**, shift-clamped into the frame. There is no motion
    extrapolation and no camera-motion shift.
-   A specialist peak ≥ 0.70 (sidecar threshold) is output and becomes
    the new last sighting.
-   Past 15 frames nothing runs, so recall in long gaps is 0% by
    construction. Documented longest misses reach 9.9 s.
-   The specialist runs **inline, after the detector**, only on frames
    the detector missed. It does not use its own CUDA stream.
-   Runtime tiers (TensorRT FP16, built on first launch → ORT GPU → ORT
    CPU) and the ONNX export already exist. The ~8 ms figure in the
    docs is PyTorch FP16 on an RTX 3090. TensorRT latency on the target
    RTX 3060-class GPU has not been measured; the live budget allocates
    ~5 ms.
-   `BallFuser.update` must stay in step with `simulate` (`--chain
    fused`) in `scripts/ball_eval/eval_ball_temporal.py`. **Every
    policy change in this plan must be made in both places**, or the
    gate-3 numbers stop describing what runs.

### 2.4 Evaluation considerations

-   The documented evaluation uses a 15 px localization tolerance.
-   Some annotations include fully hidden/occluded ball positions.
-   The documented held-out set appears to be a project-level clip
    holdout, but actual clip IDs should be checked before making any
    definitive claim about split leakage.
-   The current 15 px metric can conceal meaningful localization
    differences. Report stricter tolerances as well.

------------------------------------------------------------------------

## 3. Main weaknesses and improvement strategies

## 3.1 Make specialist activation and verification one coherent policy

This section combines the related issues of **conditional specialist
activation** and **the logic that decides whether a detector result
should be trusted or verified**.

The current policy activates the specialist only when the main detector
returns no ball at all. Any detection at `ball_conf`, high-confidence or
not, suppresses the specialist and **moves the last sighting**, so the
next specialist window is centred on the false object. Examples include
a wrong white object, a graphic, or another ball-like object.

**Cheapest first fix:** before accepting the detector's top ball, check
its displacement from the last sighting against a gap-scaled plausibility
bound. If it fails, do not update the last sighting, and run the
specialist on the old window as verification. This needs no new model.

The specialist should be triggered by the reliability of the current
observation and the tracker state---not only by whether a detection
exists.

### Proposed decision policy

  -------------------------------------------------------------------------
  Main detector result      Tracker / scene state   Proposed action
  ------------------------- ----------------------- -----------------------
  High-confidence and       Stable track, low       Accept detection;
  trajectory-consistent     uncertainty             specialist can usually
                                                    be skipped

  High-confidence but       Sudden displacement,    Verify with specialist
  trajectory-inconsistent   implausible velocity,   or alternate candidate
                            or conflict with        evaluation
                            predicted position      

  Low-confidence candidate  Candidate may be the    Run specialist
                            ball but confidence is  verification
                            insufficient            

  No detection, low         Recent reliable track   Search the predicted
  prediction uncertainty    and small expected      region
                            search region           

  No detection, high        Long gap, acceleration, Expand search and
  prediction uncertainty    possible kick, or       invoke recovery
                            uncertain motion        proposals

  Suspected camera cut or   Previous prediction is  Invalidate or
  major scene transition    no longer trustworthy   down-weight the old
                                                    prediction and
                                                    reacquire globally
  -------------------------------------------------------------------------

The exact thresholds should be calibrated from validation data rather
than guessed.

### Scheduling and latency implications

Today the specialist runs inline after the detector, so latency is
additive on miss frames. If it is moved to a parallel stream later,
check whether its trigger can be known early enough. A policy
that depends on the current frame's detector output cannot launch the
specialist before that output is available unless it uses a prior-frame
decision, speculative execution, or a separate early trigger.

Measure: - Specialist invocation rate. - Specialist latency when run
alone and concurrently. - Main-detector latency under concurrent
specialist load. - End-to-end p50 and p95 latency. - GPU utilization and
memory pressure. - Whether speculative invocations waste enough compute
to threaten the frame budget.

Do not assume CUDA streams eliminate contention.

------------------------------------------------------------------------

## 3.2 Stop letting one static window define the specialist's entire field of view

The specialist currently searches a window centred on the **last
sighting**, which stays fixed while the ball moves and the camera pans.
There is no tracker prediction to be wrong yet. The window simply does
not move. The ball can leave it after: - a fast pass or shot; - a camera
pan, zoom or cut (no camera-motion shift is applied); - a false
main-detector sighting that moved the centre (§3.1).

If the real ball lies outside the crop, the specialist cannot recover
it. Increasing the specialist's model capacity will not fix an object
that is not in its input.

**Cheapest first fixes, in order:** 1. Shift the window centre by the
camera motion already estimated in `gsr_pipeline/camera_motion.py`. 2.
Extrapolate by the last observed velocity when two recent sightings
exist. Only add a full Kalman tracker (§3.7) if containment measurements
show these are not enough.

### Proposed recovery strategy

Use multiple sources of candidate regions rather than one predicted
crop:

1.  **Primary predicted crop:** centred on the tracker's expected
    position.
2.  **Expanded uncertainty region:** grows according to predicted
    uncertainty, not just elapsed frames.
3.  **Secondary recovery proposals:** lightweight image-wide or regional
    candidate generation that can find the ball outside the primary
    crop.
4.  **Alternative hypotheses:** preserve plausible candidate locations
    when the track is ambiguous instead of committing immediately to
    one.

The tracker should guide the search, but should not dictate the only
place where the system looks.

### Practical options to test

-   Increase crop size adaptively when uncertainty rises.
-   Use multiple crops or tiles only when the track is uncertain; avoid
    running expensive tiled inference on every frame.
-   Add a lightweight global proposal mechanism for reacquisition.
-   Search a broader region after a suspected kick or sudden trajectory
    change.
-   Keep a primary and one or more secondary motion hypotheses for a
    short period after an ambiguous event.
-   Ensure crop extraction handles image boundaries correctly, including
    padding and coordinate transforms.

Measure **crop containment**: for every specialist invocation, record
whether the ground-truth ball centre was actually inside the crop. This
distinguishes a model failure from a search-region failure.

------------------------------------------------------------------------

## 3.3 Improve the main detector's small-object performance

The size-stratified results strongly suggest that small-ball detection
is a major bottleneck.

Downscaling from 1920 × 1080 to 1280 × 736 reduces a 10 px ball to
roughly 6.7 px, a 15 px ball to roughly 10 px, and a 20 px ball to
roughly 13.3 px. At the smallest sizes, the object occupies very few
feature-map cells.

The documented detector's smallest feature stride is 8. A stride-4
detection head (P2) is worth a controlled test.

### Experiments

1.  **YOLO26s P2 at 1280:** run a properly configured stride-4/P2
    experiment at the same effective input resolution as the current
    YOLO26s baseline.
2.  **Controlled comparison:** compare models using the same data,
    splits, training schedule, augmentation, and evaluation.
3.  **Dedicated ball detector:** test whether a small specialized
    detector can outperform the shared detector while remaining within
    the latency budget.
4.  **Selective high-resolution tiles:** evaluate high-resolution crops
    or tiles only for difficult/uncertain frames, rather than applying
    expensive tiling to every frame.
5.  **Hard-example diversity:** improve coverage of small, blurred,
    occluded, airborne, ground-level, and visually ambiguous balls.

The existing YOLO26n-P2 experiment at 640 is not a fair direct
comparison against YOLO26s at 1280. The P2 test should control for model
capacity and input resolution as much as possible.

Do not assume a P2 head will automatically solve the problem: measure
recall by ball size and the latency/memory cost.

------------------------------------------------------------------------

## 3.4 Improve specialist training distribution and hard-example coverage

The train-versus-held-out recall gap (approximately 84.3% versus 59.4%
for the main detector) suggests that the system is not generalizing well
enough to unseen clips or conditions.

Missed-frame weighting is useful, but weighting frames missed by the
main detector is not the same as training on the failures of the
complete detector--tracker--specialist pipeline.

### Build a failure taxonomy

For each miss, classify the primary cause:

-   Ball is too small.
-   Motion blur.
-   Occlusion or partial visibility.
-   Ball is airborne versus on the ground.
-   Abrupt movement after a kick or deflection.
-   Camera motion, zoom, or cut.
-   Ball lies outside the specialist crop.
-   Tracker has drifted or associated with the wrong candidate.
-   Specialist sees the ball but predicts the wrong location.
-   Long disappearance and failed reacquisition.
-   False association with another white/round object or another ball.
-   Annotation is ambiguous or the ball is fully invisible.

Use this taxonomy to determine which training data and architecture
changes are likely to help.

### Training-data improvements

-   Generate specialist training samples from **actual tracker-generated
    crops**, not only idealized crops centred on ground truth or
    detector outputs.
-   Include examples where the tracker prediction is offset from the
    true ball by realistic and increasingly large distances.
-   Include longer gaps between the last reliable sighting and the
    current frame.
-   Sample difficult held-out-like clips and varied camera/lighting
    conditions.
-   Include hard negatives: white lines, logos, player clothing,
    highlights, crowd details, and other ball-like objects.
-   Add realistic motion blur, compression artifacts, scale variation,
    partial occlusion, and lighting variation where appropriate.
-   Keep the train/validation/test split clip-disjoint and verify exact
    clip IDs.
-   Avoid tuning repeatedly against the final held-out set; retain a
    genuinely untouched final evaluation set.

### Train for the actual inference distribution

The specialist should experience the same kinds of crops and prediction
errors that it will see at runtime. If training crops are too clean or
too accurately centred, the model will be poorly prepared for real
tracker drift.

------------------------------------------------------------------------

## 3.5 Give the specialist more useful temporal context

The current input uses t, t−1, and t−2, which is only about 120 ms of
history at 25 FPS. This can be insufficient when the ball is blurred,
temporarily occluded, or hard to distinguish in a single frame.

**Precondition:** the gap-decay curve in §2.2 is not evidence for this
section (it is selection-biased). Run these experiments only if the
Phase 0 breakdown shows a large share of in-window, below-threshold
misses.

The earlier nine-channel raw-frame-stack experiment performed worse than
the five-channel setup and was more expensive. Therefore, simply
stacking more raw frames is not the recommended next step.

### Suggested experiments

1.  Test a sparse temporal sample such as:
    -   t
    -   t−1
    -   t−3
    -   t−5
    -   t−8
2.  Test explicit motion information, such as frame differences or
    motion channels, rather than blindly adding raw frames.
3.  Evaluate lightweight temporal feature aggregation.
4.  Only after simpler ablations, consider a lightweight ConvLSTM or
    temporal attention mechanism.

Longer temporal context must not force the model to follow stale motion
after a kick. It should learn when older evidence is no longer relevant.

Keep the current five-channel model as a baseline and compare every
temporal change on the same held-out clips and latency conditions.

------------------------------------------------------------------------

## 3.6 Preserve multiple specialist candidates instead of immediately taking one argmax

The specialist already produces a heatmap, but the downstream system
uses a single-point prediction. Taking only the global argmax discards
potentially useful ambiguity information.

### Proposed improvements

-   Extract multiple local heatmap peaks.
-   Retain candidate confidence and spatial separation.
-   Associate candidates with the tracker's predicted position and
    uncertainty.
-   Use temporal consistency to select among candidates.
-   Keep multiple hypotheses briefly when the evidence is ambiguous.
-   Estimate localization uncertainty from heatmap shape, peak
    sharpness, or a separately calibrated uncertainty output.
-   Consider predicting motion direction or displacement where it can
    help association.

This is especially useful when multiple bright/round objects or visual
distractors are present. It also avoids allowing one weak peak to force
an irreversible tracking decision.

------------------------------------------------------------------------

## 3.7 Build the ball tracker, with uncertainty modelling from the start

No ball tracker exists yet. The Kalman `BallTracker` is planned in
`BALL_TEMPORAL_PLAN.md` §7 and as Phase 6 of
`docs/path_b/plans/IMPLEMENTATION_PLAN.md`. It is the only component
that can address misses in gaps longer than 15 frames, where recall is
currently 0% by construction. Build it to the design below rather than
starting with a constant-velocity filter and a linear `r = a + b × gap`
gate and fixing them later.

### Design requirements

-   Use the Kalman filter's predicted covariance to define an
    uncertainty-aware search region, such as an ellipse, rather than
    relying only on a fixed linear radius.
-   Adapt process noise based on motion state.
-   Use a larger process-noise model when a kick, bounce, or abrupt
    acceleration is suspected.
-   Reduce or invalidate confidence in the previous trajectory after a
    scene cut or major camera transition.
-   Support multiple motion hypotheses after ambiguous events.
-   Add explicit reacquisition logic for longer gaps.
-   Distinguish a predicted position from a visually observed position
    in downstream state and evaluation.
-   Prevent long runs of prediction-only positions from being treated as
    confirmed detections.

A tracker prediction is useful evidence, but it is not proof that the
ball is still at that location.

------------------------------------------------------------------------

## 3.8 Validate camera-motion compensation

The documented camera-motion compensation uses affine/optical-flow
estimation at low resolution (approximately 480 × 270). This may be
insufficient in scenes with: - Strong parallax. - Zooms. - Fast pans. -
Large foreground motion. - Cuts or transitions. - Non-uniform motion
across the frame.

### Checks and possible improvements

-   Measure compensation error on representative pans, zooms, and cuts.
-   Detect when the estimated transform is unreliable.
-   Do not apply a bad transform as if it were trustworthy.
-   Reset or down-weight tracker motion after a cut.
-   Consider region-specific or more robust motion estimates only if the
    current affine model is shown to be a meaningful source of errors.

This should be diagnosed before adding a more expensive camera-motion
model.

------------------------------------------------------------------------

## 3.9 Audit candidate handling, thresholds, and crop mechanics

These are implementation-level checks that could reveal avoidable recall
losses.

Inspect: - Confidence thresholds (the documented pipeline includes
thresholds such as 0.15 and 0.70). - Whether low-confidence but
plausible candidates are discarded too early. - NMS behaviour and
whether nearby ball candidates suppress one another. - `max_det=300` and
whether candidate truncation is ever relevant. - Coordinate conversions
between original frame, resized detector frame, crop, and heatmap. -
Crop clipping and padding at image boundaries. - Whether specialist
predictions are mapped back to full-frame coordinates correctly. -
Whether the system preserves enough candidates for downstream
verification. - Whether the tracker gate rejects valid detections after
sudden movement. - Whether a high-confidence false detection can
suppress specialist activation. - Whether the system can reacquire after
long gaps.

Thresholds should be calibrated against recall/precision trade-offs on
the validation set. Do not raise confidence thresholds simply to improve
precision if it further harms recall.

------------------------------------------------------------------------

## 3.10 Make evaluation reflect the actual objective

A single recall number at a 15 px tolerance is not enough to understand
whether the system is improving.

### Report multiple localization tolerances

At minimum, report: - Recall at 5 px. - Recall at 10 px. - Recall at 15
px (retain the existing metric for comparability).

### Break down results by condition

Report recall by: - Ball size. - Visibility and occlusion. - Motion
blur. - Airborne versus ground-level. - Camera motion. - Time since last
reliable sighting. - Whether the ball is inside the specialist crop. -
Whether the main detector found a candidate. - Whether the tracker
prediction was correct or uncertain.

### Separate system stages

Report: 1. Main-detector recall. 2. Specialist recall on the cases it is
given. 3. Recall after detector + specialist candidate fusion. 4.
Tracking recall, including prediction-only intervals as a separately
labelled category. 5. Reacquisition recall after misses. 6. False
association rate.

### Measure temporal continuity

Include: - Longest miss duration. - Mean and p95 recovery time after a
miss. - Fraction of frames with a valid observed ball location. -
Fraction of frames with only a predicted location. - Number and duration
of false tracks/identity switches.

### Handle fully occluded annotations honestly

A fully invisible ball cannot be visually detected from that frame
alone. Report fully occluded cases separately rather than silently
treating them as ordinary detector failures or silently excluding them
from the metric. If the product requirement expects a position during
full occlusion, that is a tracking/prediction requirement, not a
visual-detection requirement.

### Protect evaluation integrity

-   Verify clip IDs across train, validation, and held-out splits.
-   Confirm no near-duplicate frames or overlapping segments cross the
    split boundary.
-   The reviewed documentation does not establish that leakage has
    occurred; this is a validation check, not an accusation.
-   Keep a final untouched test set for the 90%+ claim.

------------------------------------------------------------------------

## 3.11 Optimize the complete pipeline for the target hardware

The documented specialist estimate is approximately 8 ms in PyTorch FP16
on an RTX 3090, while the live allocation is around 5 ms. ONNX export
and TensorRT FP16 engines already exist for both models. What is
missing is a measurement on the target RTX 3060-class GPU.

### Optimization and measurement plan

-   Benchmark the existing TensorRT FP16 engines on the actual target
    GPU.
-   Measure each model alone and under concurrent load.
-   Profile preprocessing, resize, crop extraction, data transfer,
    postprocessing, tracking, and synchronization---not only
    neural-network forward time.
-   Measure specialist invocation frequency and its effect on average
    and p95 latency.
-   Test CUDA-stream overlap under realistic full-pipeline load.
-   Avoid unnecessary CPU↔GPU transfers and synchronization points.
-   Consider batching or reusing buffers where this does not increase
    latency.
-   Run a long live soak test to detect memory growth, timing
    instability, dropped frames, or GPU contention.

The relevant performance measure is end-to-end latency and sustained
throughput, not isolated model inference time.

------------------------------------------------------------------------

## 4. Proposed target architecture

The following is a target design to work toward incrementally. It is not
necessary to implement every component at once.

### Stage A --- Main observations

Run the main detector and retain a useful set of ball candidates,
including confidence and location. Avoid discarding all uncertain
candidates too early.

### Stage B --- Motion prediction

Run the tracker every frame. Produce: - Predicted ball position. -
Predicted uncertainty/covariance. - Motion-state estimate. - Track
confidence. - Scene-transition or camera-motion reliability indicators.

### Stage C --- Candidate generation and specialist policy

Combine: - Main-detector candidates. - Primary predicted crop. -
Uncertainty-expanded search region. - Lightweight recovery proposals
when needed.

Use one unified reliability/activation policy to decide whether to
accept, verify, search locally, expand the search, or reacquire
globally.

### Stage D --- Temporal specialist

Run the specialist when the observation is uncertain or suspicious. The
specialist should: - Use temporal evidence. - Return multiple plausible
heatmap peaks where appropriate. - Preserve confidence/uncertainty. - Be
trained on realistic tracker-generated crops and errors.

### Stage E --- Temporal association and state management

Associate candidates using motion, uncertainty, appearance/context where
useful, and temporal consistency. Maintain multiple hypotheses briefly
when necessary. Distinguish observed positions from predictions.

### Stage F --- Output state

Expose a clear state such as: - `observed_main_detector` -
`observed_specialist` - `predicted_only` - `reacquiring` - `unavailable`

The exact names are implementation choices. The important point is that
downstream consumers can distinguish a measured ball location from a
predicted one.

------------------------------------------------------------------------

## 5. Tentative phased build order

The phases below are ordered to reduce wasted model work. Diagnose the
failure modes before committing to more expensive architectures.

**Execution order (revised 2026-11-10 from the Phase 1 miss breakdown):**
0 -> 1 -> **4 -> 3 -> 2** -> 5 -> 6. After Phase 1 the test misses are
B5 7.0%, B7 6.6%, B3 5.5%, B1 3.6% of ball frames. Phase 4 goes first: it
reuses existing code, is tuned on cached replays (no training), works on
B1 and B3 (a predicted window instead of a stale sighting), fixes the
9.9 s longest misses, and Milestone 3 needs it in any case (predicted
positions through occlusion). Phase 3 follows so the specialist is
retrained on the tracker's real windows (§3.4). Phase 2 is last: B2 is
~0, so a better detector acts only through B7 and < 10 px balls, at the
highest cost.

**Update 2026-11-11:** Phase 4 was reverted (no tracker beat Phase 1; the
misses are detection errors, not association). Remaining order: **3 -> 2** -> 5 -> 6.

## Phase 0 --- Establish a reproducible baseline

**Status: done 2026-11-09** (`docs/results/PIPELINE_IMPROVEMENT_1_PHASE0.md`). Baseline
reproduced (test 69.4% / 86.5%). Test misses as % of ball frames: B7 main false 10.2, B3 false
sighting 7.7, B5 below threshold 7.4, B1 long gap 4.7, B4 outside window 0.0. The miss buckets
were measured by `scripts/pipeline_improvement_1/phase0/phase0_report.py` on the
`diagnose_ball_misses.py` replay rather than inside `eval_ball_temporal.py`. Still open: label
the audit sheets to confirm how many occluded-proxy misses are truly hidden.

**Goal:** Make the current system measurable and establish trustworthy
comparisons.

Tasks: - Re-run the documented baseline on a fixed, clip-disjoint
evaluation set. - Record exact model versions, checkpoints, thresholds,
preprocessing, and configuration. - Verify train/validation/test clip
IDs and check for overlapping segments or near duplicates. - Report
recall at 5/10/15 px. - Produce size-stratified recall, especially for
balls under 20 px. - Report precision, false associations, longest miss,
and p95 recovery time. - Separate main detector, specialist, fused
system, tracker, and reacquisition metrics. - Separate fully occluded
annotations from visually observable cases.

-   **Produce the miss breakdown first.** Assign every held-out miss to
    exactly one bucket:
    1.  Gap > 15 frames (specialist never runs).
    2.  Fully occluded / not visible.
    3.  A false main-detector sighting moved the window.
    4.  Ball outside the specialist window (crop containment fails).
    5.  Ball inside the window, specialist peak below 0.70.
    6.  Ball inside the window, peak above threshold but in the wrong
        place.
    7.  Main-detector miss on a visible ball with no prior sighting.
-   **Compute the achievable ceiling** from the breakdown: the recall
    if buckets 3--6 were fully fixed, and separately if bucket 1 were
    fully fixed. Bucket 2 sets the hard ceiling on visual recall.

All Phase 0 measurement code goes in the evaluator
(`eval_ball_temporal.py`), so it describes the same chain that
`BallFuser.update` runs.

**Deliverable:** A reproducible baseline report, a failure-analysis
dataset, and the miss breakdown with ceilings.

**Exit condition:** Every miss is in one bucket or explicitly marked
ambiguous, and the §6 milestones have been revised from the measured
ceilings.

## Phase 1 --- Fix system-level recovery and trigger logic

**Status: closed 2026-11-10** (`docs/results/PIPELINE_IMPROVEMENT_1_PHASE1.md`). Tasks 1-2
done: plausibility gate (`gsr_pipeline/ball_gate.py`, r0 30, speed 50, conf_hi 0.6,
max_rejects 15) + c5c threshold 0.45, test 76.3% / 90.2% (from 69.4% / 86.5%), 75.6% without
SNGS-050. Milestone 1 reached at its low end; default in `default.yaml`. Task 5 dropped after
Phase 0 (B4 ≈ 0). Not executed, carried forward:
- task 3 (cut detection) -> Phase 4, as part of reacquisition;
- task 4 (transform / clamping / NMS audit) and measuring the gate with the runtime ONNX/TRT
  detector -> Phase 5;
- task 6 (multi-peak / secondary windows; B5 is still 7.0%) -> Phase 3;
- a wider gate grid (conf_hi 0.6 and max_rejects 15 are at the grid edge): optional, ~20 min.

**Goal:** Remove avoidable misses caused by specialist activation, crop
placement, and tracker overconfidence.

Tasks, cheapest first. Each change is made in both `BallFuser.update`
and the evaluator's `simulate`, and is kept only if it improves held-out
recall at no precision loss beyond the guardrail:

1. Plausibility check before accepting the detector's top ball (§3.1).
   Targets B7 + B3 (17.9% of test ball frames).
2. c5c threshold sweep on `select` (B5, 7.4%), on the recall/precision
   curve.
3. Cut detection: on a cut, drop the last sighting.
4. Audit coordinate transforms, window clamping, NMS, and thresholds.
5. ~~Camera-motion / velocity shift of the window centre (§3.2)~~:
   dropped after Phase 0. With a correct last sighting the ball was
   outside the window 2 times in 10,178 test frames.
6. Only if B5 remains large after task 2: multi-peak candidates and
   secondary windows.

**Kill criterion:** any task that gives under +0.5 pp held-out recall,
or costs more than 1 pp precision, is reverted.

**Deliverable:** A more robust recovery pipeline, even before changing
the neural-network architectures.

**Exit condition:** A documented reduction in misses caused by
out-of-crop balls, false high-confidence detections, and tracker-gate
rejection, without violating the latency budget.

## Phase 2 --- Improve main-detector small-ball recall

**Goal:** Improve first-pass recall for the smallest visible balls.

Tasks: 1. Train/evaluate YOLO26s P2 at 1280 under controlled conditions.
2. Compare against the current YOLO26s baseline using identical data and
evaluation. 3. Evaluate a dedicated ball detector if P2 is insufficient
or too expensive. 4. Test selective high-resolution tiles for difficult
frames. 5. Expand hard-example coverage for small, blurred, occluded,
and visually ambiguous balls. 6. Record per-size recall and
full-pipeline latency for every experiment.

**Deliverable:** A measured decision on the best main-detector
configuration.

**Exit condition:** A detector configuration with a demonstrated recall
improvement that fits the end-to-end compute budget.

## Phase 3 --- Improve specialist training and temporal reasoning

**Status: in progress.** 3a done 2026-11-12 (`docs/results/PIPELINE_IMPROVEMENT_1_PHASE3A.md`): the 49 GSR
`test` clips restored to `pool` on Kaggle (SoccerNet-Tracking test, converted); Phase 1 replayed on every split
shows seen clips 92-94% recall vs unseen 76-77% (generalisation, not coverage); mining c5c's own training clips
gives only 1,485 hard positives (cross-fitting needed); multi-peak selection (task 8) REVERTED. Direction
(2026-11-12, user): a large offline teacher pseudo-labels SoccerNet full matches, distilled into real-time
students (main detector with 2 frame-difference channels, c5c retrained, cross-fitted mining); training compute is
not a constraint, inference latency is (§1.4 budget, shared with the future event model). First step: live-path
latency profile on the RTX 3090 (`scripts/pipeline_improvement_1/phase3/profile_latency.py`).

**Goal:** Make specialist recovery work on realistic, difficult pipeline
failures.

Tasks: 1. Generate training crops from actual tracker predictions. 2.
Include realistic prediction offsets and longer gaps. 3. Add hard
negatives and failure-category-balanced sampling. 4. Add realistic blur,
compression, scale, lighting, and partial-occlusion augmentation. 5.
Compare sparse longer temporal sampling (for example t, t−1, t−3, t−5,
t−8). 6. Test explicit motion channels/frame differences. 7. Test
lightweight temporal feature aggregation. 8. Add multiple-peak
extraction and candidate confidence/uncertainty. 9. Consider ConvLSTM or
temporal attention only after simpler temporal ablations. 10.
Re-evaluate on the same held-out clips.

**Deliverable:** A specialist that generalizes better and recovers from
larger tracker errors.

**Exit condition:** Specialist recall improves on difficult held-out
cases, particularly after multi-frame gaps, without unacceptable latency
or false positives.

## Phase 4 --- Build the ball tracker and long-gap reacquisition

**Goal:** Recover misses in gaps longer than 15 frames (bucket 1) and
keep the search region useful when motion is non-linear. Move this phase
ahead of Phases 2--3 if bucket 1 is the largest bucket.

**Status: closed 2026-11-11, REVERTED** (`docs/results/PIPELINE_IMPROVEMENT_1_PHASE4.md`). On
`test`, the robust tracker below reached 73.88% / 88.84% and off-the-shelf ByteTrack 73.86% /
90.18%, OC-SORT 69.57%, BoT-SORT 68.23% (`sort_compare.py`, notebook `phase4b_sort.ipynb`), against
Phase 1's 76.29% / 90.21%; none beat Phase 1 on `select` either. Association is not the bottleneck:
the misses are detection errors (B3, B7, B5). Only the two-model IMM earned its place (+3.2 pp vs one
model); camera compensation and the occlusion model did not. The code stays as a record and is not
wired into the runtime. Original design, as built:
Requirement: robust to occlusion, unpredictable movement and camera
angle changes. Built (`gsr_pipeline/ball_tracker_robust.py`, a new module;
the older `ball_tracker.py` stays for `ball_eval/ball_system.py`):

| requirement | mechanism |
|---|---|
| unpredictable movement | IMM over a calm and a manoeuvre constant-velocity model (tasks 1-3); Mahalanobis gate from the mixed covariance, floored and capped in px |
| occlusion | an unmeasured ball whose prediction lies inside an athlete box is attached to that athlete and follows the box (coast up to 50 frames instead of 10); the specialist keeps searching at the prediction |
| camera angle changes / cuts | per-frame camera-motion compensation of state and covariance; no motion estimate = camera break: covariance inflated, one strong detection restarts the track (tasks 4-5, and Phase 1 task 3) |
| reacquisition | strong detections outside the gate on 2 consecutive frames start a new track; with no track, the strongest detection starts one |
| specialist placement | c5c window centred on the prediction, for up to `max_search_frames` (not the last sighting, not capped at 15) |
| observed vs predicted | output source main / specialist / predicted, and an `occluded` flag (task 6) |

Evaluation (`scripts/pipeline_improvement_1/phase4/phase4_sweep.py`,
notebook `phase4_tracker.ipynb`): main cache re-built with athlete boxes,
knobs by coordinate descent on `select`, frozen on `test` against the
Phase 1 system; ablations without occlusion, without camera motion and
with one motion model; recall by condition (occluded, abrupt, post-cut);
measured and +predicted outputs scored separately (task 7). Runtime wiring
into `BallFuser` follows only if the tracker passes the kill criterion.

Tasks: 0. Build the `BallTracker` (§3.7). 1. Use predicted covariance
for uncertainty-aware gating. 2.
Adapt process noise for suspected kicks, bounces, and abrupt changes. 3.
Add short-lived multiple motion hypotheses for ambiguous events. 4.
Implement explicit long-gap reacquisition. 5. Validate camera-motion
compensation and reject unreliable transforms. 6. Make
observed-versus-predicted status explicit throughout the pipeline. 7.
Evaluate tracking continuity and false association separately from
detector recall.

**Deliverable:** A tracker that can guide recovery without making the
system blind to unexpected motion.

**Exit condition:** Better recovery after abrupt motion and long gaps,
with fewer false associations.

## Phase 5 --- Deployment optimization and live validation

**Goal:** Meet the real-time requirement on the actual deployment
hardware.

Tasks: 1. (Done: ONNX/TensorRT export and runtime tiers exist.) 2.
Benchmark the TensorRT FP16 engines on the target RTX 3060-class GPU. 3. Profile each stage and the
complete pipeline. 4. Measure concurrent detector/specialist performance
and CUDA-stream contention. 5. Tune specialist invocation rate and
recovery workload. 6. Optimize preprocessing, crop extraction,
transfers, and postprocessing. 7. Run a sustained live test with
multiple representative camera feeds. 8. Track p50/p95 latency,
throughput, dropped frames, GPU memory, and recall.

**Deliverable:** A deployment-ready pipeline with reproducible accuracy
and latency results.

**Exit condition:** Sustained real-time operation within the agreed
end-to-end latency budget, with no hidden degradation under concurrent
load.

## Phase 6 --- Final 90%+ validation

**Goal:** Establish that the system actually meets the new objective.

Tasks: - Freeze the model, thresholds, and pipeline configuration. -
Evaluate on the untouched final test set. - Report overall recall and
precision. - Report recall at 5/10/15 px. - Report recall by size,
visibility, motion, and recovery-gap category. - Report false
association rate and temporal continuity. - Report p50/p95 end-to-end
latency and sustained throughput on target hardware. - Compare against
the original baseline.

**Deliverable:** A final evaluation report supporting or rejecting the
90%+ claim.

**Exit condition:** At least 90% recall on the explicitly defined
evaluation population and localization tolerance, while meeting the
real-time performance requirement. The evaluation must state how fully
occluded frames are handled.

------------------------------------------------------------------------

## 6. Tentative milestones

These are engineering milestones, not guaranteed outcomes. **Revised
2026-11-09 from the Phase 0 measurements**
(`docs/results/PIPELINE_IMPROVEMENT_1_PHASE0.md`). All numbers are test
recall at 15 px on all ball-labelled frames. Report every result with and
without SNGS-050, which overlaps a `pool` clip (worth at most 0.6 pp).

How the targets were derived:
- Rejecting a false ball does not output the right one; it only hands
  the frame to the specialist, which finds the ball on about 44% of
  in-window calls. So about half of each B3/B7 ceiling is realistic.
- The occluded-proxy misses are 16.5% of test ball frames. Fixing every
  other miss gives 83.5%. 90% on all ball frames therefore needs some
  hidden balls to be recovered by prediction (the tracker), not by
  seeing them.

  ---------------------------------------------------------------------------------
  Milestone         Recall   Precision   Basis
  --------------- -------- ----------- ----------------------------------------------
  Baseline           69.4%       86.5%  measured (Phase 0)

  Phase 1 result     76.3%       90.2%  measured 2026-11-10: Milestone 1 reached

  Milestone 1      ~76-78%        ≥86%  Phase 1: plausibility check (half of
                                        B3 + B7 ≈ +6-9 pp) and threshold sweep

  Milestone 2      ~80-83%        ≥86%  Phase 3/4: B5 and B1 (7.4 + 4.7 pp),
                                        bounded by the 83.5% non-occluded ceiling

  Milestone 3          90%        ≥86%  only on a restated population: visible balls
                                        (after the audit), or all frames with the
                                        tracker counting predicted positions

  Deployment     Milestone 3   agreed   sustained on target hardware
  acceptance       target   guardrail
  ---------------------------------------------------------------------------------

Before Milestone 3 can be fixed, label the audit sheets
(`reports/diagnosis/audit/`, `human_cause`) to measure how many
occluded-proxy misses are truly hidden. If that share is above ~10% of
ball frames, restate the 90% target on visible balls only.

Expect a trade-off, not both targets at once. Verification, wider
windows, secondary searches and long-gap reacquisition all add
false-positive opportunities. Report every change as a point on a
recall/precision curve, and do not assume ≥86% precision holds at 90%
recall.

Precision should be treated as a guardrail, not optimized in isolation
at the expense of recall. If the product's acceptable false-positive
rate differs from the provisional 86% precision guardrail, revise it
explicitly before final acceptance.

The real-time criterion should be defined in terms of the complete
pipeline. The existing documentation references a p95 end-to-end target
around 25 ms; confirm that this remains the required budget and measure
it under realistic concurrent camera load.

------------------------------------------------------------------------

## 7. Prioritization: what to do first

If implementation time is limited, use this order:

1.  ~~Produce the miss breakdown and ceilings (Phase 0).~~ Done
    2026-11-09.
2.  **Done 2026-11-10 (Phase 1): plausibility-check detector balls before accepting them.** A false
    detection must not silently move the window (B3 + B7, the largest
    share of misses).
3.  **Done 2026-11-10 (Phase 1): sweep the c5c threshold** (B5; 0.45). Cheap, and needs no new model.
    (Shifting the window by camera motion or velocity is dropped: B4 ≈ 0.)
4.  **Next (in progress): build the ball tracker** (Phase 4, now before Phases 2-3) for gaps over 15 frames (B1, 3.6% after Phase 1), and
    for predicting hidden balls if Milestone 3 counts predictions.
5.  **Run a controlled YOLO26s P2@1280 experiment** if B7 and the
    < 10 px misses remain large after step 2.
6.  **Train on realistic runtime windows and failure cases.**
7.  **Temporal inputs and multi-peak outputs**, only if buckets 5/6 are
    large.
8.  **Measure the existing TensorRT engines on the target GPU**, and
    validate the complete system.

Do not start by simply making the specialist backbone larger. First
establish whether its failures are caused by poor visual recognition,
bad crop placement, bad activation decisions, or bad downstream
association.

------------------------------------------------------------------------

## 8. Open questions to resolve during implementation

These need code inspection or targeted experiments; the architecture
notes alone do not settle them.

Resolved from code (`ball_specialist.py`): the specialist runs inline
after the detector; it triggers only when the detector returns no ball
and the last sighting is ≤ 15 frames old, and it accepts peaks ≥ 0.70.
Any detector ball, correct or not, suppresses it.

Still open:

-   Does the in-graph preprocessing reduce the 9 raw channels to the
    documented five-channel representation?
-   What fraction of specialist misses have the ground-truth ball
    outside the crop?
-   How are crop boundaries, padding, and coordinate transforms handled?
-   Are multiple detector candidates retained through NMS and tracker
    association?
-   Does the tracker gate reject valid detections after kicks or abrupt
    motion?
-   How reliably are camera cuts, zooms, and unreliable optical-flow
    estimates detected?
-   What are actual TensorRT FP16 latencies on the target RTX 3060-class
    GPU?
-   What is the full pipeline's p95 latency with all camera feeds
    active?
-   Are fully occluded annotations included in the headline recall, and
    if so, what behaviour is expected from the system during those
    frames?
-   Are the exact train/validation/test clip IDs mutually exclusive?

------------------------------------------------------------------------

## 9. Implementation files to inspect next

To turn this plan into concrete code changes, inspect the implementation
corresponding to:

-   `ball_tracker.py`
-   `ball_specialist.py`
-   `ball_crop.py`
-   `detect.py` or the main detector implementation
-   Specialist training/data-generation scripts
-   Evaluation scripts and split definitions
-   Camera-motion compensation implementation
-   Runtime scheduling / CUDA-stream orchestration

The next code review should trace one difficult miss end to end:

1.  Ground-truth ball location in the source frame.
2.  Main detector candidates and confidence scores.
3.  Tracker prediction and covariance.
4.  Specialist activation decision.
5.  Crop coordinates and whether the ball is inside the crop.
6.  Specialist heatmap and candidate peaks.
7.  Candidate association and final output state.
8.  Per-stage timing.

This trace will identify whether the miss is primarily a detection,
activation, crop, temporal-reasoning, tracking, or scheduling failure.

------------------------------------------------------------------------

## 10. Final guiding principle

**The system must be able to recover when its own prediction is wrong.**

The detector should find the ball when it is visible. The tracker should
narrow the search when its prediction is reliable. The specialist should
verify uncertain observations and recover difficult cases. But no single
detector output, tracker prediction, crop, or heatmap peak should be
allowed to make the system permanently blind to the actual ball.

The path to 90%+ recall is therefore a coordinated improvement to
candidate generation, specialist activation, small-object detection,
temporal evidence, uncertainty-aware tracking, reacquisition,
evaluation, and deployment performance---not simply a larger neural
network.
