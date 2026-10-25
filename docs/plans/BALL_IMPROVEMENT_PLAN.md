# Ball detection to 80% recall: implementation plan

Status (2026-10-25): **approved; items 1–3 built, not yet run.** Four build items, run in order.
Items 1 and 2 need no retraining and run on existing caches. Items 3 and 4 need the RTX 3090 or
Kaggle. **Items 1–3 run on Kaggle** from one generated notebook: `docs/guides/BALL_IMPROVEMENT_KAGGLE_GUIDE.md`
(runner `notebooks/kaggle/ball_pipeline.py`, notebook built by `scripts/kaggle/build_ball_notebook.py`).
Item 4 waits for item 1's numbers (user, 2026-10-25).

**Built as (2026-10-25), and where it differs from the text below:**
- Item 1: `scripts/eval/ball_det/diagnose_ball_misses.py`; camera motion offline from
  `scripts/ball_temporal/camera_motion_cache.py` (masks GT people, not detections: slightly
  cleaner than the runtime).
- Item 2: `gsr_pipeline/ball_tracker.py` (pure numpy, `tests/test_ball_tracker.py`); evaluated by
  `eval_ball_temporal.py track` (`scripts/eval/ball_det/ball_system.py`). The batch path is a
  dynamic-programming pick over candidates plus camera-compensated interpolation; **RTS smoothing
  is left out** until positions, not picks, are shown to be the problem. **Not wired into
  `GSRPipeline` yet**: that follows gate T.
- Item 3: window variants (§4.1–4.2) are searched inside the `track` stage; the dataset builder
  has `--centre predicted` and `--gap2_repeat`; the trainer has `--init`, 2-GPU DDP and
  `--time_budget_h`; the exporter has `--batch`. The runtime loads batch-1 ONNX only, so the
  3-window variant needs runtime work if it wins.

Related: `docs/path_b/plans/BALL_TEMPORAL_PLAN.md` (the c5c specialist, its gates, §7 tracker
sketch), `docs/path_b/plans/IMPLEMENTATION_PLAN.md` §5 (runtime ball design),
`docs/architecture/BALL_TEMPORAL_ARCH.md` (c5c write-up), `docs/path_b/PERF_LOG.md` (numbers).

Image sizes in this document are **width × height**.

---

## 0. Goal and where we are

**Goal:** ball recall **≥ 80%** on the 15 held-out test sequences, with precision no lower than
today's 86.5%. Same scoring as gate 3: a 15 px centre match on every ball-labelled frame.

| | recall | precision | source |
|---|---|---|---|
| main detector alone (15 test seqs, 1280 cache) | 59.4% | 85.3% | PERF_LOG 2026-10-16 |
| main + c5c (15 test seqs, simulated) | 69.5% | 86.5% | PERF_LOG 2026-10-16 |
| main alone, live runtime at 1280×736 (4 local test seqs) | 56.8% | 88.2% | 2026-10-25 run |
| main + c5c, live runtime (same 4 seqs) | 64.3% | 88.5% | 2026-10-25 run |

What the numbers say:

- **Most remaining misses are runs, not single frames.** c5c's recall by gap since the last
  sighting is 66% at 1 frame, then 18%, 9%, 9%, 4%, 1%. Once both models miss twice, nothing
  recovers the ball. Longest miss: 2.7 s mean, 9.9 s max per sequence, unchanged by c5c.
- **The main detector generalises poorly, it isn't out of capacity.** Recall on its own training
  sequences is 84.3%, on new footage 59–70%.
- **The ball is small for the detector.** Frames are scaled from 1920×1080 to 1280×720 (the
  training scale, `imgsz: 1280`), so a 10–20 px ball is 7–13 px on a stride-8 grid with no
  stride-4 head.
- **Some "misses" may not be misses.** Spare balls by the boards are unlabelled (plan
  decision 11), so a share of false positives are real balls, and a share of labelled balls may
  be fully hidden behind players.

**Open question for the user (not blocking items 1–2):** does 80% have to hold **live**, or is
**batch** enough? Batch can interpolate between sightings on both sides of a gap; live can only
coast forward. Every item reports both.

---

## 1. Order of work and gates

| # | item | where | needs | done when |
|---|---|---|---|---|
| 1 | miss diagnosis | local CPU | main cache, GT | per-cause table + ceiling estimate in PERF_LOG |
| 2 | ball tracker (live + batch) | local CPU, then 3090 for the system run | item 1's displacement stats | gate T passes (§3.8) |
| 3 | c5c past the first missed frame | 3090 | item 2 (predicted centres) | gate S passes (§4.6) |
| 4 | main detector retrain | Kaggle 2×T4 or 3090 | item 1 (says whether it's worth it) | gate D passes (§5.7) |

Item 1 decides how far 2–4 go. If the diagnosis shows the reachable ceiling is below 80% for
detection (e.g. 25% of labelled balls are fully hidden), the target moves to the tracker, and
item 4 may not be worth its cost. That call is made with the user after item 1, not assumed.

After each item: re-run gate 3's system evaluation on the 15 test sequences, record a dated
PERF_LOG entry, and keep thresholds chosen on `select` only (never on `test`).

---

## 2. Item 1: miss diagnosis

**Why first:** it costs a day on CPU and tells us which of items 2–4 can actually move recall.

### 2.1 Script

`scripts/eval/ball_det/diagnose_ball_misses.py`, standalone like the other evaluators (imports
`scripts/ball_temporal/gsr_io.py` and the evaluator's `simulate`). Inputs:

- the main-detector cache (`cache_main_detections.py` output: up to 5 balls per frame at
  conf ≥ 0.05, `[cx, cy, w, h, conf]`);
- GT from `Labels-GameState.json` (ball box, every person box with role);
- c5c per-frame outputs: run `simulate` (fused chain, threshold 0.70) once and save
  `{seq, frame, x, y, conf, src, gap, x0, y0}` per frame to `spec_outputs.jsonl`, so the
  diagnosis and item 2 can replay it without a GPU.

Splits: `select` (for setting rules) and `test` (for reporting). Rules are fixed on `select`.

### 2.2 Per-miss causes

A **miss** is a ball-labelled frame with no output within 15 px. Each miss gets every cause
that applies (they overlap), plus one **primary** cause by the priority order below:

| cause | rule | what it implies |
|---|---|---|
| `near_miss` | an output within 15–30 px of GT | localisation, not detection: fix in decode/box centre |
| `low_conf` | a cached main candidate within 15 px at conf 0.05–0.15 | the tracker (item 2) can recover it by gating |
| `occluded` | GT ball centre inside a GT person box (expanded 10%) | likely unrecoverable by detection; tracker territory |
| `outside_window` | c5c ran, but GT ball was outside its 640×640 window | item 3 (window placement) |
| `not_attempted` | c5c didn't run: no sighting yet, or gap > 15 | item 2 (tracker keeps the track) and item 3 |
| `fast` | GT ball displacement > 25 px/frame (blur) | item 3 (motion blur, training) and item 4 |
| `small` | GT ball box max side < 10 px | item 4 (resolution, stride-4 head) |
| `static_restart` | GT ball moves < 1 px/frame for ≥ 25 frames (kickoff, set pieces) | item 4 (data), or a static-ball rule |
| `detector_blind` | none of the above: visible, in window, slow, normal size, nothing near | item 4 |

Priority for the primary cause: `near_miss` > `low_conf` > `occluded` > `outside_window` >
`not_attempted` > `fast` > `small` > `static_restart` > `detector_blind`.

Also report, for false positives on ball-labelled frames: distance to GT, and whether the
output sits beyond the pitch edge (likely a spare ball, using the frame's GT pitch lines where
present).

### 2.3 Displacement statistics (input to item 2)

From GT, per gap `g` = 1…25 frames: the distribution of ball displacement in pixels between
frame `t` and `t+g`, raw and after subtracting camera motion (`gsr_pipeline/camera_motion.py`
on the frames). Report p50/p90/p99. Item 2's gating radius is the p99 curve, fitted as
`r(g) = a + b·g` (or a table if the fit is poor).

### 2.4 Manual audit

200 misses sampled uniformly from `test`, written as a contact sheet (`audit/*.jpg`: the
640×640 window, GT box, outputs, cause tags). One pass by eye to check the automatic causes,
especially `occluded` (a ball in front of a player is not hidden) and spare balls. Record the
agreement rate; if a rule is below ~80% agreement, fix it on `select` and re-run.

### 2.5 Ceiling estimate

`ceiling = current recall + share of misses whose primary cause is addressable`, per item:
`low_conf` + `not_attempted` → item 2; `outside_window` → item 3; `fast` + `small` +
`static_restart` + `detector_blind` → item 4; `occluded` → item 2 only in batch (interpolation).
Reported separately for live and batch.

### 2.6 Done when

PERF_LOG entry with the per-cause table (counts and shares, `select` and `test`), the
displacement table, the audit agreement rate and the ceiling estimate per item.

---

## 3. Item 2: ball tracker

Builds `BALL_TEMPORAL_PLAN.md` §7 and `IMPLEMENTATION_PLAN.md` §5.4. Replaces `BallFuser`'s
"top ball or specialist" choice with a tracker that picks among many candidates.

### 3.1 Module and interface

`gsr_pipeline/ball_tracker.py`:

```python
@dataclass
class BallCandidate:
    x: float; y: float; conf: float
    source: str                  # "main" | "specialist"
    w: float = 0.0; h: float = 0.0

@dataclass
class BallEstimate:
    x: float; y: float; conf: float
    source: str                  # "main" | "specialist" | "predicted"
    track_id: int
    frames_since_seen: int

class BallTracker:
    def update(self, frame_idx: int, candidates: list[BallCandidate],
               camera_shift: tuple[float, float] | None) -> BallEstimate | None: ...
    def predicted_position(self, frame_idx: int) -> tuple[float, float] | None: ...
    def reset(self) -> None: ...
```

`BallFuser` stays as the candidate provider: it collects main candidates (now all balls above
the lower threshold, §3.3) and runs c5c on a miss, then hands the list to `BallTracker`. The
pipeline wiring in `_detect_track_calibrate` changes only in what `BallFuser.update` returns.

### 3.2 State and motion

- Constant-velocity Kalman filter in **image space**: state `(x, y, vx, vy)`, measurement
  `(x, y)`.
- **Camera motion compensation:** before predicting, shift the state by the frame's camera
  motion (the estimate `LiveCalibrator` already computes via `camera_motion.py`). The pipeline
  passes the per-frame shift into `update`. Without it, a pan looks like ball motion.
- Process noise from item 1's displacement stats (the residual after camera compensation);
  measurement noise per source (main boxes vs specialist points), set on `select`.

### 3.3 Candidates

- **Main detector:** lower its ball floor from 0.15 to **0.05** for the tracker (config
  `detector.ball_conf`), keep up to 5 per frame (as the cache does). The tracker's gating is what
  rejects the extra false positives; precision is checked at the gate.
- **Specialist:** its peak when it ran, at a lower floor than today's 0.70 (e.g. 0.3), since the
  tracker gates it. The floor is chosen on `select`.
- Each candidate's confidence enters the association cost; specialist confidences are
  calibrated against main's on `select` (a simple per-source scale) so they are comparable.

### 3.4 Association, gating, birth, coasting

Per frame:

1. Predict (after camera shift). Gate radius `r(g)` from §2.3, where `g` = frames since the last
   accepted measurement.
2. Candidates inside the gate: pick the lowest cost
   `d / r(g) − λ·conf` (λ set on `select`). Prefer main over specialist at equal cost.
3. Accept → Kalman update, `frames_since_seen = 0`, output `source = main|specialist`.
4. Nothing in the gate → **coast**: output the prediction with `source = predicted` for up to
   `max_coast_frames` (start 10, set on `select` from item 1's gap stats), then drop the track
   and output nothing.
5. **Track restart:** a strong candidate (conf ≥ `restart_conf`, start 0.5) outside the gate on
   **2 consecutive frames** within `r(1)` of each other starts a new track (cut, long pass). A
   scene cut (`LiveCalibrator`'s `scene_cut` trigger) resets the tracker immediately.
6. No track and no strong candidate: output nothing.

`track_id` increments on every new track (it replaces today's constant 90000). Check
`possession.py` and `data_ingestion`'s `CVPipelineSource` when wiring: both should key the ball
by role, not id.

### 3.5 Batch-only additions

`process_sequence` has the whole sequence, so after the frame loop:

- **Global path:** a Viterbi pass over every frame's candidates (plus a "no ball" state per
  frame), with transition cost from the displacement model and emission cost from confidence.
  This replaces the greedy online choice; the online tracker still runs per frame to place c5c
  windows.
- **Smoothing:** Rauch–Tung–Striebel over each track.
- **Gap interpolation:** between two accepted measurements at most `max_interp_frames` apart
  (start 25), fill the gap along the smoothed path, `source = interpolated`.

### 3.6 Where c5c's window goes

`BALL_TEMPORAL_PLAN.md` decision 6 kept the window on the last *observed* position "until the
tracker exists". Item 2 adds a flag `ball_specialist.window_centre: last_seen | predicted`,
default `last_seen`. Item 3 evaluates `predicted`; the default changes only if it wins on
`select`.

### 3.7 Evaluation

New `track` stage in `scripts/eval/ball_det/eval_ball_temporal.py`:

- **Replay mode (CPU):** from the cache and item 1's `spec_outputs.jsonl`. Exact only while
  the window centre is `last_seen` and the specialist's call pattern is unchanged; used for
  fast tuning of gating, λ, coasting and restart on `select`.
- **Full mode (3090):** runs c5c on the windows the tracker actually places. Required for the
  gate and for any `predicted` window centre.
- Reports recall and precision **three ways**: measured only; + coasted; + interpolated (batch).
  Coasted and interpolated outputs count as hits only within 15 px, like any other output.
- Also: longest miss (mean and max), recall per gap bucket, ms per frame for the tracker.

### 3.8 Tests and gate

- Pytest `tests/test_ball_tracker.py`, pure logic: prediction with camera shift, gate growth
  with gap, cost ordering (main preferred at a tie), coasting then drop, 2-frame restart, scene
  cut reset, Viterbi on a hand-built candidate lattice, interpolation bounds.
- Config section `ball_tracker:` with every tunable above and where its value came from.
- Wire field: the ball row's `ball_source` gains `predicted` (live) and `interpolated` (batch);
  `api/event_serializer.py` already passes `ball_source` through. `data_ingestion` maps by role,
  so no change there, but tell its owners that `predicted` positions are estimates.
- **Gate T** (15 test sequences): live recall (measured + coasted) above gate 3's 69.5%,
  precision ≥ 86.5%, longest miss shorter; batch numbers reported alongside. Tracker cost
  < 1 ms per frame.

---

## 4. Item 3: c5c past the first missed frame

Recall falls from 66% at gap 1 to 18% at gap 2. The causes come from item 1
(`outside_window`, `fast`, `occluded`); the levers are below. Each is an A/B on `select`
against c5c as wired today.

### 4.1 Window placement

- **Predicted centre** (from §3.6): window centred on the tracker's prediction instead of the
  last sighting. Needs the training crops to match (lesson 4 of `BALL_TEMPORAL_PLAN.md` §1):
  if it wins, rebuild the dataset (§4.3) with crops centred the same way.
- **More than one window from gap 2:** c5c's input is fixed at 640×640 and it was trained at
  1:1 scale, so the window can't simply grow (downscaling a larger window changes the ball's
  size the model learned). Instead, from gap 2 run **3 windows** (centred on the prediction and
  shifted ± half a window along the predicted velocity) as one batch of 3. Needs a re-export
  with batch 3 (static) or a dynamic batch dimension, and a new engine. Pick the highest peak
  across windows.
- Cost: 3× the specialist call on gap ≥ 2 frames only (≈ 8 ms per call in PyTorch FP16 on the
  3090 today; TensorRT timing not yet recorded, `BALL_TEMPORAL_PLAN.md` §10 item 17).

### 4.2 Run on low-confidence main frames too

Today c5c runs only when main has no ball ≥ 0.15. With the tracker, also run it when main's best
gated candidate is below 0.3, and let the tracker choose. Measured as recall/precision change
and extra calls per frame.

### 4.3 Training data

- **Centring:** if §4.1's predicted centre wins, rebuild with
  `build_ball_temporal_dataset.py --centre predicted` (new flag: the centre is the tracker's
  prediction replayed from the cache, with the same ±48 px jitter fallback).
- **More gap ≥ 2 positives:** oversample hard frames at gap ≥ 2 (×2 on top of today's ×3) so the
  model sees more balls far from the window centre.
- **Occlusion:** item 1's `occluded` misses that a person can see in the audit (ball partly in
  front of a player) are kept as positives; fully hidden ones are excluded from training
  positives (they teach the model to fire on players).

### 4.4 Training

- `train_ball_temporal.py --channels 5 --motion_blur 0.25 --epochs 60`, warm-started from
  `c5c/best.pt` (c5c's `select` score was still rising ~0.1 per epoch at epoch 30, learning
  rate at its floor). Restart the schedule from a lower peak (e.g. 0.3× the original) for the
  warm start.
- Same selection score (recall averaged over 1–10% negative FPR on `select`).
- Name the run `c5d` (or `c6*` if the data changes); never overwrite `c5c`.

### 4.5 Export and wiring

`export_ball_specialist.py` with the new checkpoint (and batch 3 if §4.1 is adopted), parity
check against PyTorch (point diff < 0.5 px, conf diff < 1e-3), new sidecar threshold chosen on
`select` with the `spec_precision` rule. `ball_specialist.onnx_path` switches only after gate S.
TensorRT FP16 latency recorded on the 3090 (closes `BALL_TEMPORAL_PLAN.md` gate 4).

### 4.6 Gate S

On the 15 test sequences with the item 2 tracker: recall at gap 2–5 at least doubled from 18%,
system recall up, precision ≥ 86.5%, specialist ≤ 5 ms per call on TensorRT FP16 (3090).

---

## 5. Item 4: main detector retrain

Only after item 1 shows enough `small`, `fast`, `static_restart` or `detector_blind` misses to be
worth days of GPU. Two levers: more varied data, and a resolution that suits a 10–20 px ball.

### 5.1 Data

Today the detector trained on 57 GSR `train` sequences. Available and not yet used for it:
33 non-test, non-select `valid` sequences and the 49 GSR `test` sequences (both now in c5c's
`pool`).

**Conflict:** c5c's hard examples come from footage the main detector never saw. Training the
detector on `pool` removes that source. Resolve with **2-fold hard-example mining**:

1. Split `pool` into halves A and B by sequence.
2. Train detector `D_A` on `train` + A, `D_B` on `train` + B (same recipe).
3. Mine c5c's hard examples on B with `D_A`, and on A with `D_B` (cache step re-run per fold).
4. The shipped detector `D` trains on `train` + A + B. c5c retrains on the mined examples.

`select` (10) and `test` (15) stay out of every detector's training.

### 5.2 Resolution

Three variants, A/B'd on `select` ball recall, people recall and latency:

| variant | training | export (W×H) | extra cost vs today (estimate) |
|---|---|---|---|
| R0 (today's recipe, more data) | `imgsz 1280` | 1280×736 | none |
| R1 native scale | `imgsz 1920` | 1920×1088 | ~2.2× detector compute, ≈ +19 ms/frame on the 3090 |
| R2 stride-4 head | `imgsz 1280`, model yaml with a P2 (stride 4) detect layer | 1280×736 | ~1.3–1.6× (to be measured) |

- R1 must be trained at 1920; running today's 1280-trained weights at 1920 is a scale mismatch
  (1.5×, at the edge of the ±50% scale augmentation) and is only a quick probe, not a candidate.
- R2: start from `yolo26s.pt` weights for the shared layers; the P2 head trains from scratch.
  Check the installed Ultralytics supports a P2 variant of YOLO26 before committing; if not,
  R2 falls back to the YOLOv8/11 `-p2` yaml at the same size and is compared like-for-like.
- Memory: R1 at batch 16 needs 2×T4 DDP, or the 3090 at batch 8 with gradient accumulation.

### 5.3 Ball-focused training settings

- `copy_paste` of balls (deferred since 2026-10-12): paste GT ball crops at random pitch
  positions, 0.3 probability, sizes from the GT size distribution.
- Ball class weight: raise `cls` weight for the ball, or oversample ball-heavy mosaics, A/B on
  `select`.
- Keep `scale: 0.5`, `mosaic: 1.0`, `close_mosaic: 10` from the current run (`args.yaml`).

### 5.4 Training script

`scripts/train_yolo26_gsr.py` gains: `--extra_splits` (adds `valid`-non-test-non-select and
GSR `test` sequences), `--fold A|B` (the §5.1 halves, written to a split file so folds are
reproducible), `--model_yaml` (for R2), and records the split file and git commit in the run
directory. Default `--imgsz` changes from 640 to 1280 so the default matches what was actually
trained.

### 5.5 Export, runtime and parity

- `scripts/export_athlete_detector.py --imgsz 1088 1920` for R1 (the script takes height then
  width), new sidecar, `scripts/build_trt_engine.py`, then the parity gate
  (`scripts/parity_gate.py`) on ORT CPU and TensorRT FP16.
- `detector.onnx_path` switches only after gate D.

### 5.6 Knock-on work

A new main detector changes which balls it misses, so c5c (trained on the old detector's
misses) must be re-validated: re-run the main cache with the new detector, rebuild the c5c
dataset (§5.1 step 3), retrain c5c, re-select its threshold on `select`, re-run gate 3 and
gate T. Budget this as part of item 4, not after it.

### 5.7 Gate D

On the 15 test sequences: main-detector ball recall up at least 8 points with precision no
lower; people recall no lower than today's on the same frames; latency recorded. Then the full
system (detector + c5c + tracker) re-measured: this is the number that answers the 80% goal.

---

## 6. Live versus batch

Every report carries both:

- **Live:** measured outputs + coasted predictions (`max_coast_frames`), no look-ahead.
- **Batch:** Viterbi path + RTS smoothing + interpolation.

If only batch reaches 80%, the options for live are a fixed output delay (emit frame `t` at
`t + k`, so a short look-ahead can fill gaps up to `k` frames) or accepting lower live recall.
That is a product decision for the user, raised with item 2's results.

---

## 7. Risks

- **The ceiling may be below 80% for any detector.** Item 1 exists to find this out before
  spending GPU time.
- **Coasting and interpolation can hide misses.** They count only within 15 px, and every
  report gives measured-only numbers next to them.
- **Tuning on test.** Every threshold, radius and λ is set on `select`. A rule changed after
  seeing test results is recorded as such, as gate 3's second rule was.
- **Latency.** R1 (≈ +19 ms on the 3090) and 3-window c5c only fit 25 fps live with the threaded
  Phase 2 pipeline (`IMPLEMENTATION_PLAN.md` §12), which is not built.
- **Label noise.** Unlabelled spare balls lower measured precision; the audit (§2.4) estimates
  how much.

---

## 8. Files touched

| file | change |
|---|---|
| `scripts/eval/ball_det/diagnose_ball_misses.py` | new (item 1) |
| `scripts/eval/ball_det/eval_ball_temporal.py` | save per-frame outputs; `track` stage (item 2) |
| `gsr_pipeline/ball_tracker.py` | new (item 2) |
| `gsr_pipeline/ball_specialist.py` | candidate list out; `window_centre`; multi-window batch (items 2–3) |
| `gsr_pipeline/pipeline.py` | camera shift into the tracker; batch Viterbi/interpolation pass (item 2) |
| `gsr_pipeline/configs/default.yaml` | `ball_tracker` section; `detector.ball_conf` 0.05 with the tracker |
| `tests/test_ball_tracker.py` | new (item 2) |
| `scripts/ball_temporal/build_ball_temporal_dataset.py` | `--centre predicted`, gap ≥ 2 oversampling (item 3) |
| `scripts/ball_temporal/train_ball_temporal.py` | warm start from a checkpoint (item 3) |
| `scripts/ball_temporal/export_ball_specialist.py` | batch-3 export (item 3) |
| `scripts/train_yolo26_gsr.py` | `--extra_splits`, `--fold`, `--model_yaml`, default imgsz 1280 (item 4) |
| `docs/path_b/PERF_LOG.md` | one dated entry per item |
| `docs/path_b/plans/BALL_TEMPORAL_PLAN.md` | link here from its status line |
