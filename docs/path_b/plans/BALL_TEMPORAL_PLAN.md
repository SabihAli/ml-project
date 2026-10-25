# Ball detection beyond the main detector: plan

Status (2026-10-16): **steps 1–5 of §9 done (step 5 without TensorRT timing); the specialist is
`c5c`** (§10 item 18). v1, a single-frame YOLO26s on 640×640 crops, was scratched (§1). Its
replacement sees **three frames** (§4), predicts a **point** (§5) and trains on **hard examples**
(§6). On the 15 test sequences, main detector + `c5c` raises ball recall from 59.4% to 69.5%
(+10.1) and precision from 85.3% to 86.5%: gates 1 and 3 pass. **Wired into the pipeline
2026-10-25**, batch and live (`gsr_pipeline/ball_specialist.py`, implementation plan Phase 6,
partial): it runs inline after the main detector, not yet on its own CUDA stream. The ball
tracker (§7) is not built. Model write-up and history: `docs/architecture/BALL_TEMPORAL_ARCH.md`.

How to run it: `docs/guides/BALL_TEMPORAL_GUIDE.md` (local GPU, RTX 3090) or
`docs/guides/BALL_TEMPORAL_KAGGLE_GUIDE.md` (Kaggle 2×T4). The v1 scripts are still in
`scripts/` (`build_ball_crop_dataset.py`, `train_ball_specialist.py`,
`evaluate_ball_specialist.py`, `snap_ball_labels.py`) as a record; nothing new uses them. The
runtime design this feeds is
`docs/path_b/plans/IMPLEMENTATION_PLAN.md` §5 (crop planner, specialist, fusion,
`BallTracker`).

**The problem.** On the 15 held-out test sequences, the main Path B detector finds the ball in
59% of frames, at 84% precision. Its average longest miss per sequence is 2.6 s, and the worst
is 6.9 s. Every ball-based event (possession, passes, shots) sits on top of this.

---

## 1. What v1 taught us (Kaggle 2×T4, 2026-10-11/12)

v1 was YOLO26s, initialised from the main detector, trained on native-resolution 640×640 crops
centred near the ball. All numbers are on the 15 test sequences, with a 15 px centre match.

**It works on crops in isolation.** On the same 4,511 test crops, scored the same way at conf
0.15:

| model | recall | recall, 10–20 px balls | precision |
|---|---|---|---|
| main detector at its runtime scale (416) | 57.7% | 58.2% | 81.2% |
| v1 specialist (640, native) | **67.7%** | **68.6%** | 70.1% |

**It barely helps the system.** The specialist runs only when the main detector has missed
the ball. Across 11,250 frames, the best of 7 settings added +4.7 points of recall for −6.2
points of precision; the most precise added +2.9 for −2.2. The gate was +10 recall with no
precision loss, and no cutoff, threshold or camera-compensation setting came close.

What that means for the new design:

1. **The frames that matter are the hard ones.** On frames the main detector missed, the
   specialist found the ball 36% of the time one frame after a sighting, and under 10% after 4
   or more frames. Those frames are motion blur, occlusion and crowds. **A single frame can't
   resolve them, at any resolution** → temporal input (§4).
2. **Resolution was not the bottleneck.** Crops reached the model at native resolution; native
   scale alone was worth about 3 points. The remaining misses aren't about pixels → don't
   spend effort on P2 heads, tiling or upscaling.
3. **GSR ball boxes are noisy by a few pixels**, which wrecks IoU on 10–20 px boxes
   (validation mAP50 0.46, mAP50-95 0.18 for a model that finds two thirds of balls). The
   pipeline only uses the centre → point output (§5).
4. **Training crops must look like runtime crops.** v1 trained on every labelled frame,
   centred on the true ball 3 frames earlier. At runtime it saw only missed frames, centred on
   the last *prediction* → hard examples and runtime-like centring (§6).
5. **Training on few distinct clips overfits fast.** The first run peaked at epoch 17 of 42.
   57 training clips give many frames but few scenes.
6. **Don't move the window with camera motion.** The broadcast camera follows the ball, so the
   ball stays near the same image position while the pitch moves. Shifting the window by the
   pan moved it away from the ball and made every setting slightly worse.
7. **Some misses can't be fixed by any detector.** About 30% of missed balls in a sample of 24
   couldn't be seen in that frame at all. Those are for tracking to fill (§7), later.

---

## 2. Ground truth facts the design uses (measured 2026-10-11)

Source: every `Labels-GameState.json` in `SoccerNetGS/gamestate-2024/{train,valid}`.
Reproduce with `python scripts/ball_label_stats.py <valid.zip> <train.zip>`.

| | train | valid |
|---|---|---|
| sequences / labelled frames | 57 / 42,750 | 58 / 43,500 |
| frames with a ball label | 40,234 (94.1%) | 40,439 (93.0%) |

**Ball size** (`max(w, h)`, native 1920×1080 px): median 14–15, p5 8–9, p95 24–25. About
80–85% of balls are under 20 px, and 9–14% under 10 px.

**Ball displacement between frames** (image px, includes camera pans; valid):

| gap (frames, 25 fps) | p50 | p90 | p99 |
|---|---|---|---|
| 1 | 4 | 18 | 46 |
| 2 | 8 | 35 | 80 |
| 5 | 21 | 85 | 184 |
| 10 | 41 | 160 | 339 |

**A fixed 640×640 window centred on the last ball position** still contains the ball 99.8% of
the time after 5 frames, 98.6% after 10, and 95.9% after 15. So the window doesn't need to grow,
and never needs downscaling.

**Edges:** 30% of balls are within 320 px of a frame edge.

---

## 3. What carries over from v1

These held up and stay as they are:

- **Window: 640×640 native pixels, fed 1:1** (no resize). Built and verified.
- **Edge rule: shift the window into the frame, never pad.** `x0 = clamp(cx − 320, 0, 1280)`,
  `y0 = clamp(cy − 320, 0, 440)`. A ball cut by the crop border is labelled if its centre and at
  least 50% of its box are inside; otherwise the crop is dropped.
- **Splits, by sequence** (revised by decision A, 2026-10-12; code in
  `scripts/ball_temporal/gsr_io.py`):
  - `train`: all 57 `train` sequences. The main detector was trained on these, so they give
    easy positives and negatives only, never hard examples;
  - `pool`: the 33 `valid` sequences outside `select` and `test`. The main detector never
    trained on them, so all hard examples come from here;
  - `select`: 10 `valid` sequences, evenly spaced through the 43 non-test ones, for model
    selection and early stopping;
  - `test`: 15 held-out `valid` sequences (021, 022, 023, 026, 030, 033, 036, 040, 044, 047,
    050, 054, 080, 086, 090), never used for training or selection.
- **Negatives only from ball-labelled frames.** A frame with no ball label may still show an
  unlabelled ball.
- **The kickoff stays in the data and isn't targeted.** The ball is static, white and half
  hidden. A future idea, out of scope: place a ball prior on the centre spot when both teams are
  in their own halves.
- **Both Kaggle T4s:** DDP for training, two shards for evaluation passes, and data builds on the
  CPU in the same session.
- **Reusable code:** `SeqReader`, `discover`, `parse_labels` and `clamp_window` in
  `build_ball_crop_dataset.py`; the `main` and `fuse` stages and their per-sequence cache in
  `evaluate_ball_specialist.py`; the main-detector results for the 15 test sequences
  (`*.main.json`), already computed.

Dropped: the v1 model and checkpoints, the v1 YOLO training config, centring on the true ball
3 frames earlier, box labels, the 85% crop-recall target, and camera-motion shifting of the
window.

---

## 4. Temporal input

The specialist sees **3 consecutive frames, t−2, t−1 and t**, cut with the same window (the
window is chosen at t). Motion separates a ball from pitch lines, socks and heads, a blurred
ball becomes a consistent streak, and a ball hidden at t is often visible at t−1 or t−2.

- **Input, two versions to compare:**
  - **9 channels:** three RGB crops stacked (the TrackNet default);
  - **5 channels:** RGB at t plus two grey difference images (t − t−1, t−1 − t−2), which is
    cheaper and puts motion in directly.

  Frame t always stays full RGB, so a static ball (kickoff, set pieces) is still visible.
- **Runtime cost:** the live pipeline keeps recent decoded frames in its GPU frame ring
  (implementation plan §1). The extra cost is two more 640² slices and a wider first layer.
- **Start of a sequence or after a cut:** if t−1 or t−2 doesn't exist or is from another shot,
  repeat frame t. Training includes that case at 5%, so the model handles it.

---

## 5. Point output

The model predicts the **ball's centre** as a heatmap, not a box. That's what the pipeline uses
and what every gate measures, and it removes the box-label noise.

- **Architecture: a small custom heatmap network in PyTorch, not Ultralytics.** YOLO's head and
  loss are box-based, and multi-channel input plus a heatmap head is less work as a standalone
  model.
  - **Encoder:** ResNet-18-class, ImageNet-initialised, with the first conv widened to 5 or 9
    channels. The RGB weights are copied into each frame's channels and scaled so activations
    stay the same size.
  - **Decoder:** U-Net-style, to a **stride-2 heatmap** (320×320 for a 640 crop), plus a
    2-channel sub-pixel offset.
- **Target:** a Gaussian at the labelled centre, σ = max(1.5, ball size / 4) in heatmap pixels.
  A crop with no ball has an empty heatmap.
- **Loss:** CenterNet focal loss on the heatmap, plus L1 on the offset at the true peak.
- **Decode:** the heatmap's maximum, kept if above a confidence threshold, refined by the offset.
- **Labels:** GSR box centres; box size is used only for σ. Centres are far less affected by the
  label noise than box edges.
- **Augmentation:** horizontal flip, the same colour jitter across all frames, small
  translation. Scale stays close to 1:1, because the runtime feeds crops 1:1.
- **Training:** both T4s with DDP, AMP, early stopping on validation distance-recall (not loss),
  and a patience short enough for the overfitting v1 showed (epoch 17 of 42). *As run
  (2026-10-15):* one local RTX 3090, no DDP, AMP, patience 8; the selection score is recall
  averaged over the thresholds at which 1–10% of `select` negatives fire (§10 item 12).
- **Export:** ONNX, with input (N, C, 640, 640) and outputs heatmap + offset. A parity gate
  against PyTorch, and TensorRT FP16 timing. The runtime contract changes from boxes to points:
  `BallSpecialist` returns (x, y, conf) (implementation plan §5.3–5.4).
- **Fallback if the custom model slips:** the same temporal input on YOLO26s with Ultralytics'
  multi-channel support and an NWD box loss, after checking the installed Ultralytics version
  supports both.

---

## 6. Hard examples and runtime-like crops

Train on what the specialist will see at runtime: frames the main detector missed, with the
window placed where the runtime would put it.

1. **Main-detector pass over every sequence** (`scripts/ball_temporal/cache_main_detections.py`),
   on both T4s. It caches up to 5 ball detections per frame at conf ≥ 0.05, so the hard-frame
   threshold can change without another GPU pass. That is about 86k frames across all four
   splits; test is redone so every split has the same format.
2. **Mark hard frames:** GSR has a ball, and the main detector has no ball within 15 px at conf
   0.15. **Only in `pool`** (decision A): the main detector trained on `train`, so its misses
   there aren't the kind it makes on new footage. `select` and `test` are marked the same way
   for reporting. The cache summary compares main-detector recall on `train` against the other
   splits, to confirm this.
3. **Centre each crop like the runtime does:** on the main detector's most recent prediction
   within the last 15 frames. If there is none, use the true ball position from the most recent
   labelled frame in that range, plus ±48 px of jitter.
4. **Sampling:**

   | crop kind | share | from |
   |---|---|---|
   | hard positives | all of them, repeated ×3 | every hard frame |
   | easy positives | every 3rd frame | other ball-labelled frames |
   | hard negatives | 15% of crops | windows on players or pitch lines, with no ball inside |
   | random negatives | 5% of crops | random windows with no ball inside |

   Hard frames are all kept because they're rare and are the whole point. Easy frames stay
   subsampled because neighbouring frames are near-duplicates.
5. **Validation and test** are built the same way, so the crop-level gate measures the runtime
   case. Hard frames are reported as their own group.
6. **Output per sample:** the 3-frame window saved together (one file per sample), a point
   label (`cx cy size` in crop pixels, or empty), and a manifest with:
   - `split`, `seq`, `frame`, `kind`;
   - `hard` (0/1) and `centre_source` (`pred` or `gt`);
   - the main detector's confidence at t, window `x0`/`y0`, ball size and offset from the window
     centre.

**Builder:** `scripts/ball_temporal/build_ball_temporal_dataset.py` (built 2026-10-13), standalone
with `gsr_io.py`; it reads the step 1 cache. It also writes a sixth kind, `outside`: the runtime
window misses the labelled ball. Many of these windows show a real, unlabelled ball, so training
and the negative false-positive rate leave them out (§10 item 11).

---

## 7. Ball tracker (later, after the specialist is measured)

Not built until the new specialist has passed or failed its gates (§8, gates 1–3). Then it is
measured on top of main + new specialist, for the misses no detector can fix (lesson 7) and
the false positives off the ball's path.

- **State:** image-space position and velocity, constant-velocity Kalman filter. Pitch
  coordinates come later, once calibration feeds the live path.
- **Update:** each frame, with the fused detection (main detector first, then the specialist).
- **Gating:** a detection is accepted only within a radius that grows with frames since the last
  update (about the p99 displacement for that gap, §2). Detections outside it are rejected as
  false positives; a strong detection far away, seen on 2 consecutive frames, starts a new track
  (a cut or a long pass).
- **Coasting:** with no accepted detection, the tracker outputs its predicted position for up to
  `max_coast_frames`, marked `source = predicted`, then drops the track.
- **Batch path only:** forward-backward smoothing (Rauch–Tung–Striebel) and gap interpolation
  between two confident sightings, since the whole sequence is available.
- **The specialist window still centres on the last observed position.** The user ruled out
  velocity-predicted window centres (2026-10-11), so the tracker's prediction is output only.
  Whether it may place the window is an open question for when the tracker exists.
- **Evaluation:** a `track` stage in `evaluate_ball_specialist.py`, run from cached per-frame
  results (no GPU). Coasted positions are reported both counted and not counted, so they can't
  hide misses.
- **Tests:** pure-logic pytest for the filter, gating, coasting and track restart.

---

## 8. Gates

All on the 15 test sequences, with a 15 px centre match.

| # | gate | pass |
|---|---|---|
| 1 | crop level, all test crops | new specialist beats the main detector (416) on the same crops |
| 2 | crop level, **hard** test crops | reported separately; this is the number the new specialist exists to raise |
| 3 | system level: main + specialist, every test frame | recall ≥ main + 10 points, precision ≥ main's |
| 4 | runtime | ONNX parity; TensorRT FP16 latency recorded (`kaggle-t4`); ≤ 5 ms target on the RTX 3060 |
| 5 | tracker (§7), later | main + specialist + tracker: recall up and precision no lower than gate 3's result, longest miss shorter |

Gate 3 also reports specialist recall per gap since the last sighting, which sets
`max_miss_frames`. With v1, recall fell below 10% after 4 frames.

---

## 9. Order of work

| # | step | where | done when |
|---|---|---|---|
| 1 | main-detector cache, `scripts/ball_temporal/cache_main_detections.py` (**built**; guides: `docs/guides/BALL_TEMPORAL_GUIDE.md`, `BALL_TEMPORAL_KAGGLE_GUIDE.md`) | local RTX 3090 or Kaggle 2×T4 | all 115 sequences cached; summary table recorded. **Done 2026-10-13**, local RTX 3090; see the 2026-10-13 entry in `docs/path_b/PERF_LOG.md` |
| 2 | `build_ball_temporal_dataset.py` | local smoke test, then Kaggle | spot-check by eye; hard-frame count recorded. **Done 2026-10-13** (`scripts/ball_temporal/build_ball_temporal_dataset.py`, local); see the 2026-10-13 step 2 entry in `docs/path_b/PERF_LOG.md` |
| 3 | heatmap model + training script | local smoke test, then Kaggle DDP | first run trained, curves saved. **Done 2026-10-15** (local, single RTX 3090): `runs/ball_temporal/c9b`, `c5b`; see the 2026-10-15 entry in `docs/path_b/PERF_LOG.md` |
| 4 | evaluator support for point models in `crops` and `fuse`; 5-ch vs 9-ch | Kaggle | gates 1–3 in a dated `PERF_LOG.md` entry. **Measured 2026-10-15** (local, `scripts/eval/ball_det/eval_ball_temporal.py`): gate 1 pass; gate 3 pass for `c9b`, `c5b` short by 0.8 points under the `select` threshold rule; see `PERF_LOG.md` |
| 4b | `select`-only system threshold rule, re-run gate 3 for `c5b` (a second look, §10 item 14) | local | gate 3 re-reported. **Done 2026-10-15**: the new rule also picks 0.75; gate 3 still fails by 0.8 points (+9.2 recall, +1.2 precision); see `PERF_LOG.md` |
| 4c | retrain `c5c`: GSR `test` sequences added to `pool` (§10 item 15), motion blur (item 16); gates re-run with the fixed `spec_precision` rule | local | gate 3 measured for `c5c`. **Done 2026-10-16:** gate 3 **PASS** (+10.1 recall, +1.2 precision, threshold 0.70 from `select`); see `PERF_LOG.md` |
| 5 | export, parity, runtime contract change to points | Kaggle or local 3090 | gate 4 passes. **Partly done 2026-10-15:** `scripts/ball_temporal/export_ball_specialist.py` (ONNX + sidecar, parity vs PyTorch on ORT CPU: PASS, max point diff 0.0000 px); the pipeline builds missing TensorRT engines at startup (`trt_runner.ensure_engine`). TensorRT timing deferred (§10 item 17) |
| 6 | `BallTracker` + pytest; `track` evaluator stage | local | gate 5 measured |

**Deferred** (user, 2026-10-12; revisit after step 4): copy-paste of balls, motion-blur
augmentation, and more distinct clips (moving most of the 43 non-test val sequences into
training). *Revisited 2026-10-15:* more clips came from GSR `test` instead (item 15), so `select`
and `test` stay intact; motion blur is in (item 16); copy-paste is still deferred.

---

## 10. Decisions

1. **v1 scratched** (user, 2026-10-12). It beat the main detector on crops but failed the system
   gate in all 7 settings tried (§1). Its model isn't used in any later step.
2. **New direction** (user, 2026-10-12): temporal input, point output, hard examples (§4–§6).
3. **Tracking after the specialist, not first** (user, 2026-10-12, §7).
4. **Kept from v1:** 640×640 native window, shift-clamp edge rule, sequence splits with the same
   15 test sequences, negatives only from ball-labelled frames (§3).
5. **No camera-motion shift of the window** (measured, §1).
6. **No velocity-predicted window centres** (user, 2026-10-11). Revisit when the tracker exists
   (§7).
7. **Kickoff in the data, not targeted** (user, 2026-10-11).
8. **Deferred:** copy-paste, motion blur, more clips (user, 2026-10-12).
9. **Decision A** (user, 2026-10-12): hard examples come only from `valid` sequences the main
   detector never trained on (`pool`). 10 non-test `valid` sequences (`select`) are held back
   for model selection (§3).
10. **This phase's scripts live in `scripts/ball_temporal/`** (user, 2026-10-12), standalone,
    so the v1 scripts aren't needed. Evaluators live in `scripts/eval/ball_det/` (user, 2026-10-15).
11. **`outside` crops out of training** (measured, 2026-10-14). Windows centred on a main-detector
    sighting often hold a real but unlabelled ball (a spare ball by the boards); labelled empty,
    they taught the first runs (`c9`, `c5`) to ignore real balls. Those runs are superseded.
12. **Selection score** (2026-10-14): recall averaged over 1–10% negative false-positive rates.
    Recall at a single threshold swung 16–41% between epochs on noise.
13. **`c5b` chosen** (user, 2026-10-15): 5 channels (RGB at t + two difference images) beat
    9 channels on hard test crops (58.5% vs 52.1%) and at every system threshold, and is about
    1.6× cheaper per training step.
14. **Gate 3 threshold rule to be replaced** (2026-10-15). "Precision ≥ main's on `select`"
    picked 0.75 for `c5b` because main's precision is 88.3% on `select` but 85.3% on `test`.
    The new rule is set on `select` only; because test results prompted the change, its re-run
    is reported as a second look. *Outcome (step 4b):* rule "lowest `select` threshold with >= 90%
    of specialist outputs on the labelled ball" also gives 0.75, so gate 3 stays at FAIL by 0.8
    points. Not re-tuned further.
15. **GSR `test` joins `pool`** (2026-10-15). Its 49 sequences are fully labelled and the main
    detector never trained on them (it trained on GSR `train`, validated on `valid`), so their
    misses are real hard examples. `pool` grows from 33 to 82 sequences and hard positives from
    7,563 to 14,432. `select` and `test` are unchanged. (`gsr_io.py` maps the folder to
    `gsrtest`; it is unrelated to our held-out `test`.)
16. **Motion-blur augmentation** (2026-10-15): `--motion_blur p`, same linear blur on all three
    frames. Off by default so `c5b` reproduces.
17. **Step 5 without TensorRT timing** (user, 2026-10-15): export and parity now; engines are
    built by the pipeline itself on first launch or when missing, not by hand. Gate 4 timing
    waits until the engines are wanted.
18. **`c5c` replaces `c5b`** (user, 2026-10-16). Same network, trained with the 49 extra clips and
    motion blur; gate 3 passes (+10.1 / +1.2 at threshold 0.70 from the fixed `spec_precision`
    rule). Exported: `runs/ball_temporal/c5c/ball_specialist_c5c.onnx`, parity checked.
