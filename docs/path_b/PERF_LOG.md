# Path B — Performance and Accuracy Log

One dated entry per phase, per `IMPLEMENTATION_PLAN.md` §10.4: hardware, numbers, what
changed. Numbers from a machine that is not the production host are tagged as such and do
**not** count against the §1.4 targets.

## Hardware tags

| Tag | Machine | Notes |
|---|---|---|
| `dev-cpu` | 13th Gen Intel Core i5-1335U, Intel Iris Xe (no NVIDIA GPU) | torch 2.13.0+**cpu**, `torch.version.cuda = None`. Cannot run tiers 1–2, NVDEC, CUDA events or NVML. Useful only for correctness and for A/B against other `dev-cpu` runs. |
| `kaggle-t4` | Kaggle notebook, Tesla T4, 2 vCPU | torch 2.11.0+cu128, ORT 1.22.0, TensorRT 10.16.1.11. Real CUDA, but not the production card, and stages run back to back on one thread. Doesn't count against §1.4. |
| `rtx-3090` | Local desktop, NVIDIA RTX 3090 24 GB, driver 610.88, Windows 11 | torch 2.6.0+cu124, Ultralytics 8.4.176, Python 3.11. A 30-series card, so it can be compared with the `prod-gpu` targets, but it isn't the provisioned production host. |
| `prod-gpu` | RTX 30-series, not yet provisioned | Every §1.4 target is written against this. See plan §16 item 2. |

---

## 2026-10-16: ball temporal plan step 4c, `c5c` (more clips + motion blur)

Tag: `rtx-3090`. `train_ball_temporal.py --channels 5 --motion_blur 0.25`, 51,866 training crops
(80.7k samples per epoch), 30 epochs at 15 min (7.5 h); stopped at the epoch cap, best at epoch 30.
`select` score 74.0% (hard 57.9%, easy 95.6%), up from `c5b`'s 71.3%; still rising by about 0.1
per epoch at the end, with the learning rate at its floor. Gates with the fixed `spec_precision`
rule (no re-tuning). Report: `data/outputs/ball_temporal/eval/c5c/report.md`.

| | `c5b` | **`c5c`** |
|---|---|---|
| crop recall, all / hard (5% `select` negative FPR) | 72.6% / 58.5% | 69.9% / 54.5% |
| negative crops firing at that threshold | 5.4% | 3.6% |
| system threshold (`select`, >= 90% specialist outputs on the ball) | 0.75 | 0.70 |
| **system recall (main alone 59.36%)** | 68.56% (+9.2) | **69.47% (+10.1)** |
| **system precision (main alone 85.29%)** | 86.52% (+1.2) | **86.51% (+1.2)** |
| gate 3 | FAIL | **PASS** (by 0.1 points) |

- Crop numbers are at different operating points (the 5% threshold sits higher for `c5c`, which
  fires on fewer negatives), so they are not a like-for-like drop.
- Recall by gap since the last sighting: 66% at 1 frame (`c5b` 60%), then 18%, 9%, 9%, 4%, 1%.
- Longest miss unchanged (2.7 s mean / 9.9 s max).

---

## 2026-10-15: ball temporal plan step 5 (export) and step 4c data

Tag: `rtx-3090`.

- **Export (`c5b`):** `scripts/ball_temporal/export_ball_specialist.py` →
  `runs/ball_temporal/c5b/ball_specialist_c5b.onnx` (57.9 MB, opset 17, static 1×9×640×640,
  preprocessing in the graph) + sidecar. Parity against PyTorch FP32 on 64 test crops, ORT CPU:
  max point diff 0.0000 px, max confidence diff 1.2e-6, **PASS**. ORT GPU not checked: the CUDA
  execution provider did not activate in `.ball_venv` (CUDA/cuDNN runtime libraries not found).
- **Engine build at startup:** `trt_runner.ensure_engine`, used by `select_backend` and NBJW.
  8 new pure-logic tests; `pytest tests/` 59 passed. No engine built (tensorrt not installed).
- **More clips:** the 49 GSR `test` sequences added to `pool` (plan §10 item 15). Main detector
  over them: 12 min, 16 ms/frame. Cache summary now:

| split | sequences | frames with GT ball | main recall | hard frames |
|---|---|---|---|---|
| train | 57 | 40,234 | 84.32% | 6,238 |
| pool | 82 | 56,991 | 69.60% | 16,853 |
| select | 10 | 7,074 | 63.95% | 2,492 |
| test | 15 | 10,178 | 59.36% | 4,013 |

  The main detector does better on the new clips (pool recall 61.6% → 69.6%), but still far
  below its 84% on the clips it trained on.
- **Dataset rebuilt:** 66,999 samples (11.3 min); `pool` hard positives 7,563 → **14,432**;
  51,866 training crops (was 32,364).

---

## 2026-10-15: ball temporal plan step 4b, gate 3 second look (`c5b`)

Tag: `rtx-3090`. Same evaluator and settings as step 4, new threshold rule fixed before the run
(plan §10 item 14): the lowest `select` threshold at which >= 90% of the specialist's own outputs
on ball-labelled frames are on the labelled ball. Report: `data/outputs/ball_temporal/eval/c5b_rule2/report.md`.

- The rule picks **0.75**, the same threshold as the first rule, so the result is unchanged:
  recall 68.56% (+9.2), precision 86.52% (+1.2). **Gate 3: FAIL by 0.8 points.**
- On `select`, specialist outputs are 84–87% on the labelled ball from 0.45 to 0.70 and reach 90%
  only at 0.75 (on `test` at 0.75: 95.4%). Some "misses" on `select` may be real but unlabelled
  balls (spare balls; see the step 2 correction), so this share is a lower bound.
- The rule is not changed again: a third rule chosen after seeing this would be tuning on test.
  Both rules are recorded; the diagnostic test sweep (step 4 entry) stays diagnostic.

---

## 2026-10-15: ball temporal plan step 4, test gates 1-3

Tag: `rtx-3090`. Script: `scripts/eval/ball_det/eval_ball_temporal.py`. 15 test sequences, 15 px
centre match, main detector from the step 1 cache (imgsz 1280, conf >= 0.15; the plan's gate 1
names the 416 runtime scale, so this baseline is the stronger one). System simulated frame by
frame: specialist runs only when main misses, on a window centred on the last sighting, for up
to 15 frames. `fused` = the runtime (a specialist hit re-centres the window); `main` = only main
sightings place it. System threshold rule: highest `select` recall with precision >= main's on
`select`; frozen for test. Reports: `data/outputs/ball_temporal/eval/<run>/report.md`.

**Gates 1-2 (test crops, threshold at 5% `select` negative FPR):**

| crops | main | `c5b` | `c9b` |
|---|---|---|---|
| all positives (5,551) | 36.4% | **72.6%** | 68.1% |
| hard (3,533) | 0% | **58.5%** | 52.1% |
| easy (2,018) | 100% | 97.3% | 96.2% |
| negatives fire (1,463) | 3.1% | 5.4% | 5.8% |

Gate 1: PASS for both.

**Gate 3 (system, every test frame; main alone: recall 59.36%, precision 85.29%, longest miss
2.9 s mean / 9.9 s max):**

| run | chain | `select` threshold | recall | precision | gate 3 |
|---|---|---|---|---|---|
| `c5b` | fused | 0.75 | 68.56% (+9.2) | 86.52% (+1.2) | FAIL (by 0.8 points) |
| `c9b` | fused | 0.55 | 70.24% (+10.9) | 86.41% (+1.1) | **PASS** |
| `c5b` | main | 0.70 | 68.49% (+9.1) | 86.37% (+1.1) | FAIL |

- **Diagnostic test sweep (not used to choose):** `c5b` fused beats `c9b` at every threshold, and
  keeps precision above main's 85.29% from 0.20 to 0.80 (e.g. 0.40: recall 73.99%, +14.6;
  precision 86.16%). `c5b`'s failure is the threshold rule: main's precision is 88.3% on
  `select` but 85.3% on `test`, so ">= main on select" picks a conservative threshold.
- Specialist outputs are 93-95% on the labelled ball, above main's own precision, so adding them
  raises system precision.
- Fused chain: recall by gap since the last sighting falls from 60-69% at 1 frame to under 20%
  at 2+ (frames at gap 2+ are ones the specialist already missed). Main chain holds 26-48% across
  gaps 1-15 but ends at the same total.
- Longest miss max stays 9.9 s: long gaps exceed the 15-frame window, the tracker's job (§7).
- PyTorch FP16, batch 1: about 8 ms per specialist call on the 3090 (not gate 4; TensorRT is step 5).
- Compare v1 (2026-10-12): best system setting +4.7 recall for −6.2 precision.

---

## 2026-10-15: ball temporal plan step 3, first heatmap specialist runs

Tag: `rtx-3090`. Scripts: `scripts/ball_temporal/heatmap_model.py`, `train_ball_temporal.py`.
Trained on `train` + `pool` (32,364 crops, `outside` excluded; hard `pool` positives ×3, 47,490
samples per epoch), batch 32, AdamW 5e-4, AMP, patience 8. Scored on `select` (5,016 crops).
Score = distance recall (15 px) averaged over the thresholds at which 1–10% of negative crops
fire. These are `select` numbers, not the test gates (step 4).

| run | input | best epoch / stopped | score (hard / easy) | recall at 5% neg FPR (hard / easy) | s/step | min/epoch |
|---|---|---|---|---|---|---|
| `c9b` | 9 ch (3 RGB) | 18 / 26 | **71.1%** (55.0% / 92.8%) | 72.6% (56.7% / 94.0%) | 0.56 | 14.3 |
| `c5b` | 5 ch (RGB + 2 diffs) | 23 / 30 (resumed at 11 with patience 16) | **71.3%** (54.8% / 93.4%) | 73.1% (56.7% / 95.2%) | 0.36 | 9.3 |

- Both plateau at about 70% (`c9b` from epoch 13, `c5b` from epoch 11); select loss rose while
  recall held (overconfidence, not lost recall). 5 and 9 channels are tied on `select`; 5 channels
  is about 1.6× cheaper per step, so it is the default candidate for step 4 unless the test gates
  separate them.
- `outside fire` stayed at 25–43%: windows centred on a main-detector sighting often contain a
  real ball the specialist finds, whether or not it is the labelled one.
- Earlier runs `c9`/`c5` (2026-10-13/14) are superseded: their selection metric was unstable and
  `outside` crops taught them to suppress real balls (see the step 2 correction above).

---

## 2026-10-13: ball temporal plan step 2, temporal crop dataset

Tag: `rtx-3090` machine, CPU only (8 workers). Script: `scripts/ball_temporal/build_ball_temporal_dataset.py`,
from the step 1 cache. 46,170 samples, 16 GB (one 640×1920 JPEG per sample: crops t−2 | t−1 | t,
quality 95), 6.4 min. Spot-checked by eye on preview sheets from a 2-sequence smoke run (SNGS-024,
SNGS-060): labels sit on the ball, hard positives are blur, balls in the air and balls in a
goalkeeper's hands, and negatives contain no ball.

| split | samples | hard | easy | outside | neg_hard | neg_random | hard frames sampled (any kind) | sum of repeat | GT-centred |
|---|---|---|---|---|---|---|---|---|---|
| train | 16669 | 0 | 13306 | 83 | 2438 | 842 | 61 | 16669 | 256 |
| pool | 16972 | 7563 | 4783 | 1194 | 2541 | 891 | 8692 | 32098 | 1806 |
| select | 5016 | 2006 | 1494 | 507 | 749 | 260 | 2486 | 5016 | 453 |
| test | 7513 | 3533 | 2018 | 499 | 1095 | 368 | 3994 | 7513 | 675 |

- **Hard-example set:** 7,563 hard positives from `pool` (×3 in training via `repeat`), out of
  8,719 hard frames. The rest are frames with no sighting in the last 15 frames (no crop), a
  ball cut ambiguously by the window (dropped), or the ball **outside** the runtime window.
- **`outside`:** about 13% of hard frames in `pool`, `select` and `test` put the labelled ball
  outside a window centred like the runtime. **Correction (2026-10-14):** many of these windows
  are centred on a *real* but unlabelled ball (a spare ball by the advertising boards; SNGS-038
  frames 124–150 are a long run of them), because the main detector's last sighting was that
  ball. Labelled empty, they taught the first training runs to suppress real balls and made the
  selection score count correct detections as false positives. Training now excludes them and
  scores false positives on `neg_hard`/`neg_random` only. They are a fusion/tracker problem
  (which ball is the match ball), not a specialist one.

---

## 2026-10-13: ball temporal plan step 1, main-detector cache

Tag: `rtx-3090`. Ultralytics 8.4.176, PyTorch 2.6.0+cu124. Guide: `docs/guides/BALL_TEMPORAL_GUIDE.md`.
Detector `runs/yolo26s_athlete_ball/best.pt`, imgsz 1280, FP16, batch 32, 4 decode threads. Source:
the extracted GSR `train` and `valid` splits in `data/datasets/SoccerNetGS/` (the guide's
`data/gsr/` was not used; the guide still says `--src data/gsr`).

- **Smoke test:** SNGS-021, 32 frames: output shape correct, no ball detected (kickoff frames,
  which every model misses). 111 ms/frame, mostly warm-up. A second check on SNGS-024, 480
  frames, gave **26 ms/frame**; 435 of 480 frames had a detection and the top one was within
  15 px of GT in 392.
- **Full run:** 115 sequences, 86,250 frames, about **30 min** wall time (first to last cache
  file timestamp).
- **Sanity checks:** split counts 57/33/10/15 as expected. `test` recall 59.36% against the
  ~59% measured on Kaggle 2026-10-12. `train` recall is clearly above the other splits, so
  hard examples come from `pool` only (plan decision A holds).

| split | sequences | frames | frames with GT ball | main recall (top ball) | hard frames | hard share |
|---|---|---|---|---|---|---|
| train | 57 | 42750 | 40234 | 84.32% | 6238 | 15.50% |
| pool | 33 | 24750 | 23187 | 61.63% | 8719 | 37.60% |
| select | 10 | 7500 | 7074 | 63.95% | 2492 | 35.23% |
| test | 15 | 11250 | 10178 | 59.36% | 4013 | 39.43% |

The hard-example set before oversampling is the `pool` count: **8,719 frames**.

---

## 2026-10-03 / 2026-10-05: first GPU runs, Path B batch end to end

Tag: `kaggle-t4`. Runner: `notebooks/path_b_eval.py`, 3 valid sequences × 100 frames. Full
report: `docs/results/PATH_B_EVAL_T4_2026-10-05.md` (the 2026-10-03 report had the same
detection and role numbers and was removed; its one extra figure, calibrate 479.9 ms/frame, is
recorded below). Latency work
since then is tracked in `docs/path_b/plans/OPTIMIZATION_PLAN.md`.

- **Phase 1b, tier 2 parity:** ORT GPU vs Ultralytics `.pt`, rectangular 736×1280: PASS,
  worst min IoU 0.999982, worst conf delta 2.0e-6, no count mismatches. The square variant
  was skipped (no valid reference, see 2026-09-26). The tier 1 detector engine is **not
  built**, so FP16 detector accuracy is still unmeasured.
- **Detection (pooled):** athlete R/P 97.94% / 94.71%; ball R/P **60.00% / 66.91%**, ball
  frame recall 60%, longest miss 4.0 s. Ball recall by GT box size: 10–20 px 56.4% (n=275),
  ≥20 px 100% (n=25). The small-ball gap is the case for the plan §5 ball specialist.
- **Roles on GT:** role accuracy 95.52%, team 98.22%; referee recall 81.77%.
- **GS-HOTA (jersey ignored, no refinement):** persons 58.95%, with_ball 57.75%.
- **Latency, sequential (2026-10-05):** 85.9 ms mean / 178.1 ms p95 per frame (11.65 fps).
  Detector 24.3 ms (ORT GPU, inference 17.6 ms). Calibration 37.5 ms mean with NBJW on
  TRT FP16 and K=5, down from 479.9 ms on 2026-10-03 once camera-parameter voting was turned
  off and propagation turned on.

---

## 2026-09-26 — Phase 1a: detector runtime, export, backend fallback

Tag: `dev-cpu`. Scope: plan §12 Phase 1a — everything that can be closed without a GPU.
**No latency numbers are recorded in this entry**, deliberately: §1.4's budget is a GPU
budget, and CPU timings would not be comparable to it in any direction. Phase 0 (`perf.py`,
CUDA events, NVML) has not been built, so there is also no instrumentation to record them
with yet.

### What was built

| Artifact | Status |
|---|---|
| `scripts/export_athlete_detector.py` | working; both variants exported |
| `scripts/build_trt_engine.py` | written; **unverified** (no GPU) — exits 2 with an actionable message |
| `scripts/parity_gate.py` | working |
| `gsr_pipeline/runtime/{postprocess,ort_runner,trt_runner,backend}.py` | tiers 2–3 working; tier 1 unverified |
| `gsr_pipeline/detect_athlete.py` (`AthleteBallDetector`) | working |
| `evaluate_detection.py` 2-class mode, `--start_frame`, `--backend` | working |
| `tests/` (28 tests) | all passing |

Exported artifacts (`runs/yolo26s_athlete_ball/`):

| File | Input | Output tensor |
|---|---|---|
| `best_736x1280.onnx` | 1×3×736×1280 | `(1, 6, 19320)` |
| `best_1280x1280.onnx` | 1×3×1280×1280 | `(1, 6, 33600)` |

`6 = 4 box + 2 classes`; 19320 = 92·160 + 46·80 + 23·40, i.e. strides 8/16/32 over
736×1280, which confirms the head layout rather than assuming it.

### Correction: the head is not NMS-free

The plan claimed an "NMS-free end-to-end head (`nms: null`)". Loading the checkpoint shows
`Detect`, `end2end=False`, `reg_max=1`, `max_det=300`. `nms: null` in `args.yaml` is an
unset export argument, not evidence. NMS is therefore required and runs host-side in torch
(`torchvision.ops.batched_nms`), shared by every backend tier — decision 7 of 2026-09-26.
Plan §3, §3.2, §3.3, §3.5 and §1.4 were corrected.

### Preprocessing: the parity-critical finding

Matching Ultralytics' letterbox is not a formality. Measured on SNGS-021 frame 200 against
Ultralytics `.pt` at 736×1280 — 17 boxes on both sides in every variant:

| Resize path | mean IoU | min IoU | max conf delta |
|---|---|---|---|
| torch bilinear, `antialias=True` | 0.98840 | 0.95707 | 0.011980 |
| torch bilinear, `antialias=False` | 0.99709 | 0.97848 | 0.032877 |
| **`cv2.resize` INTER_LINEAR on uint8** | **1.00000** | **1.00000** | **0.000001** |

The intuition is backwards here: `cv2.INTER_LINEAR` does **not** antialias when
downscaling, so enabling antialiasing moves *away* from the reference. `preprocess` now
takes the cv2 path for numpy input (exact) and `antialias=False` for tensor input (the
GPU path, 0.997 — to be re-gated in Phase 2 when GPU preprocessing actually exists).

### Parity gate (§3.5) — tier 3, ORT CPU

**SNGS-021, open play (frames 168–749, 582 frames), scored end-to-end through
`evaluate_detection.py --path b`:**

| Metric | Ultralytics `.pt` (reference) | runner + ORT CPU | delta |
|---|---|---|---|
| athlete recall / precision | 96.24% / 94.26% | 96.24% / 94.26% | **0.0000 pts** |
| ball recall / precision | 78.60% / 71.27% | 78.60% / 71.27% | **0.0000 pts** |
| ball frame recall | 78.87% | 78.87% | 0.0000 |
| longest consecutive ball miss | 22 frames (0.88 s) | 22 frames | 0 |
| athlete TP/FP/FN | 8729 / 532 / 341 | 8729 / 532 / 341 | identical |
| ball TP/FP/FN | 459 / 185 / 125 | 459 / 185 / 125 | identical |

Not "within 1 point" — **identical counts**. Gate passes with the entire tolerance unused.

This also reproduces `DETECTOR_AND_DATA_BLOCKERS.md` §2, which reported 79% / 71%
open-play ball recall/precision and a 22-frame longest gap for this model at conf 0.15.
Reproducing it required the new `--start_frame 168`: the kickoff (frames 0–167, ball
static, white-on-white, occluded) is missed by every model at every threshold and drags
whole-sequence ball recall down by roughly 20 points, hiding the difference the gate exists
to measure.

The reference side runs at the **training config's** `iou=0.7`, `max_det=300`, not
`detect.py`'s Path A `iou_threshold=0.45`. Scoring against the wrong NMS setting would have
measured the setting rather than the runner.

**Box-for-box, 5 sequences, rectangular 736×1280, ORT CPU** (`scripts/parity_gate.py`;
thresholds min IoU ≥ 0.99, max conf delta ≤ 0.01, zero count or class disagreements):

| Sequence | boxes | count mismatches | min IoU | max conf delta | class disagreements |
|---|---|---|---|---|---|
| SNGS-021 | 102 | 0 | 0.999994 | 0.00000095 | 0 |
| SNGS-022 | 141 | 0 | 0.999993 | 0.00000137 | 0 |
| SNGS-033 | 44 | 0 | 0.999990 | 0.00000113 | 0 |
| SNGS-047 | 134 | 0 | 0.999987 | 0.00000215 | 0 |
| SNGS-090 | 88 | 0 | 0.999991 | 0.00000077 | 0 |

**The square 1280×1280 variant has no valid reference, and cannot be parity-tested this
way.** Ultralytics' predictor uses `LetterBox(auto=True)`, which pads only to a stride
multiple: asked for `imgsz=(1280, 1280)` on a 1920×1080 frame it still feeds the model
`(1, 3, 736, 1280)`. Verified directly — `(736,1280)`, `(1280,1280)` and `1280` all produce
the same 736×1280 tensor.

A first run of the gate did not check this and reported the square variant as FAIL (up to
19 count mismatches per sequence, one min IoU of 0.000000). That failure was **an artifact
of comparing a 1280-tall ONNX against a 736-tall reference**, not a defect in the runner.
`parity_gate.py` now probes the reference's actual input shape and skips the variant with
an explanation rather than emitting a FAIL that implicates the runner.

Consequence for plan §3.4: **square vs. rectangular is an accuracy question, not a parity
one.** It has to be settled by scoring both against ground truth with
`evaluate_detection.py`, plus a speed comparison on a GPU. Neither is done. Phase 1a
establishes only that the rectangular export is numerically faithful to the reference.

### Backend fallback

Verified live on `dev-cpu`: tier 1 rejected (`no CUDA device`), tier 2 rejected
(`no CUDA device`), tier 3 selected, with a WARNING naming the degraded mode. Every
rejection carries its reason, so a host that silently lands on CPU is diagnosable from the
log rather than from a stopwatch.

The paths that need a real GPU are covered by fake-backend unit tests instead
(`tests/test_backend_selection.py`): TensorRT preferred when present, sidecar mismatch
demoting to ORT GPU without taking the service down, a missing engine reported with the
command that fixes it, warm-up failure rejecting a tier that loaded but cannot infer, and a
pinned tier raising rather than silently falling back.

Artifact contract refusals, all tested: wrong class list, embedded NMS (would
double-suppress), ONNX/sidecar checksum drift, missing sidecar, unknown output layout.

### Not done — carried to Phase 1b (needs a CUDA host)

1. Build a TensorRT engine and run `scripts/build_trt_engine.py` for real.
2. Parity gate on tier 1 (TRT FP16) and tier 2 (ORT GPU). **FP16 is the one still-open
   accuracy risk**: it is the first thing that would damage ball recall, and it is
   completely unmeasured. Tier 3 parity says nothing about it.
3. Force each fallback live, including a genuinely mismatched sidecar.
4. Decide square vs. rectangular. Two separate measurements, neither done: **accuracy**
   (score both exports against GT — the square variant cannot be parity-tested against
   Ultralytics at all, see above) and **speed** (the ~44% padding saving is a GPU claim).
   Phase 1a establishes only that the *rectangular* export is faithful to the reference.

### Notes

- CPU inference is ~2 s/frame at 736×1280 on `dev-cpu`, consistent with the blockers doc's
  0.73 s/frame on a different CPU. This is why the box-level gate runs on frame subsets
  while the scored gate runs the full open-play span of one sequence.
- Phase 1a ran **before** Phase 0, at the user's direction. Nothing here imports `perf.py`,
  so the reordering is safe, but it means this entry carries no perf baseline.

### Code review and fixes (2026-09-27), before closing Phase 1a

Six findings against the Phase 1a diff (`evaluate_detection.py`'s Path B additions,
`gsr_pipeline/runtime/`, `detect_athlete.py`, the three `scripts/*.py`); the other three
findings from the same pass were on unrelated pre-existing code outside Phase 1's scope and
are not addressed here. All six verified against the actual code before fixing, then
re-verified after (28/28 tests pass; the parity gate re-run below reproduces the same
numbers as the original run).

1. **`evaluate_detection.py` F1 truthiness bug.** `if recall and precision else None` used
   truthiness instead of `is not None`, so a total miss (`recall=0.0, precision=0.0` — a
   real, computable F1 of 0.0) printed "n/a" instead of "0.0%", silently hiding a class
   that was missed entirely. Confirmed by direct evaluation: the fixed code correctly
   distinguishes "undefined" (no GT and no predictions) from "zero" (a genuine miss).
   Fixed with explicit `is None` checks and an explicit 0.0 case for the 0/0 harmonic mean.
2. **`evaluate_detection.py` no check on `cv2.imread` failure.** A missing or corrupt
   frame file silently became `None`, which would fail inside `preprocess()` on Path B with
   a confusing `AttributeError` instead of naming the bad file. Now raises
   `FileNotFoundError` naming the exact path.
3. **`parity_gate.py` used per-row `argmax` instead of Hungarian (one-to-one) matching.**
   `evaluate_detection.py`'s own `match()` already uses `linear_sum_assignment` for the
   documented reason — two nearby boxes (routine on a soccer pitch) can both have their
   highest IoU against the same counterpart under argmax, stealing a true match and
   producing a spuriously low `min_iou` / high `max_conf_delta`, i.e. a false FAIL on
   numerically identical detectors. Fixed by importing `_iou_matrix` from
   `evaluate_detection.py` (removing a duplicate local copy — finding 4) and matching with
   `linear_sum_assignment`. Re-run on 3 sequences after the fix: identical PASS, identical
   min IoU/max conf-delta figures to before, confirming this was a robustness fix rather
   than a behaviour change on this clean data.
4. **Duplicate `iou_matrix` in `parity_gate.py`** vs. `evaluate_detection.py`'s `_iou_matrix`
   — the two could drift independently. Removed in favour of the shared one (see 3).
5. **`build_trt_engine.py --fp32` wrote to the same filename `select_backend()` always
   loads as the FP16 tier** (`ModelArtifact.engine_path` is hardcoded to `..._fp16.engine`).
   An operator debugging an FP16 accuracy issue with `--fp32` would get an FP32 engine
   silently served as "tensorrt, FP16" with no indication in the logs. Fixed: `--fp32`
   without an explicit `--engine` now writes to a `_fp32.engine` name instead, with a
   printed note that this file is deliberately outside what `select_backend()` will find.
6. **`ort_runner.py`'s `_infer_bound` allocated a fresh output tensor (`torch.empty`) on
   every call**, unlike `TrtRunner`, which preallocates once — a CUDA allocation on every
   frame of a live stream on the ORT-GPU tier, against the codebase's stated "allocate
   once" design (plan principle 4). Fixed: the output buffer is now allocated once in
   `__init__` and reused. **Unverified on real hardware** (no GPU on this machine, same as
   the rest of tier 2) — reviewed and compiled, not executed.

All fixes re-verified: `pytest tests/` 28/28 pass; `evaluate_detection.py --path b` spot
check shows a correct non-"n/a" F1; `parity_gate.py` re-run on 3 sequences reproduces PASS
with the same figures as the pre-fix run.

**Phase 1a is closed as of 2026-09-27.** Everything CPU-verifiable in plan §12's Phase 1a
scope is built, gated, tested and reviewed. The Phase 1b items above (a CUDA host) remain
open and are unaffected by this review — findings 5 and 6 touch that code but could only be
reviewed, not run, here.
