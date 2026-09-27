# Path B — Performance and Accuracy Log

One dated entry per phase, per `IMPLEMENTATION_PLAN.md` §10.4: hardware, numbers, what
changed. Numbers from a machine that is not the production host are tagged as such and do
**not** count against the §1.4 targets.

## Hardware tags

| Tag | Machine | Notes |
|---|---|---|
| `dev-cpu` | 13th Gen Intel Core i5-1335U, Intel Iris Xe (no NVIDIA GPU) | torch 2.13.0+**cpu**, `torch.version.cuda = None`. Cannot run tiers 1–2, NVDEC, CUDA events or NVML. Useful only for correctness and for A/B against other `dev-cpu` runs. |
| `prod-gpu` | RTX 30-series, not yet provisioned | Every §1.4 target is written against this. See plan §16 item 2. |

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

Exported artifacts (`model_testing/yolo26s_athlete_ball/`):

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
