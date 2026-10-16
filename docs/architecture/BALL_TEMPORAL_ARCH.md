# Temporal ball specialist: model, history and state

Status (2026-10-16): **`c5c` is the current model.** It is trained, passes test gates 1–3 and is
exported to ONNX, but **the pipeline does not load it yet** (implementation plan Phase 6). This
doc is the place to start if the work is picked up again. Plan and decisions:
`docs/path_b/plans/BALL_TEMPORAL_PLAN.md`. Every measurement: `docs/path_b/PERF_LOG.md`
(entries 2026-10-13 to 2026-10-16). How to run each step: `docs/guides/BALL_TEMPORAL_GUIDE.md`.

---

## 0. Current state at a glance

| | |
|---|---|
| model | `c5c`: 3-frame heatmap specialist, 5-channel input, epoch 30 of 30 |
| result (15 test clips) | system ball recall **59.4% → 69.5%** (+10.1), precision **85.3% → 86.5%** (+1.2) |
| gates (plan §8) | 1 PASS · 2 reported · 3 **PASS** (by 0.1 points) · 4 not measured (TensorRT timing deferred) · 5 needs the tracker |
| system threshold | **0.70** (chosen on `select`, see §6.2) |
| checkpoint | `runs/ball_temporal/c5c/best.pt` (58 MB) |
| ONNX + sidecar | `runs/ball_temporal/c5c/ball_specialist_c5c.onnx` (+ `.json`), parity with PyTorch checked |
| model code | `scripts/ball_temporal/heatmap_model.py` |
| training | `scripts/ball_temporal/train_ball_temporal.py` |
| dataset builder | `scripts/ball_temporal/build_ball_temporal_dataset.py` (needs `cache_main_detections.py` first) |
| evaluation | `scripts/eval/ball_det/eval_ball_temporal.py` |
| export | `scripts/ball_temporal/export_ball_specialist.py` |
| in the pipeline | no; the engine auto-build it will use already exists (§6.3) |

> **What is in git.** The `c5c` deliverables are tracked in `runs/ball_temporal/c5c/`: `best.pt`,
> `ball_specialist_c5c.onnx` + `.json`, `args.json`, `history.csv`, `curves.png` (about 116 MB,
> two 58 MB files). **Not tracked** (`data/` and the rest of `runs/` are gitignored): `last.pt`
> (optimiser state, 174 MB), the other runs (`c5b`, `c9b`, ...), the step 1 cache and the 67k-sample
> dataset. These can be rebuilt from the code and the GSR download (§12); training itself is not
> reproducible bit for bit.

## 1. What it does

When the main detector loses the ball, the specialist looks at a **640×640 patch** of the frame,
centred on where the ball was last seen, in the **current frame and the two before it**. It
returns one point (x, y) and a confidence from 0 to 1. No box: the pipeline only uses the centre.

## 2. Input: the 5 channels

A colour image is three layers of numbers (red, green, blue): three **channels**. The model takes
five layers:

| channels | what |
|---|---|
| 1–3 | frame t in colour |
| 4 | grey(t) − grey(t−1): a **difference image**, bright where something moved in the last frame |
| 5 | grey(t−1) − grey(t−2): the same, one frame earlier |

Static things (grass, lines, stands) cancel out in a difference image; a moving ball shows up as a
small bright blob or streak. A 9-channel version (three colour frames, no differences) was also
trained and lost (§8).

## 3. Architecture

```
5 × 640 × 640
   │
Encoder: ResNet-18, ImageNet-pretrained              what is in the patch
   640 → 320 → 160 → 80 → 40 → 20 px
   │                          ▲ skip connections (earlier, sharper features)
Decoder: U-Net style                                 where exactly
   20 → 40 → 80 → 160 → 320 px
   ├─► heatmap  1 × 320 × 320   chance the ball centre is in each 2×2 px cell
   └─► offset   2 × 320 × 320   sub-pixel correction per cell
```

- **First layer widened from 3 to 5 inputs.** The colour channels keep ImageNet's weights; the
  two difference channels start from a greyscale copy (weights summed over RGB, halved).
- **Decoder blocks:** upsample, concatenate the encoder feature of that size, two 3×3 conv +
  batch-norm + ReLU. Heads: 3×3 conv + ReLU + 1×1 conv. The heatmap bias starts at −2.19
  (sigmoid ≈ 0.1), as in CenterNet.
- **Reading the answer:** take the highest heatmap cell, add its offset, multiply by 2 (the
  stride) to get patch pixels; the cell's sigmoid is the confidence.
- **Size and speed:** 14.5 M parameters, 58 MB. About **8 ms per call** in PyTorch FP16, batch 1,
  RTX 3090 (measured on `c5b`, the same network). TensorRT is expected to be faster; not measured.

## 4. Data

Source: SoccerNet GSR `gamestate-2024`: 1920×1080, 25 fps, 750 frames (30 s) per clip, labelled
ball boxes. Splits by clip (`scripts/ball_temporal/gsr_io.py`):

| split | clips | from | used for |
|---|---|---|---|
| `train` | 57 | GSR `train` | training; easy examples only (the main detector trained on these) |
| `pool` | 82 | 33 GSR `valid` clips + all 49 GSR `test` clips | training; **the only source of hard examples** |
| `select` | 10 | GSR `valid` | choosing the epoch and the threshold (a "practice exam") |
| `test` | 15 | GSR `valid`: 021 022 023 026 030 033 036 040 044 047 050 054 080 086 090 | final gates only |

The main detector never trained on `pool`, `select` or `test`. GSR's `test` folder is not our
`test` split.

**Building the samples** (`build_ball_temporal_dataset.py`), per ball-labelled frame t:

1. **Hard frame** = a ball is labelled, and the main detector has no ball at confidence ≥ 0.15
   within 15 px of it. Found from a cached main-detector pass over every frame
   (`cache_main_detections.py`: conf ≥ 0.05, top 5 per frame, imgsz 1280).
2. **Window** centred like the runtime: on the main detector's last ball (≥ 0.15) in frames t−1 …
   t−15; failing that, the true ball in that range ± 48 px; failing both, no sample. The window
   is shifted to stay inside the frame, never padded. Frames t−1 and t−2 use the same window.
3. **Kinds:**

| kind | what | count (all splits) |
|---|---|---|
| `hard` | hard frame, ball in the window; outside `train` only; weighted ×3 in `pool` | 19,971 |
| `easy` | main detector found the ball; every 3rd frame | 30,066 |
| `neg_hard` | no ball in any of the 3 frames; window on a player or a pitch line | 9,977 |
| `neg_random` | no ball; random window | 3,375 |
| `outside` | the labelled ball is outside the runtime window | 3,610 (**not trained on**, §8) |

Each sample is one JPEG with the three crops side by side (640×1920, t−2 | t−1 | t, quality 95),
plus a row in `manifest.csv` with the label (`cx cy size` in crop pixels). 66,999 samples (23 GB).
Training uses 51,866 of them: `train` + `pool`, without `outside`.

## 5. Training recipe (`c5c`)

| | |
|---|---|
| target | a Gaussian bump at the ball centre on the 320×320 grid, σ = max(1.5, ball size / 4) in grid cells; empty for no ball |
| loss | CenterNet focal loss on the heatmap + L1 on the offset at the true cell |
| sampling | with replacement, weighted by `repeat` (hard `pool` ×3): 80,704 samples per epoch |
| augmentation | mirror flip; the same contrast/brightness/colour gain on all 3 frames (on the GPU); ±16 px shift; 5% of samples with frame t copied over t−1 and t−2 (clip start or a cut); **motion blur** on 25%: one linear kernel, 3–9 px, random angle, on all 3 frames |
| optimiser | AdamW, lr 5e-4, weight decay 1e-4, 1-epoch warm-up then cosine to 5e-6; gradient clip 10; mixed precision |
| batch / epochs | 32 / 30 (patience 8, not reached) |
| epoch selection | **score on `select`**: recall within 15 px, averaged over the thresholds at which 1%, 2% … 10% of negative crops fire. Best: epoch 30, 74.0% (hard 57.9%, easy 95.6%) |
| hardware / time | one RTX 3090, 0.35 s per step, 15 min per epoch, **7.5 h** |
| command | `python scripts/ball_temporal/train_ball_temporal.py --out runs/ball_temporal/c5c --channels 5 --motion_blur 0.25` |

The score was still creeping up at the end (+0.1 per epoch), with the learning rate at its floor.
A fresh, longer schedule (~45 epochs) might add about a point.

## 6. How it runs, and the deployment contract

### 6.1 In the system (simulated by the evaluator; the runtime version is not built)

Per frame:

1. The main detector finds a ball (confidence ≥ 0.15) → use it; it becomes the "last sighting".
2. Otherwise, if the last sighting is ≤ 15 frames old → run the specialist on the window centred
   there. Confidence ≥ **0.70** → use its point, and it becomes the new last sighting.
3. Otherwise → no ball this frame (the tracker's job, plan §7).

### 6.2 Why 0.70

Below the threshold, a guess is thrown away. A lower threshold finds more balls but makes more
wrong guesses. The rule, fixed before `c5c` was scored: **the lowest threshold at which ≥ 90% of
the specialist's own guesses on `select` are on the labelled ball.** For `c5c` that is 0.70. It
is chosen on `select` only and never re-tuned on `test`.

### 6.3 ONNX and the TensorRT engine

- `export_ball_specialist.py` writes the ONNX (opset 17, static batch 1) and a sidecar with the
  checkpoint, epoch, channels, threshold (0.70) and the ONNX sha256.
- **Input** `frames`: float32 (1, 9, 640, 640): RGB crops of t−2, t−1, t, values 0–255. The
  normalisation and the difference images are inside the graph.
- **Outputs** `heatmap` (1, 1, 320, 320) logits and `offset` (1, 2, 320, 320); decode on the host
  as in §3.
- **Parity:** ONNX Runtime CPU vs PyTorch FP32 on 64 test crops: max point difference 0.0000 px,
  confidence 1.1e-6. ONNX Runtime GPU and TensorRT were not checked (the CUDA provider did not
  load in `.ball_venv`; TensorRT is not installed).
- **Engine:** never built by hand. The pipeline builds `<onnx stem>_fp16.engine` beside the ONNX
  on first launch, or whenever the file is missing (`gsr_pipeline/runtime/trt_runner.ensure_engine`,
  config `auto_build_engine`). That mechanism already serves the athlete/ball detector and the
  NBJW calibration networks; the specialist will use it once a loader exists.

## 7. Results (15 test clips, 15 px match)

**Crops on their own** (test patches; specialist threshold where 5% of `select` negatives fire):

| | main detector | `c5c` |
|---|---|---|
| all patches with a ball (5,551) | 36.4% | 69.9% |
| hard: main detector missed (3,533) | 0% | 54.5% |
| easy (2,018) | 100% | 96.9% |
| negative patches that fire (1,463) | 3.1% | 3.6% |

**System, every frame** (main detector + specialist, §6.1):

| | ball recall | precision | longest miss (mean / max) |
|---|---|---|---|
| main detector alone | 59.4% | 85.3% | 2.9 s / 9.9 s |
| **+ `c5c` (threshold 0.70)** | **69.5% (+10.1)** | **86.5% (+1.2)** | 2.7 s / 9.9 s |

- 94.4% of the specialist's guesses are on the labelled ball, more than the main detector's own
  85%. That is why precision goes **up**.
- Recall by frames since the last sighting: 66% at 1 frame, 18% at 2, 9% at 3–5, 4% at 6–10, 1% at
  11–15. It is a "catch it straight after we lose it" model.
- Recall here means: of the 10,178 test frames with a labelled ball, the share where the system's
  single point is within 15 px of it. Frames without a label are not scored (some show an
  unlabelled ball).

**Against earlier attempts** (same 15 clips):

| approach | system recall change | system precision change |
|---|---|---|
| v1: single-frame YOLO26s on 640 crops (scratched) | +4.7 (best) / +2.9 (most precise) | −6.2 / −2.2 |
| `c5b`: this design, 90 training clips | +9.2 | +1.2 |
| **`c5c`: + 49 clips, motion blur** | **+10.1** | **+1.2** |

## 8. Training history

| date | run | what happened | lesson |
|---|---|---|---|
| 2026-10-11/12 | v1 | single-frame YOLO26s on 640 crops; good on crops, failed the system gate in 7 settings | one frame can't resolve blur or occlusion; train like the runtime; GSR boxes are noisy, so use centres |
| 2026-10-13 | dataset | 46k samples from 115 clips; 13% of hard frames put the ball outside the runtime window | — |
| 2026-10-13 | `c9`, `c5` | first runs; the selection score swung 16–41% between epochs; `c9` stopped early with epoch 1's weights | selecting on recall at one threshold is noise; use a recall-vs-false-positive area |
| 2026-10-14 | diagnosis | the most confident "false positives" were all `outside` crops holding a **real spare ball** (e.g. SNGS-038 frames 124–150) | `outside` crops are mislabelled and taught the model to ignore real balls; excluded from training and from the negative count |
| 2026-10-14 | speed | `channels_last` made the network **4× slower**; CPU colour jitter cost 60 ms per sample | removed `channels_last`, moved jitter to the GPU: 2.4 → 0.56 s per step (9 ch) |
| 2026-10-15 | `c9b`, `c5b` | clean runs, both ~71% on `select`; 5 channels better on hard test crops (58.5% vs 52.1%) and 1.6× cheaper | motion as ready-made difference images beats letting the network compare frames |
| 2026-10-15 | gate 3, `c5b` | +9.2 / +1.2, FAIL by 0.8; two threshold rules, each fixed before running, both gave 0.75 | don't keep changing the rule until it passes |
| 2026-10-15 | more data | GSR `test.zip` holds 49 labelled clips the main detector never saw → `pool` 33 → 82 clips, hard examples 7.6k → 14.4k | — |
| 2026-10-16 | **`c5c`** | 5 ch + the new clips + motion blur; `select` 74.0%; gate 3 **PASS**, +10.1 / +1.2 | more distinct clips mattered most |

**Why 5 channels beat 9** (likely, not tested in isolation): the 9-channel first layer starts by
averaging the three frames (the pretrained weights are split evenly), which hides motion until the
network learns to undo it from few clips; and frame t loses its untouched pretrained weights.

## 9. Why it works

- **Versus the main detector:** a small, full-resolution patch already centred where the ball
  probably is, plus motion, instead of the whole 1920×1080 frame.
- **Versus v1:** motion (a blurred or half-hidden ball moves, so it stands out); a point target
  (no noisy box edges); training on the hard frames the runtime will hand it, centred where the
  runtime centres them.

## 10. Engineering notes

- **Windows:** DataLoader workers re-import the script, so every script needs an
  `if __name__ == "__main__"` guard. Without one, a script silently runs many times over.
- **Evaluator:** memoises specialist calls by (clip, frame, window), so a threshold sweep re-uses
  results. The `fused` simulation is sequential, because a specialist hit moves the next window.
- **`.ball_venv`** (Python 3.11, torch 2.6.0+cu124, ultralytics 8.4.176) runs the ball work and the
  pure-logic pytest suite. It is not the pipeline's full environment: `requirements.txt` was not
  installed into it, to keep the CUDA build of torch.
- **Reproducing `c5b`:** train without `--motion_blur` on a dataset built from GSR `train` +
  `valid` only (without the GSR `test` folder, `pool` is 33 clips).

## 11. Open items

1. **Wire it into the runtime** (implementation plan Phase 6): crop planner, a `BallSpecialist`
   that returns (x, y, conf) instead of boxes, frames t−1/t−2 kept in the frame ring.
2. **Gate 4:** TensorRT FP16 latency (target ≤ 5 ms on an RTX 30-series), deferred.
3. **Tracker** (plan §7): recall falls below 20% two frames after a sighting, and the longest miss
   is still 9.9 s. Filling those gaps is the tracker's job.
4. **Spare balls:** the specialist confirms whatever ball the window holds. Picking the match ball
   is a fusion/tracker problem.
5. **Gate 3 passes by 0.1 points**, on 15 clips from one dataset. Treat the margin as thin.
6. Possible gains not tried: a longer schedule (~+1 point on `select`), copy-pasting balls into
   crops (deferred), an ablation of 5 vs 9 channels.

## 12. Rebuilding from scratch

From the repo root, with the GSR `train`, `valid` and `test` splits extracted under one folder
(`<src>`; guide §3), and `.ball_venv` set up (guide §2):

```bash
python scripts/ball_temporal/cache_main_detections.py run --src <src> --out data/outputs/ball_temporal/main_cache --weights runs/yolo26s_athlete_ball/best.pt --device 0 --half --batch 32 --decode_threads 4
python scripts/ball_temporal/build_ball_temporal_dataset.py build --src <src> --out data/outputs/ball_temporal/dataset --workers 8
python scripts/ball_temporal/train_ball_temporal.py --out runs/ball_temporal/c5c --channels 5 --motion_blur 0.25
python scripts/eval/ball_det/eval_ball_temporal.py all --ckpt runs/ball_temporal/c5c/best.pt --src <src> --out data/outputs/ball_temporal/eval/c5c
python scripts/ball_temporal/export_ball_specialist.py --ckpt runs/ball_temporal/c5c/best.pt --out runs/ball_temporal/c5c/ball_specialist_c5c.onnx --threshold 0.70
```

About 45 min (cache), 20 min (dataset), 7.5 h (training), 10 min (evaluation) and 1 min (export)
on an RTX 3090. Training samples at random, so a rerun lands close to these numbers, not exactly on
them.
