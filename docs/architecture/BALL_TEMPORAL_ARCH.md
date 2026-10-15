# Temporal ball specialist (`c5b`): what it is and why it works

Status (2026-10-15): chosen model for the ball specialist. Plan: `docs/path_b/plans/BALL_TEMPORAL_PLAN.md`.
Numbers: `docs/path_b/PERF_LOG.md` (2026-10-13 to 2026-10-15 entries). Not yet exported or wired into
the pipeline (plan §9 steps 5–6).

| | |
|---|---|
| checkpoint | `runs/ball_temporal/c5b/best.pt` (epoch 23 of 30, 58 MB) |
| model code | `scripts/ball_temporal/heatmap_model.py` |
| training | `scripts/ball_temporal/train_ball_temporal.py --channels 5` |
| evaluation | `scripts/eval/ball_det/eval_ball_temporal.py` |

---

## 1. What it does

When the main detector loses the ball, the specialist looks at a **640×640 patch** of the frame,
centred on where the ball was last seen. It sees that patch in the **current frame and the two
frames before it**, and answers one question: *where in this patch is the ball, and how sure am I?*

Output: one point (x, y) and a confidence from 0 to 1. No box: the pipeline only uses the ball's
centre.

## 2. Input: what "channels" are

A colour image is three layers of numbers (red, green, blue). Each layer is a **channel**. A model
can take any number of layers stacked together. Two versions were trained:

| version | channels | what they are |
|---|---|---|
| `c9b` | 9 | frames t−2, t−1, t, each as plain RGB (3 × 3) |
| **`c5b`** | **5** | frame t as RGB (3) + two greyscale **difference images** (2): t − (t−1) and (t−1) − (t−2) |

A difference image is near zero wherever nothing changed (grass, lines, stands) and bright wherever
something moved. A moving ball shows up as a small bright blob or streak. `c5b` gets motion handed
to it directly, instead of having to learn how to compare frames.

## 3. Architecture

```
5 × 640 × 640 input
   │
   ▼
Encoder: ResNet-18 (ImageNet-pretrained)          "what is in the patch"
   640 → 320 → 160 → 80 → 40 → 20 px, richer features at each step
   │                          ▲
   ▼                          │ skip connections (sharp, early features)
Decoder: U-Net style                               "where exactly"
   20 → 40 → 80 → 160 → 320 px
   │
   ├─► Heatmap head  1 × 320 × 320   chance the ball's centre is in each 2×2-px cell
   └─► Offset head   2 × 320 × 320   sub-pixel correction for each cell
```

- **Encoder.** A standard image network that already knows edges, textures and shapes from 1.2
  million ImageNet photos. Its first layer was widened from 3 to 5 inputs. The colour channels keep
  the pretrained weights as they are; the two difference channels start from a greyscale version of
  them.
- **Decoder.** Shrinking the image loses position detail, and the ball is only about 14 px across.
  The decoder scales back up to 320×320, pulling in the encoder's earlier, sharper layers at each
  step, so it has both meaning and exact position.
- **Reading the answer.** Take the highest heatmap cell, add its offset, multiply by 2 to get patch
  pixels. That cell's value is the confidence.
- **Size and speed.** 14.5 million parameters. About 8 ms per call in PyTorch FP16 on an RTX 3090,
  before TensorRT. The gate 4 target is ≤ 5 ms with TensorRT (step 5).

## 4. Training

- **Data:** 32,364 three-frame patches from the 57 GSR `train` sequences and the 33 `pool`
  sequences (part of GSR `valid`). Built by `scripts/ball_temporal/build_ball_temporal_dataset.py`.
  - **Hard examples:** frames where the main detector missed the ball. Taken only from `pool`,
    because the main detector trained on `train`. Shown 3× as often.
  - **Easy examples:** frames where the main detector found the ball (every 3rd frame).
  - **Negatives:** patches with no ball, on players or pitch lines (look ball-like) or random.
  - **Centring:** on the main detector's last sighting, the same way the runtime centres the window.
  - Left out: patches where the labelled ball is outside the window. Many of these show a real but
    unlabelled ball (a spare ball by the advertising boards); labelled "empty", they taught the
    first runs to ignore real balls.
- **Target:** a small Gaussian bump at the true ball centre; an empty heatmap when there is no ball.
- **Loss:** CenterNet focal loss on the heatmap (concentrates on the peak and on ball-like
  background), plus L1 on the offset.
- **Augmentation:** mirror flip; the same lighting and colour change on all three frames; a small
  shift; and in 5% of samples, frame t copied over the earlier frames (start of a clip or a cut).
- **Model selection:** scored every epoch on the 10 `select` sequences (recall within 15 px,
  averaged over thresholds where 1–10% of negative patches fire). Best: epoch 23.
- **Hardware:** one RTX 3090, batch 32, mixed precision, about 9 min per epoch.

## 5. How it runs in the system

Per frame:

1. If the main detector finds the ball (confidence ≥ 0.15), use it. Its position is the new
   "last sighting".
2. Otherwise, if the last sighting is at most 15 frames old, run the specialist on a 640×640 window
   centred there (shifted to stay inside the frame). If its confidence is above the threshold, use
   its point, and re-centre the window on it for the next frame.
3. Otherwise, no ball for this frame (the tracker's job, plan §7).

## 6. Results (15 held-out test sequences, 15 px centre match)

### Crops: the specialist on its own (gates 1–2)

Test patches built the same way as the training data. Specialist threshold: the one where 5% of
`select` negatives fire.

| patches | main detector | `c9b` (9 ch) | **`c5b` (5 ch)** |
|---|---|---|---|
| all with a ball (5,551) | 36.4% | 68.1% | **72.6%** |
| **hard: main detector missed (3,533)** | **0%** | 52.1% | **58.5%** |
| easy: main detector found (2,018) | 100% | 96.2% | 97.3% |
| no ball, false detection (1,463) | 3.1% | 5.8% | 5.4% |

### System: main detector + specialist on every frame (gate 3)

Gate: recall up by ≥ 10 points, precision not lower than the main detector's.

| | recall | precision | longest miss (mean / max) |
|---|---|---|---|
| main detector alone | 59.4% | 85.3% | 2.9 s / 9.9 s |
| + `c5b`, threshold chosen on `select` (0.75) | 68.6% (**+9.2**) | 86.5% (**+1.2**) | 2.5 s / 9.9 s |
| + `c5b` at 0.40 (test sweep, diagnostic only) | 74.0% (**+14.6**) | 86.2% (**+0.9**) | — |

- `c5b` keeps system precision above the main detector's at every threshold from 0.20 to 0.80.
- 93–95% of the specialist's detections are on the labelled ball, more than the main detector's own
  85%, so adding them **raises** precision.
- At the threshold chosen on `select`, `c5b` misses gate 3 by 0.8 points. The rule compared against
  the main detector's precision on `select` (88.3%), which is higher than on `test` (85.3%), so it
  chose a cautious threshold. A `select`-only rule will be fixed before step 5; that re-run is a
  second look, not a first result.

## 7. Improvement over earlier attempts

All on the same 15 test sequences.

| approach | on frames the main detector missed | system recall | system precision |
|---|---|---|---|
| main athlete/ball detector (YOLO26s, whole frame) | — | 59.4% | 85.3% |
| v1: single-frame YOLO26s on 640 crops (scratched 2026-10-12) | 36% at 1 frame after a sighting, < 10% at 4+ | **+4.7** (best setting) | **−6.2** |
| v1, most precise setting | | +2.9 | −2.2 |
| **`c5b`: 3-frame heatmap specialist** | **58.5%** of hard patches | **+9.2** (`select` threshold) to **+14.6** (0.40) | **+0.9 to +1.2** |

In short: about **2–3× the recall gain of v1**, and precision goes **up** instead of down.

## 8. Why it works

**Against the main detector.** The main detector scans a whole 1920×1080 frame for everything. The
specialist gets a small, full-resolution patch already centred where the ball probably is, plus
motion.

**Against v1 (single-frame crops).** Three changes, each fixing a v1 lesson:

1. **Motion.** In one frame, a blurred ball, a ball half behind a leg, or a ball on a white line
   looks like noise. Across three frames it moves, so it stands out.
2. **A point, not a box.** GSR ball boxes are a few pixels off, a big error on a 14 px box. v1
   learned that noisy box shape. The specialist learns only the centre, which the labels get right.
3. **Training that matches runtime.** v1 trained on every frame, centred on the true ball. The
   specialist trains on the hard frames the runtime will actually hand it, centred where the runtime
   centres them.

**Against the 9-channel version.** `c5b` is 6.4 points better on hard test patches and better at
every system threshold. Likely reasons (not tested in isolation):

1. **Motion arrives ready-made.** `c9b`'s first layer starts by averaging the three frames (the
   pretrained weights are split evenly across them), which hides motion until the network learns to
   undo it, from only 90 training clips.
2. **The current frame is untouched.** `c5b` sees frame t through the original pretrained weights;
   `c9b` dilutes them across three frames.

`c5b` is also about 1.6× faster to train per step (0.36 s vs 0.56 s), mostly from the smaller first
layer.

## 9. Limits

- **Strong right after a sighting, weak after.** About 60% recall one frame after the last sighting,
  under 20% at two or more. A ball that is really hidden can't be found by any detector; that is
  the tracker (plan §7).
- **Longest miss is unchanged at 9.9 s.** Long gaps outlast the 15-frame window.
- **It can't tell the match ball from a spare one.** A window centred on a spare ball by the boards
  gets that ball reported.
- **Not yet in the runtime.** ONNX export, parity check and TensorRT timing are step 5.
