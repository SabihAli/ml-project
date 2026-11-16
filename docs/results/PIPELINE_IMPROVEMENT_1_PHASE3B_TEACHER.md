# Pipeline improvement 1, Phase 3b: offline teacher T2 (2026-11-15)

Phase 3b of `docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md`. Hardware `rtx-3090`. Teacher = offline ball detector
whose labels will train the real-time students; inference cost does not matter, only accuracy.

## Result

`t2m_s1920_ft`: YOLO26s, imgsz 1920, 5 input channels (BGR(t) + grey(t-1) + grey(t+1)), initialised from the
live detector `runs/yolo26s_athlete_ball/best.pt`, ball only. Scored by `train_teacher_yolo.py eval`: one ball
per frame (top score), centre within 15 px; threshold chosen on `select` (highest recall at precision >= 90%),
frozen for `test`.

| | recall @15 px | precision | F1 | recall @5 px | outputs on unlabelled frames |
|---|---|---|---|---|---|
| select (10 seq), thr 0.05 | 83.21% | 90.07% | 86.50% | 71.25% | 195 |
| **test (15 seq)** | **84.93%** | **93.49%** | **89.00%** | 71.02% | 316 |
| test without SNGS-050 | 84.77% | 93.63% | 88.98% | 70.84% | 316 |
| test, best-F1 thr 0.01 (oracle) | 86.83% | 92.10% | 89.39% | 72.44% | 504 |
| live system (Phase 1) | 76.29% | 90.21% | | | |

- **+8.6 pp recall, +3.3 pp precision over the live system**, from one single-frame network plus two
  neighbour channels. `test` >= `select` and SNGS-050 changes nothing: no sign of clip memorisation.
- **Gate (test recall >= 88% at precision >= 90%): not met**, by 3.1 pp.
- **The model is at its ceiling.** The threshold is near the floor (0.05) and even 0.01 reaches only 86.8%:
  ~13% of ball frames have no candidate near the ball at any score. A better threshold or a stronger
  single-frame classifier cannot fix those; more temporal context can (next steps).
- Outputs on unlabelled frames (316 on test) are not yet audited: some may be real balls GSR left unlabelled.

Raw: `data/outputs/pipeline_improvement_1/phase3b/teacher_yolo/eval/t2m_s1920_ft/teacher_eval.{json,md}`;
run: `runs/teacher/t2m_s1920_ft/` (gitignored).

## Offline trajectory pass (2026-11-15)

`scripts/pipeline_improvement_1/phase3b/trajectory_pass.py` over the teacher's top-5 candidates per frame
(`train_teacher_yolo_motion.py dump`, conf >= 0.001; GPU ~13 min once). CPU, 40 s for 486 configurations.
(1) Temporal support: a candidate's score adds the best-matching candidates within +-W frames, weighted by a
Gaussian on distance whose width grows with the frame gap; per frame the best-supported candidate wins.
(2) Gap fill: frames between two kept frames <= G apart, at a plausible speed, are interpolated.

| test (15 seq) | recall @15 px | precision | F1 | recall @5 px | w/o SNGS-050 R / P |
|---|---|---|---|---|---|
| T2 raw top-1 (thr 0.05) | 84.92% | 93.48% | 88.99% | 71.03% | 84.77% / 93.63% |
| + temporal support only (thr 0.06) | 86.88% | 94.04% | 90.32% | 72.56% | 86.68% / 94.17% |
| **+ gap fill, G=10 (thr 0.09)** | **88.06%** | **93.89%** | **90.88%** | **73.50%** | 87.72% / 93.99% |

Select for the chosen row: 86.05% / 90.16%. Knobs W=2, sigma0=8 px, v=25 px/frame, w=0.5, G=10, vmax=40
px/frame, chosen on `select` from 486 configurations (a 72-configuration grid picked the same, so the choice
is not at a grid edge). The raw row reproduces the eval within 0.03 pp (fp16 inference run twice).

**Gate (test recall >= 88% at precision >= 90%): met, at the margin** -- 88.06% with 3.9 pp precision to spare;
without SNGS-050 87.72%, just under. Both steps help and precision rises, so the pass removes false positives as
well as filling gaps. Read it as "teacher + pass is at the gate", not comfortably past it: T1 is still worth
building, and the T1 + T2 ensemble is the next lever.

## T1 and the T1 + T2 ensemble (2026-11-15)

T1 (`t1_r34`, `train_teacher_heatmap.py`): 5 RGB frames t-2..t+2, ResNet-34 U-Net, stride-2 heatmap + offset,
960x544 crops, batch 8 (~6 GB). Select recall (at P >= 90%, every 10th frame) peaked at 78.8% at epoch 10 and
hovered at 66-73% otherwise while train loss kept falling; early-stopped after 15 epochs. It plateaued with the
LR still at 25-50% of peak, so more epochs would not help.

Candidate coverage on test (any top-5 candidate within 15 px of the ball): T2 93.29%, T1 90.91%, **union 95.28%**
(select 92.14 / 92.39 / 95.09%). The two models miss different frames.

`fuse_candidates.py`: per frame, candidates within 15 px are one object, score = a * T2 + (1 - a) * T1 (a missing
partner counts 0), position score-weighted; then `trajectory_pass.py` (W 1-3, sigma0 8, v 25/40, w 0.5/1, G
0/10/15, vmax 40). The weight a is chosen on select like the other knobs.

| a (weight on T2) | 0.3 | 0.5 | **0.7** | 0.8 | 0.9 |
|---|---|---|---|---|---|
| select recall (P >= 90%) | 86.09% | 86.61% | **87.49%** | 87.14% | 86.56% |

| test (15 seq) | recall @15 px | precision | F1 | recall @5 px | w/o SNGS-050 R / P |
|---|---|---|---|---|---|
| T1 raw top-1 (thr 0.36) | 77.80% | 92.13% | 84.36% | 66.43% | 77.93% / 92.47% |
| T1 + trajectory pass | 83.79% | 92.04% | 87.72% | 70.42% | 83.73% / 92.30% |
| T2 + trajectory pass (above) | 88.06% | 93.89% | 90.88% | 73.50% | 87.72% / 93.99% |
| **T2 + T1 fused (a 0.7) + pass** (W=1, v=25, w=0.5, G=10, thr 0.13) | **88.41%** | **93.83%** | **91.04%** | **74.60%** | **88.17%** / 94.03% |

Small but consistent gain (+1.4 pp select, +0.35 pp test) and the first result past 88% without SNGS-050. a = 0.8
scores higher on test (89.01%) but was not the select choice. Still a thin margin over the gate for students that
will lose some recall, so the next teacher is a 7-channel T2 (grey t-2, t-1, t+1, t+2) on YOLO26m, COCO init with the
head from COCO's `sports ball` row (smoke test: 24.5 GB at batch 4, ~85 min/epoch), to be re-fused with T1.

## 7-channel YOLO26m (`t2m7_m1920_coco`, 2026-11-16): not adopted

YOLO26m at 1920, 7 channels (BGR(t) + grey t-2, t-1, t+1, t+2), COCO init, head from COCO's `sports ball` row;
batch 4, 16 epochs. Val mAP50 0.668 at epoch 11 (T2: 0.660), flat from epoch ~6 while train loss kept falling.
Same top-5 coverage as T2 (test 93.26% vs 93.29%), but it ranks the ball first less often:

| test (15 seq), choices on select | select R | test recall | test precision | w/o SNGS-050 R |
|---|---|---|---|---|
| m7 raw top-1 | 79.57% | 80.67% | 95.11% | 80.39% |
| m7 + trajectory pass | 82.87% | 85.09% | 94.75% | 84.58% |
| m7 + T1 fused (a 0.8; select flat 85.6-85.8% for a 0.5-0.8) + pass | 85.81% | 86.82% | 94.43% | 86.39% |
| T2 + m7 + T1 three-way (best on select: 0.85 / 0.8) + pass | 87.29% | 89.31% | 94.21% | 88.97% |
| **T2 + T1 fused + pass (kept)** | **87.49%** | **88.41%** | **93.83%** | **88.17%** |

The three-way fusion scores higher on test but lower on select (all six weight pairs 86.7-87.3% vs 87.49%), so by the
select rule it is not chosen; the select differences are within noise either way. Bigger model + more frames did not
help; losing the football init likely cost the ranking. **Teacher for the students: T2 + T1 fused (0.7 / 0.3) + the
trajectory pass, test 88.41% / 93.83%.**

## Training

`scripts/pipeline_improvement_1/phase3b/train_teacher_yolo_motion.py train --batch 4 --epochs 30 --patience 5`,
stopped after 16 epochs (~34 min each, 14.5 GB). Train: GSR `train` + `pool`, every 3rd frame (34,750 frames,
32,408 with a ball); val: `select` every 10th frame. Mosaic, flips, scale +-0.25, translate 0.1; no HSV jitter
(Ultralytics skips colour augmentation on non-3-channel input).

| epoch | 1 | 6 | 11 | 13 | 16 |
|---|---|---|---|---|---|
| val mAP50 | 0.586 | 0.617 | 0.660 | 0.658 | 0.659 |
| val mAP50-95 | 0.263 | 0.283 | 0.308 | 0.309 | 0.303 |
| val cls loss | 1.82 | 1.69 | 1.53 | 1.53 | 1.56 |
| train cls loss | 1.46 | 1.33 | 1.18 | 1.15 | 1.10 |

Plateau from epoch ~11: train loss keeps falling, val flat (mild memorisation onset). mAP50-95 means little for a
5-15 px ball (a 2 px box error fails the high IoU thresholds); the eval above is the decision metric.

## How the channels are built

No extra files: the script replaces Ultralytics' dataset `imread` so a listed frame is read with its two
neighbours from the hard-link mirror (`train_teacher_yolo.py build` links every frame, listed or not). At a clip
end the missing neighbour is the frame itself. Channel order is B, G, R, grey-, grey+ in training and eval
(Ultralytics reverses BGR->RGB only for 3 channels). The first conv is 5-channel: RGB weights from the init
(reversed to BGR), grey channels zero, so step 0 is the single-frame detector.

Checked before trusting the run: frame ids contiguous in all 139 train sequences; neighbours differ by 8.4 grey
levels mean vs 19.5 one second apart (they are adjacent); after mosaic/flip/scale the grey channels align with
BGR 2-3x better than an 8 px shift; label boxes sit on the ball (visual check).

## Decisions on the way (user, 2026-11-14/15)

- **No large model.** The first run (YOLO26l + P2 head, COCO init, ~2 h/epoch) was stopped: on 139 correlated
  clips a larger model mostly adds memorisation capacity; resolution and temporal input are the levers.
- **Same model size as live, full resolution, football init** instead: YOLO26s at 1920 from the live detector.
  P2 head off (only 360/902 tensors transfer into the P2 layout vs 696/708 without).
- **SAHI not used.** Tiles at native scale give the ball no more pixels than full-frame 1920 and lose context;
  only upscaled tiles would help, and only if resolution turns out to be the lever.
- **Run A (3-channel control) skipped.** It would only split the gain between motion and resolution/init; we
  keep the motion model either way, and it costs ~9 GPU hours.
- **DETR (T3) deprioritised.** Its COCO edge is on medium/large objects; finest feature stride 8; needs more data;
  no Ultralytics tooling. The remaining misses are frames without a candidate, which need temporal context,
  not a stronger single-frame classifier.

## Bug caught by the smoke tests

Initialising a 1-class model from the 2-class live detector (`athlete`, `ball`): Ultralytics builds the
trainer's model before it knows the dataset's class names, so the head was not remapped by name and started as
the **athlete** classifier. Every player became a confident ball (cls loss 64-190, 0 hits on 10 frames vs 8 for
the init). Fixed by `ball_row_callback` in `train_teacher_yolo.py` (copies the init's `ball` row into the head,
both one-to-many and one-to-one branches, matched by stride; EMA reloaded). After the fix, cls loss 1.8 and val
P/R 0.59/0.41 within 52 steps. The stopped YOLO26l run used COCO init and was not affected.

## Next

1. ~~Offline trajectory pass~~ done (above): test 88.06% / 93.89%, gate met at the margin.
2. ~~**T1**, multi-frame heatmap teacher, then the T1 + T2 ensemble~~ done (above): fused test 88.41% / 93.83%. Built
   (`train_teacher_heatmap.py`): ResNet-34 U-Net on 5 RGB frames t-2..t+2 instead of HRNet (torchvision weights,
   no new dependency; c5c's proven design). Smoke test: 4.5 GB at batch 8, ~43 img/s, full-frame inference
   ~15 fps; select recall 45% at 90% precision after 120 steps on 8 sequences.
3. ~~**7-channel YOLO26m**~~ done (above): worse than T2, not adopted.
4. Produce the teacher's labels on the student training data (next phase step).
