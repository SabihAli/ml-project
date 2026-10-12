SNGS-023  |  athlete R/P 98.81%/97.96%  ball R/P 100.00%/100.00%  |  GS-HOTA persons 67.48%  |  role acc 94.66%  |  89 ms/frame

# Path B evaluation

- run: 2026-10-05T05:45:14Z   gpu: Tesla T4   torch 2.11.0+cu128   ort 1.22.0   tensorrt 10.16.1.11
- detector backend: **ort_gpu**   sequences: 3   max_frames: 100

## Parity gate (runner vs Ultralytics .pt, box for box)

- **ort_gpu**: PASS (rc=0)
  - rectangular 736x1280: PASS, worst min-IoU 0.999982, worst conf delta 2.03e-06, count mismatches 0
  - square 1280x1280: skipped, worst min-IoU None, worst conf delta None, count mismatches 0

## Detection (3 sequences, counts pooled)

| class | GT | TP | FP | FN | recall | precision | F1 |
|---|---|---|---|---|---|---|---|
| athlete | 4849 | 4749 | 265 | 100 | 97.94% | 94.71% | 96.30% |
| ball | 300 | 180 | 89 | 120 | 60.00% | 66.91% | 63.27% |

Ball frame-recall 60.00% over 300 labelled frames; longest miss 4.00s (max), 1.47s (mean).
Ball recall by GT box size: <10px n/a (n=0), 10-20px 56.36% (n=275), >=20px 100.00% (n=25)

## Role / team classifier on GT (3 sequences, held-out frames)

- role accuracy 95.52% over 5183 detections
- team accuracy 98.22% over 4666 pairs (permutation-invariant: team_a/team_b are arbitrary names)
- per-role recall: player 96.55% (n=4666), goalkeeper 99.25% (n=133), referee 81.77% (n=384)

## GS-HOTA, end-to-end Path B (3 sequences, mean of per-sequence scores)

Jersey in IdSim: **ignored**.

| variant | DetA | AssA | GS-HOTA |
|---|---|---|---|
| persons | 52.89% | 71.78% | 58.95% |
| with_ball | 50.42% | 70.62% | 57.75% |

`persons` excludes the ball from GT and predictions; `with_ball` keeps it.

## Latency per frame, ms (tag: `Tesla T4`)

Wall time of each stage run back to back on one thread, pooled over all frames (first 3 frames of each sequence dropped as warm-up). GPU stages end in a device-to-host copy, so their GPU time is included. This is the sequential cost, not a pipelined production figure: plan 1.4's targets assume threaded stages, NVDEC and an RTX 30-series, so they are not comparable.

| stage | n | mean | p50 | p95 | p99 | max | share of total |
|---|---|---|---|---|---|---|---|
| frame load (cv2.imread of a JPEG; production uses NVDEC) | 291 | 10.7 | 10.8 | 12.2 | 13.3 | 14.2 | 12% |
| detector end-to-end (preprocess + inference + NMS + D2H) | 291 | 24.3 | 24.2 | 25.4 | 26.1 | 28.2 | 28% |
| tracker (ByteTrack) | 291 | 6.9 | 6.3 | 8.5 | 9.3 | 183.3 | 8% |
| calibration (LiveCalibrator: NBJW on fresh frames, motion + propagation otherwise) | 291 | 37.5 | 15.9 | 128.8 | 130.7 | 132.7 | 44% |
| pitch projection | 291 | 1.3 | 1.3 | 1.6 | 1.8 | 4.1 | 1% |
| torso descriptors + record assembly | 291 | 5.2 | 5.4 | 6.9 | 8.7 | 9.5 | 6% |
| **WHOLE PIPELINE, per frame** | 291 | 85.9 | 64.7 | 178.1 | 182.3 | 241.0 |  |
| **whole pipeline minus calibration + projection** | 291 | 47.1 | 47.3 | 51.6 | 54.1 | 224.2 |  |

Sequential throughput: **11.65 fps** (mean), 5.62 fps at p95.
Without calibration and projection: 21.2 fps (mean).
Calibration: backend `tensorrt/tensorrt`, calib_every_k=config, mode=config; homography source per frame: fresh 59, propagated 241.
Role/team fit + predict + vote: 49 ms per sequence (~0.49 ms/frame amortised), not in the per-frame rows above.

### Sub-stage breakdown (SNGS-021, 40 frames, first 8 dropped; GPU steps synchronised)

| step | mean | p50 | p95 | max |
|---|---|---|---|---|
| detector: preprocess (CPU letterbox + H2D) | 5.0 | 5.0 | 5.4 | 5.7 |
| detector: inference (GPU, synced) | 17.6 | 17.6 | 17.7 | 17.8 |
| detector: decode + NMS + D2H | 1.9 | 1.9 | 2.1 | 2.2 |
| calibration: image prep (H2D of uint8 frame + on-device resize) | 2.6 | 2.6 | 2.7 | 2.8 |
| calibration: keypoint network (synced) | 32.9 | 33.0 | 34.1 | 34.2 |
| calibration: line network (synced) | 37.6 | 37.7 | 38.4 | 38.5 |
| calibration: heatmap decode + keypoint completion | 39.4 | 39.3 | 40.3 | 40.6 |
| calibration: camera update (add keypoints) | 0.0 | 0.0 | 0.1 | 0.1 |
| calibration: homography (RANSAC) | 0.8 | 0.8 | 0.9 | 1.0 |
| calibration: camera-param voting (only when compute_cam_params) | 19.0 | 18.9 | 21.7 | 23.4 |

Detector steps sum to 24.5 ms; calibration steps sum to 132.5 ms.
Compare with the detector and calibration rows above; a large gap means this breakdown has drifted from the real code and the main-pass rows are the ones to trust.

## Scope notes

- GS-HOTA here is the repo's evaluate.py unchanged. Detections without a pitch projection score as misses.
- No tracklet refinement (needs ReID); calibration is per-frame NBJW (the accuracy reference, plan 7).
- Ball: single highest-confidence detection per frame, one constant track id (plan Phase 6 BallTracker not built).
- Unassociated detections become one-frame singleton tracklets rather than being dropped.
- Roles: AthleteRoleTeamClassifier fitted per sequence on the first --fit_frames frames (0 = whole sequence), then a per-tracklet majority vote. team_a (lower mean pitch x) is labelled 'left'.
- athlete_roles tunables were fitted on SNGS-021 only; every other sequence is a held-out test of them.

wrote /kaggle/working/out_trt/summary.json and summary.md; per-sequence results and predictions.json under /kaggle/working/out_trt
# Path B evaluation

- run: 2026-10-05T05:45:14Z   gpu: Tesla T4   torch 2.11.0+cu128   ort 1.22.0   tensorrt 10.16.1.11
- detector backend: **ort_gpu**   sequences: 3   max_frames: 100

## Parity gate (runner vs Ultralytics .pt, box for box)

- **ort_gpu**: PASS (rc=0)
  - rectangular 736x1280: PASS, worst min-IoU 0.999982, worst conf delta 2.03e-06, count mismatches 0
  - square 1280x1280: skipped, worst min-IoU None, worst conf delta None, count mismatches 0

## Detection (3 sequences, counts pooled)

| class | GT | TP | FP | FN | recall | precision | F1 |
|---|---|---|---|---|---|---|---|
| athlete | 4849 | 4749 | 265 | 100 | 97.94% | 94.71% | 96.30% |
| ball | 300 | 180 | 89 | 120 | 60.00% | 66.91% | 63.27% |

Ball frame-recall 60.00% over 300 labelled frames; longest miss 4.00s (max), 1.47s (mean).
Ball recall by GT box size: <10px n/a (n=0), 10-20px 56.36% (n=275), >=20px 100.00% (n=25)

## Role / team classifier on GT (3 sequences, held-out frames)

- role accuracy 95.52% over 5183 detections
- team accuracy 98.22% over 4666 pairs (permutation-invariant: team_a/team_b are arbitrary names)
- per-role recall: player 96.55% (n=4666), goalkeeper 99.25% (n=133), referee 81.77% (n=384)

## GS-HOTA, end-to-end Path B (3 sequences, mean of per-sequence scores)

Jersey in IdSim: **ignored**.

| variant | DetA | AssA | GS-HOTA |
|---|---|---|---|
| persons | 52.89% | 71.78% | 58.95% |
| with_ball | 50.42% | 70.62% | 57.75% |

`persons` excludes the ball from GT and predictions; `with_ball` keeps it.

## Latency per frame, ms (tag: `Tesla T4`)

Wall time of each stage run back to back on one thread, pooled over all frames (first 3 frames of each sequence dropped as warm-up). GPU stages end in a device-to-host copy, so their GPU time is included. This is the sequential cost, not a pipelined production figure: plan 1.4's targets assume threaded stages, NVDEC and an RTX 30-series, so they are not comparable.

| stage | n | mean | p50 | p95 | p99 | max | share of total |
|---|---|---|---|---|---|---|---|
| frame load (cv2.imread of a JPEG; production uses NVDEC) | 291 | 10.7 | 10.8 | 12.2 | 13.3 | 14.2 | 12% |
| detector end-to-end (preprocess + inference + NMS + D2H) | 291 | 24.3 | 24.2 | 25.4 | 26.1 | 28.2 | 28% |
| tracker (ByteTrack) | 291 | 6.9 | 6.3 | 8.5 | 9.3 | 183.3 | 8% |
| calibration (LiveCalibrator: NBJW on fresh frames, motion + propagation otherwise) | 291 | 37.5 | 15.9 | 128.8 | 130.7 | 132.7 | 44% |
| pitch projection | 291 | 1.3 | 1.3 | 1.6 | 1.8 | 4.1 | 1% |
| torso descriptors + record assembly | 291 | 5.2 | 5.4 | 6.9 | 8.7 | 9.5 | 6% |
| **WHOLE PIPELINE, per frame** | 291 | 85.9 | 64.7 | 178.1 | 182.3 | 241.0 |  |
| **whole pipeline minus calibration + projection** | 291 | 47.1 | 47.3 | 51.6 | 54.1 | 224.2 |  |

Sequential throughput: **11.65 fps** (mean), 5.62 fps at p95.
Without calibration and projection: 21.2 fps (mean).
Calibration: backend `tensorrt/tensorrt`, calib_every_k=config, mode=config; homography source per frame: fresh 59, propagated 241.
Role/team fit + predict + vote: 49 ms per sequence (~0.49 ms/frame amortised), not in the per-frame rows above.

### Sub-stage breakdown (SNGS-021, 40 frames, first 8 dropped; GPU steps synchronised)

| step | mean | p50 | p95 | max |
|---|---|---|---|---|
| detector: preprocess (CPU letterbox + H2D) | 5.0 | 5.0 | 5.4 | 5.7 |
| detector: inference (GPU, synced) | 17.6 | 17.6 | 17.7 | 17.8 |
| detector: decode + NMS + D2H | 1.9 | 1.9 | 2.1 | 2.2 |
| calibration: image prep (H2D of uint8 frame + on-device resize) | 2.6 | 2.6 | 2.7 | 2.8 |
| calibration: keypoint network (synced) | 32.9 | 33.0 | 34.1 | 34.2 |
| calibration: line network (synced) | 37.6 | 37.7 | 38.4 | 38.5 |
| calibration: heatmap decode + keypoint completion | 39.4 | 39.3 | 40.3 | 40.6 |
| calibration: camera update (add keypoints) | 0.0 | 0.0 | 0.1 | 0.1 |
| calibration: homography (RANSAC) | 0.8 | 0.8 | 0.9 | 1.0 |
| calibration: camera-param voting (only when compute_cam_params) | 19.0 | 18.9 | 21.7 | 23.4 |

Detector steps sum to 24.5 ms; calibration steps sum to 132.5 ms.
Compare with the detector and calibration rows above; a large gap means this breakdown has drifted from the real code and the main-pass rows are the ones to trust.

## Scope notes

- GS-HOTA here is the repo's evaluate.py unchanged. Detections without a pitch projection score as misses.
- No tracklet refinement (needs ReID); calibration is per-frame NBJW (the accuracy reference, plan 7).
- Ball: single highest-confidence detection per frame, one constant track id (plan Phase 6 BallTracker not built).
- Unassociated detections become one-frame singleton tracklets rather than being dropped.
- Roles: AthleteRoleTeamClassifier fitted per sequence on the first --fit_frames frames (0 = whole sequence), then a per-tracklet majority vote. team_a (lower mean pitch x) is labelled 'left'.
- athlete_roles tunables were fitted on SNGS-021 only; every other sequence is a held-out test of them.