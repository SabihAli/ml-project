# The GSR pipeline: how it works and how well

What `gsr_pipeline` does to turn broadcast football video into game-state data, and what each
stage measures. Formerly `IMPLEMENTATION.md` at the repo root (renamed 2026-10-12 and checked
against the code).

Two detector paths exist side by side:

- **Path A** is what `GSRPipeline` loads today: a 4-class YOLOv8n (`yolov8n_gsr.pt`) whose
  classes carry the role (player, goalkeeper, referee, ball).
- **Path B** is the replacement being built: a 2-class YOLO26s (athlete, ball) with roles and
  teams recovered downstream. It runs end to end in the evaluation runner, but isn't wired into
  `GSRPipeline` yet (`docs/path_b/plans/IMPLEMENTATION_PLAN.md` §12.1).

---

## 1. Task

Game State Reconstruction (GSR, SoccerNet 2024): for every frame of a broadcast clip, output
every player, goalkeeper, referee and the ball, with:

- a **track ID** that is stable over time;
- a **pitch position** in metres;
- a **role**, and for players and goalkeepers a **team** (left/right) and **jersey number**.

Output is SoccerNet GSR JSON (`predictions.json`), scored by GS-HOTA (§4).

This repo reimplements the official baseline (`sn-gamestate`) with smaller, simpler parts
and no TrackLab/Hydra dependency:

| stage | official baseline | this repo |
|---|---|---|
| detection | YOLOv8m/YOLO11m | YOLOv8n (Path A) / YOLO26s (Path B) |
| re-identification | PRTReId / BPBreID | OSNet x0.25 (torchreid) |
| tracking | StrongSORT | ByteTrack |
| jersey OCR | MMOCR | EasyOCR |
| calibration | TVCalib | NBJW (No Bells, Just Whistles) |

---

## 2. Stages (batch path, `GSRPipeline.process_sequence`)

Per frame, in order. Each stage takes and returns a pandas DataFrame of detections.

1. **Detection** (`detect.py`, Path A). YOLOv8n with 4 classes. Confidence threshold 0.25 for
   people and 0.15 for the ball, which is the weakest class. The class *is* the role, so no
   later stage re-derives player, goalkeeper or referee.
2. **Appearance** (`reid.py`). OSNet x0.25 embeddings and colour histograms per person crop.
   They're used for team clustering and for tracklet merging in refinement, not by the tracker.
3. **Tracking** (`track.py`). ByteTrack (Ultralytics' implementation): IoU association with a
   Kalman motion model, high/low confidence passes, 30-frame track buffer.
4. **Calibration** (`calibrate.py`, `calibration_live.py`).
   - NBJW predicts pitch keypoints and lines with HRNet heatmaps (vendored in
     `gsr_pipeline/nbjw_calib/`), and fits a homography from the image to the pitch.
   - `LiveCalibrator` runs NBJW every **5** frames (`calibration.calib_every_k`) and carries the
     homography forward by camera motion in between (`camera_motion.py`). It re-runs NBJW early
     on a trigger: low motion inliers, a broken chain, or a large zoom change.
   - Each person's bottom-centre point is projected to pitch metres.
5. **Jersey OCR** (`jersey.py`). EasyOCR, digits only, on player and goalkeeper crops.

Then over the whole sequence, once every frame is done:

6. **Tracklet aggregation** (`tracklet_agg.py`). One role per track by majority vote; one jersey
   number per track by summed OCR confidence.
7. **Team classification** (`team.py`). 2-cluster KMeans on per-tracklet averaged embeddings and
   colour histograms, fitted on player tracklets only. Clusters are named left/right by mean
   pitch x. Goalkeepers get the nearest cluster afterwards; referees get no team.
8. **Refinement** (`refine.py`, optional). Splits a track whose pitch position jumps implausibly,
   merges tracklets that look like the same person (by ReID distance), and linearly
   interpolates gaps of up to 15 frames.
9. **Output** (`output.py`). SoccerNet GSR JSON, and optionally an annotated video.

The **live path** (`GSRPipeline.process_stream`) runs steps 1, 3 and 4 per frame and emits
results immediately. Appearance, jersey OCR and team assignment run on a background worker
pool (`reid_worker_pool.py`, `team_live.py`). Steps 6–8 need a whole sequence, so they don't run
live. See `STREAMING_ARCHITECTURE.md`.

---

## 3. Path B differences

- **Detector:** YOLO26s, 2 classes (`athlete`, `ball`), input 736×1280. It runs through
  `gsr_pipeline/runtime/` (TensorRT → ONNX Runtime GPU → ONNX Runtime CPU fallback chain), not
  Ultralytics.
- **Roles and teams:** `athlete_roles.py` assigns player/goalkeeper by team, referee or other,
  from torso colour fitted per match plus pitch position.
- **Ball:** the main detector finds 59% of balls on the held-out test sequences. A temporal
  specialist (`docs/architecture/BALL_TEMPORAL_ARCH.md`) raises that to 68.6% in simulation; it
  is trained and evaluated but not wired into the pipeline yet (`docs/path_b/plans/BALL_TEMPORAL_PLAN.md`).

---

## 4. Metric: GS-HOTA

`evaluate.py` implements the official formula:

- **LocSim:** a Gaussian on pitch distance, `exp(ln(0.05) · d² / τ²)` with τ = 5 m.
- **IdSim:** 1 only if role, team and jersey number all match.
- Detections are Hungarian-matched to ground truth at 19 thresholds. **DetA** (detection) and
  **AssA** (association over time) combine into GS-HOTA.

---

## 5. Results

The two results below come from different pipeline versions and different data. **Don't
compare them with each other:**

| | Path A (5.1) | Path B (5.2) |
|---|---|---|
| data | full official test set | 3 `valid` sequences × 100 frames |
| jersey in IdSim | counted | ignored, which makes GS-HOTA easier |
| code | an older version, since changed | current |

No run yet scores both paths on the same frames with the same settings. Phase 9 of the Path B
plan records one before Path A is removed (`docs/path_b/plans/IMPLEMENTATION_PLAN.md` §12.2).

### 5.1 Path A, as originally reported (not reproduced)

Mean over the official SoccerNet-GSR test set: **GS-HOTA 30.72%** (DetA 25.07%, AssA 37.65%),
against the official baseline's 22.26%. These are the authors' numbers. They were produced by an
earlier version of the pipeline, with 3-cluster KMeans (teams plus referee) and a spatial
goalkeeper heuristic, both since replaced. They haven't been re-measured against the current
code.

### 5.2 Path B, end to end (Kaggle T4, 2026-10-05)

3 `valid` sequences × 100 frames; jersey ignored in IdSim. Full report:
`docs/results/PATH_B_EVAL_T4_2026-10-05.md`.

| measure | result |
|---|---|
| athlete recall / precision | 97.9% / 94.7% |
| ball recall / precision | 60.0% / 66.9% |
| role accuracy / team accuracy (on GT boxes) | 95.5% / 98.2% |
| GS-HOTA, persons only | 58.95% (DetA 52.9%, AssA 71.8%) |
| GS-HOTA, with ball | 57.75% |
| latency, whole pipeline, T4 | 85.9 ms/frame |

Later measurements, including latency work, are in `docs/path_b/PERF_LOG.md`.

---

## 6. Known limits

- **The ball** is the weakest input to every ball-based event; see §3.
- **Batch-only stages:** aggregation, team KMeans and refinement need the whole sequence.
- **Calibration** drifts between NBJW runs on fast pans. The re-run triggers bound the drift;
  K=5 was chosen on one sequence only.
- **Path B** isn't the default yet, and Path A stays the baseline until Path B's Phase 9 GS-HOTA
  comparison (`docs/path_b/plans/IMPLEMENTATION_PLAN.md` §12).
