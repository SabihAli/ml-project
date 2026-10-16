# AGENTS.md — cv_service

Guidance for AI coding agents (and humans) working in this repository. This file documents the repository **as it actually exists**, verified by reading source. Anything not directly verifiable is marked "Unknown."

## Project Overview

`cv_service` is a standalone, lightweight Python reimplementation of the **SoccerNet Game State Reconstruction (GSR)** baseline (package name: `gsr_pipeline`). Given broadcast football video frames, it performs end-to-end:

- Athlete/ball detection
- Multi-object tracking
- Pitch (camera) calibration and 2D→pitch-coordinate projection
- Team/referee/goalkeeper role classification
- Jersey number OCR
- Tracklet aggregation and temporal refinement
- Output as SoccerNet GSR-format JSON and an annotated video
- Official **GS-HOTA** metric evaluation against ground truth

Per `gsr_pipeline/__init__.py`'s module docstring, this is a deliberate "nano-scale" substitution of a heavier baseline (`sn-gamestate`): YOLOv8n instead of YOLOv8m/YOLO11m, torchreid OSNet instead of PRTReId/BPBreID, EasyOCR instead of MMOCR, ByteTrack instead of StrongSORT, and no TrackLab/Hydra framework dependency. `docs/architecture/GSR_PIPELINE.md` (formerly `IMPLEMENTATION.md`) documents the approach and reports experimental results (Mean GS-HOTA 30.72% vs. official baseline 22.26%) — treat these numbers as the authors' reported results, not independently verified here.

A former `video_ingestion/` notebook (SoccerNet video download) was removed on 2026-10-12 as unused; it is in git history (commit `0bc1857`). Dataset download is now in `docs/guides/BALL_TEMPORAL_GUIDE.md` §3.

**Evolution (partially implemented):** `docs/architecture/PIPELINE_EVOLUTION.md` is the plan for turning this offline benchmark pipeline into the production CV service — the sole owner of the video→structured-event surface in the wider BallKnowledge system, emitting events (player/referee/ball position, identity, possession, turnovers, and eventually classifier-based events like pass/shot/tackle/touch) into `data_ingestion` rather than writing `predictions.json` files. The live position/identity path is built: `process_stream` → `api/` (HTTP + WS) → `data_ingestion`'s `CVPipelineSource`. Live `POSSESSION`/`TURNOVER` and the whole classifier tier are not. The batch pipeline (JSON-file output) described in most of this file still exists alongside it. See "Production architecture" under Architecture below.

## Tech Stack

- **Language**: Python (no version pinned anywhere in the repo — Unknown minimum version; `torch>=1.13.0` suggests Python 3.8+).
- **Detection**: Ultralytics YOLOv8 (`ultralytics` package), custom fine-tuned 4-class model (`yolov8n_gsr.pt`, ~17.5 MB, committed at repo root).
- **Re-Identification**: `torchreid` (OSNet, `osnet_x0_25`, Market1501-pretrained weights by default).
- **Tracking**: Ultralytics' internal BYTETracker, wrapped by a custom `ByteTracker` class with version-compatibility shims.
- **Pitch calibration**: Vendored HRNet-based package `gsr_pipeline/nbjw_calib/` (its own `setup.py`, independently pip-installable; checkpoints auto-downloaded from Zenodo at runtime).
- **Jersey OCR**: EasyOCR.
- **Clustering**: scikit-learn (KMeans, 2 clusters) for team-side separation among already-detector-classified "player" tracklets only.
- **Data manipulation**: pandas, numpy — the pipeline's core data structure is a per-frame/per-detection pandas DataFrame threaded through every stage.
- **CV/video I/O**: OpenCV (`opencv-python`).
- **Config format**: YAML (`pyyaml`), loaded once and passed as nested dicts.
- **Web framework**: FastAPI + uvicorn (`api/`, added for `docs/architecture/STREAMING_ARCHITECTURE.md` §3's `/streams` endpoints — the batch `gsr_pipeline/` package itself remains framework-free). **Still no database, message queue, or Docker.** The offline batch pipeline (`gsr_pipeline/`) is unaffected by this — `api/` is an additive layer on top, not a rewrite.

## Repository Structure

```
cv_service/
├── gsr_pipeline/            # Core Python package — all pipeline logic
│   ├── configs/default.yaml # Single source of runtime configuration
│   ├── nbjw_calib/           # Vendored pitch-calibration sub-package (own setup.py)
│   ├── pipeline.py           # GSRPipeline orchestrator (main integration point)
│   ├── run.py                 # CLI entry point for inference
│   ├── detect.py, reid.py, track.py, calibrate.py, jersey.py, team.py,
│   │   tracklet_agg.py, refine.py, output.py, visualize.py    # Pipeline stages
│   ├── evaluate.py           # GS-HOTA metric CLI
│   ├── visualize_only.py     # Standalone re-visualization CLI (marked "Temporary script")
│   ├── train_yolo.py         # Standalone YOLO fine-tuning / label-conversion script
│   ├── rename_img1_to_images.py, subset_dataset.py  # One-off dataset utility scripts
│   └── utils.py               # Shared helpers (frame loading, bbox math, color histograms)
├── data/                      # Gitignored, local only: datasets/SoccerNetGS/, outputs/, pretrained_models/ (calibration checkpoints, auto-downloaded)
├── docs/                      # Index: README.md's Documentation section
│   ├── architecture/          # Service-level design
│   │   ├── GSR_PIPELINE.md           # How the pipeline works, the metric, results (was IMPLEMENTATION.md)
│   │   ├── PIPELINE_EVOLUTION.md     # Event taxonomy, revision semantics, open issues
│   │   ├── STREAMING_ARCHITECTURE.md # Live path (Path A) design + build status (§8)
│   │   ├── BALL_TEMPORAL_ARCH.md     # Temporal ball specialist (c5c): model, training history, state
│   │   └── REQUIREMENTS.md
│   ├── path_b/                # Path B (2-class detector) work
│   │   ├── plans/IMPLEMENTATION_PLAN.md  # Path B plan, per-phase status in §12.1
│   │   ├── plans/OPTIMIZATION_PLAN.md    # Path B latency plan, build status in §7
│   │   ├── plans/BALL_TEMPORAL_PLAN.md   # Ball specialist plan (temporal, point output)
│   │   ├── DETECTOR_AND_DATA_BLOCKERS.md  # Why Path B, ball-detection options
│   │   └── PERF_LOG.md               # Dated performance/accuracy log
│   ├── guides/                # Step-by-step run guides (BALL_TEMPORAL_GUIDE.md: local GPU; _KAGGLE_GUIDE.md)
│   └── results/               # Evaluation reports (PATH_B_EVAL_T4_2026-10-05.md)
├── notebooks/path_b_eval.py   # Kaggle runner: Path B parity, detection, roles, GS-HOTA, latency
├── scripts/                   # Path B export/engine/parity tools, YOLO26 training + inference
│   ├── ball_temporal/         # Ball specialist v2 scripts: main-detector cache, 3-frame dataset, heatmap model, training (docs/path_b/plans/BALL_TEMPORAL_PLAN.md); v1 ball scripts sit in scripts/ as a record
│   └── eval/ball_det/         # Ball specialist evaluation: test gates 1-3 (eval_ball_temporal.py)
├── tests/                     # pytest, pure logic only (see Testing)
├── runs/yolo26s_athlete_ball/ # Path B detector: best.pt + exported ONNX and sidecars (tracked); everything else under runs/ is gitignored
├── yolov8n_gsr.pt             # Committed fine-tuned detector weights
├── requirements.txt
├── README.md
└── .gsr_venv/                 # A committed-looking but gitignored local virtualenv directory
```

**Important**: `data/` (datasets, outputs, calibration checkpoints) is gitignored in full: no dataset is tracked. The SoccerNet GSR splits are downloaded per `docs/guides/BALL_TEMPORAL_GUIDE.md` §3. The small `valid_sample/` that used to be tracked was removed from the repo on 2026-10-12 (it is in git history). `runs/` is gitignored except the Path B detector in `runs/yolo26s_athlete_ball/` (weights, ONNX files and their sidecars) and the ball specialist `c5c` in `runs/ball_temporal/c5c/` (weights, ONNX + sidecar, training args, history, curves; not `last.pt`).

## Architecture

The system is a **linear, per-frame-then-per-sequence batch pipeline**, orchestrated by `GSRPipeline` (`gsr_pipeline/pipeline.py`). There is no service/API layer, no persistent database, and no background job queue — it runs as a one-shot CLI process over a fixed, already-complete directory of frame images (`img1/`). It is **not currently a streaming/live-video system** — see "Batch-only design vs. real-time goal" below, which is a known gap agents must be aware of before extending this for production.

### Data flow

Processing happens in two phases. Phase 1 runs once per frame, in order, as frames are read. Phase 2 only starts after **every** frame in the sequence has been through Phase 1 — it needs the complete set of per-frame detections before it can do anything.

**Phase 1 — per-frame (runs frame by frame, in a loop over the whole sequence):**

1. **Frame loading** (`utils.load_frames_from_dir`) reads all images from a sequence's `img1/` (or `images/`) directory into memory up front — the whole sequence's frame list is built before the per-frame loop starts (`pipeline.py`'s `process_sequence`).
2. For each frame, in this order:
   - **Detection** (`detect.YOLOv8nDetector`) → DataFrame of boxes with `class_id` (player=0, goalkeeper=1, referee=2, ball=3).
   - **ReID + role classification** (`reid.OSNetReID._assign_roles`) → appends 512-d embeddings, `role_detection`, and `role_confidence`. Role is a direct pass-through of the detector's `class_id` (player/goalkeeper/referee/ball), not a separate heuristic — see Architecture Rules below.
   - **Tracking** (`track.ByteTracker`) → assigns `track_id` per detection via IoU-matched BYTETracker output.
   - **Calibration** (`calibrate.NBJWKeypointDetector` + `NBJWHomography`) → per-frame homography; `project_detections` maps bbox bottom-center to pitch (meter) coordinates, expanded into DataFrame columns.
   - **Jersey OCR** (`jersey.EasyOCRJerseyDetector`) → digit-only jersey number per player/goalkeeper crop.
   - All per-frame DataFrames are concatenated into one sequence-level `full_df` once every frame is done.

**Phase 2 — sequence-level post-processing (only starts after Phase 1 has processed the entire sequence; operates on the whole `full_df`, not one frame at a time):**

3. **Tracklet aggregation** (`tracklet_agg.TrackletAggregator`) — in simple terms: a "tracklet" is one player's/referee's/ball's full trail of detections across the sequence, linked by a shared `track_id`. This step looks at every detection belonging to one `track_id` and picks a single consensus value per attribute for the *whole* tracklet — the most common `role_detection` value wins by majority vote (e.g. if a player was misdetected as referee in 2 of 300 frames, the other 298 "player" votes win), and `jersey_number` is decided by summing OCR confidence per candidate number and taking the highest-confidence one. It then overwrites every row of that tracklet with the winning values, so a single track has one consistent role/jersey for its entire lifetime instead of flickering frame to frame. **This requires having already seen every frame the tracklet appears in** — it can't produce a final answer until the tracklet is over (or, in a live setting, until you decide to close/finalize it).
4. **Team classification** (`team.TeamClassifier`) — KMeans (2 clusters) on each player-tracklet's *averaged* color histogram + ReID embedding, applied only to tracklets whose `role_detection == "player"`; the two clusters are then labeled "left"/"right" by comparing their average pitch-x position. Referees and goalkeepers are never clustered — they are already classified by the detector and pass through unchanged. **This also requires full tracklets**: it averages a tracklet's appearance across all its frames before clustering, and KMeans itself needs the complete set of tracklets in the sequence to have anything to split into two groups.
5. **Refinement** (`refine.refine_tracks`, optional, config-gated) — repairs tracking mistakes after the fact: splits a tracklet if its pitch position jumps implausibly far frame-to-frame (a likely identity swap), greedily merges tracklets that look like the same person but got assigned different IDs, and interpolates short gaps where detection dropped out for a few frames. All three operations compare a tracklet's *later* frames against its *earlier* frames (or against other tracklets across the whole sequence) to decide what to do, so this is inherently a whole-sequence, look-ahead-and-look-behind operation, not a per-frame one.
6. **Output** (`output.save_results_json`) — writes SoccerNet-GSR-format `predictions.json`; optionally `visualize.generate_video` renders an annotated MP4.
7. **Evaluation** (`evaluate.py`, separate CLI) — computes GS-HOTA (`LocSim` × `IdSim`, Hungarian-matched across 19 alpha thresholds) between `predictions.json` and ground-truth `Labels-GameState.json`.

### Batch-only design vs. real-time goal (known gap — read before building a streaming version)

This pipeline was built to score well on the offline SoccerNet-GSR benchmark (fixed-length recorded clips, full dataset available up front), and its design reflects that: **steps 3–5 above are not runnable per-frame or incrementally as currently written.** They take the full sequence's DataFrame as input and return a fully-processed full sequence DataFrame — there is no windowed/streaming variant.

Concretely, if `cv_service` needs to serve near-real-time data from a live stream, each of these needs a genuine redesign, not a config change:
- **Team classification** needs *some* amount of buffered history before it can cluster reliably — averaging one frame's appearance per player is noisy, and KMeans on a single frame's worth of players is unstable. A real-time version would need a rolling buffer (e.g. first N seconds, or first M frames per player) to fit initial clusters, then classify new tracklets against those fixed cluster centroids incrementally, rather than re-clustering the whole match after the fact.
- **Tracklet aggregation** (role/jersey consensus) would need to become an online/incremental vote (update a running tally per `track_id` as each new detection arrives, and expose a "best guess so far" instead of waiting for the tracklet to end).
- **Refinement** (split/merge/interpolate) would need to operate on a bounded look-back window (e.g. last N frames) instead of the entire sequence, trading off some correction accuracy for the ability to emit results incrementally.

Agents should **not** assume steps 3–5 can be trivially "called per frame" — their algorithms as written depend on already having the full tracklet/sequence. Treat any request to make this pipeline real-time as a design task requiring explicit discussion of these trade-offs, not a quick patch.

### Production architecture (`docs/architecture/PIPELINE_EVOLUTION.md` — partially implemented)

The production direction resolves the real-time gap above not by making team/jersey/refinement incremental, but by **not gating latency-sensitive output behind them at all**:

- **Fast/slow split (built for the live path).** `process_stream` emits position/role events every frame from detect → role → track → calibrate, while ReID, jersey OCR, and live team assignment run on `ReidWorkerPool` and arrive later as appearance events. Batch `process_sequence` is still linear. Note that calibration sits on the fast path, which `PIPELINE_EVOLUTION.md` §2 flags as unresolved.
- **Two event tiers**:
  - *Rule-based*: `PLAYER_POSITION` (players and goalkeepers), `REFEREE_POSITION`, `BALL_POSITION`, and `PLAYER_IDENTIFIED` are **emitted live**. `POSSESSION`/`TURNOVER` exist only in the batch path (`possession.py`).
  - *Classifier-based* (not started): `PASS`, `SHOT`, `CARRY`, `DRIBBLE`, `TACKLE`, `TOUCH`, `INTERCEPTION`, `DUEL` — planned as a sequential video model over short cropped clips, with a fast short window revised later by a long window. No classifier-tier event may be emitted until the revision semantics in `PIPELINE_EVOLUTION.md` §3.4 are implemented.
- **Integration is a network boundary, not an in-process import.** cv-service runs as its own service (`api/`: `POST /streams`, WS event feed). `data_ingestion`'s `CVPipelineSource` (`data_ingestion/src/data_ingestion/cv_pipeline_source.py`, a separate repo) registers a stream over HTTP, consumes the WS feed, and translates it into `DataEvent`s; it does not import `gsr_pipeline`. This replaced the original in-process plan so that torch/ultralytics/easyocr never enter `data_ingestion`, while `data_ingestion` stays the only Kafka producer. The WS feed has no replay buffer, so events dropped before `CVPipelineSource` are lost.
- **Every detector role must map to exactly one event type.** Referees used to fall through to `BALL_POSITION`, publishing phantom balls; if the detector gains or renames a class, update `_ROLE_EVENT_TYPES` in `cv_pipeline_source.py` in the same change.

Do not assume `pipeline.py`'s linear ordering (Architecture Rule #3) holds for the live path — it describes `process_sequence`. For build status, `docs/architecture/STREAMING_ARCHITECTURE.md` §8 is authoritative. For the event taxonomy, revision semantics, and open design issues, `docs/architecture/PIPELINE_EVOLUTION.md` is authoritative. Where either disagrees with the code, the code wins and the doc should be fixed.

Every stage communicates via a shared **pandas DataFrame** passed by reference through the pipeline — this is the primary internal contract. Module functions/classes generally take and return `pd.DataFrame`.

## Important Files

- `gsr_pipeline/pipeline.py` — the integration point; read this first to understand stage order and DataFrame contract.
- `gsr_pipeline/run.py` — CLI entry point (`python -m gsr_pipeline.run`).
- `gsr_pipeline/evaluate.py` — CLI entry point for metric scoring (`python -m gsr_pipeline.evaluate`).
- `gsr_pipeline/possession.py` — rule-based `POSSESSION`/`TURNOVER` assignment (build-order step 2 of `docs/architecture/PIPELINE_EVOLUTION.md` §5). Currently an **offline post-processing step** over a completed sequence's final DataFrame (same shape as `predictions.json`), not yet wired into a live detect+track stream — see its module docstring for why (team assignment requires the whole-sequence Phase 2 step). Run via `gsr_pipeline/possession_demo.py`.
- `gsr_pipeline/frame_source.py` — `FrameSource` abstraction: `ImageDirFrameSource`, `VideoFileFrameSource`, `LiveBroadcastFrameSource` (RTMP/RTSP/HLS via OpenCV's FFmpeg backend, transparent reconnect-on-drop with exponential backoff, `disconnect_reason` surfaced once retries are exhausted — validated only via synthetic `cv2.VideoCapture` failure injection, `gsr_pipeline/validate_live_broadcast.py`, 12/12 checks; no real live feed exists in this dev environment). `GSRPipeline.process_sequence` accepts either a `FrameSource` instance or a plain path string (auto-wrapped in `ImageDirFrameSource`, preserving prior behavior exactly). Also implements `FrameSource.validate()` (FR-002's black/frozen/unreachable/undecodable checks). Regression-verified: pre- and post-refactor `predictions.json` are byte-for-byte identical on the then-committed `valid_sample` sequence (since removed from the repo). See `docs/architecture/STREAMING_ARCHITECTURE.md` §1.
- `gsr_pipeline/pitch_view_gate.py` — `PitchViewGate`: per-frame "is this actually a pitch view" gate (calibration succeeded + minimum player detections), used by `team_live.py`'s bootstrap and, eventually, `process_stream`'s ongoing cutaway handling. See `docs/architecture/STREAMING_ARCHITECTURE.md` §1.7.2.
- `gsr_pipeline/team_live.py` — `LiveTeamAssigner`: incremental, per-frame, per-track live team assignment (classify once at track birth from a single frame, lock permanently above a confidence threshold, adaptive confidence-weighted centroids, no whole-sequence dependency). Supersedes the batch `team.py`'s whole-sequence KMeans for the live/streaming case. Now wired into `GSRPipeline.process_stream` (`pipeline.py`) and verified end-to-end against the real model stack, not just synthetic data — see `docs/architecture/STREAMING_ARCHITECTURE.md` §1.7 and §8 build order.
- `gsr_pipeline/reid_worker_pool.py` — `ReidWorkerPool`: background thread pool decoupling ReID embedding extraction + jersey OCR (both expensive) from `process_stream`'s fast per-frame loop (detect→role→track→calibrate), so position/role events are never gated behind them. Batch mode (`process_sequence`) is unaffected — it still calls `reid.extract_appearance_features`/`jersey.process_frame` inline, synchronously. See `docs/architecture/STREAMING_ARCHITECTURE.md` §1.7.6.
- `gsr_pipeline/pipeline.py`'s `GSRPipeline.process_stream` — the live-mode counterpart to `process_sequence`: Phase 1 only (no tracklet aggregation/batch team/refinement, since those need a complete sequence a live stream never has), wired to `PitchViewGate`, `LiveTeamAssigner`, and `ReidWorkerPool`. Shares `_detect_track_calibrate` (the fast-path stages) with `process_sequence` — extracted as a private method both call, so there is one source of truth for that subset, not two copies. Verified end-to-end (real detector/ReID/calibration/jersey models, driven by a real `.mp4` via `VideoFileFrameSource` as a live-source stand-in) via `gsr_pipeline/validate_process_stream.py` — 7/7 checks passing.
- `gsr_pipeline/evaluate_team_live.py` — live-mode team-assignment accuracy harness (docs/architecture/STREAMING_ARCHITECTURE.md §1.7.8), the live counterpart to `evaluate.py`'s batch GS-HOTA. Replays a labeled sequence through `process_stream`, matches predictions to `Labels-GameState.json` by nearest position + role, reports accuracy among locked vs. all matched predictions. First real run (20 frames, the `valid_sample` clip, since removed from the repo): 100% accuracy among locked predictions, 80.1% lock coverage — see `docs/architecture/STREAMING_ARCHITECTURE.md` §8 step 5 for the full numbers and caveats (small sample, not a production benchmark).
- `api/` (new top-level package, sibling to `gsr_pipeline/`) — the FastAPI HTTP + WS layer implementing `docs/architecture/STREAMING_ARCHITECTURE.md` §3-4: `POST /streams`, `POST /streams/{stream_id}/stop`, `GET /health`, `GET /streams/{stream_id}/events` (WS). `api/stream_manager.py` owns the IDLE/VALIDATING/STREAMING/STOPPING state machine and the background thread running `GSRPipeline.process_stream`; `api/auth.py` enforces a static API-key (`X-CV-Service-Key` header for REST, `?api_key=` query param for the WS handshake — same `CV_SERVICE_API_KEY` env var, fails closed if unset, never silently open); `api/schemas.py` are the pydantic request/response models; `api/event_broadcaster.py` (`EventBroadcaster`) is the thread-safe hand-off from `process_stream`'s background thread to the WS connection's asyncio event loop (bounded queue, drop-oldest backpressure, single-consumer enforcement, no replay buffer); `api/event_serializer.py` converts `process_stream`'s raw DataFrame-carrying event dicts into the JSON wire shape. **First web framework in this repo** (`fastapi`, `uvicorn` added to `requirements.txt`) — previously "no web framework, offline batch pipeline only" per this file's own prior description. Verified end-to-end against a real running server: `api/validate_api.py` (REST, 11/11 checks), `api/validate_ws.py` (WS feed — auth, single-consumer rejection, event shape, clean close on stop, 9/9 checks). The WS wire format is a deliberate scoped-down v1 (raw "fast"/"appearance" per-frame detections, not the full `event_type`/`EventType`/`payload` taxonomy originally proposed) — see `docs/architecture/STREAMING_ARCHITECTURE.md` §4.2 for why, and build-order step 9 for where that translation is deferred to.
- `gsr_pipeline/athlete_roles.py` — `AthleteRoleTeamClassifier`: turns role-agnostic `athlete` detections into `player_team_a/b`, `goalkeeper_team_a/b`, `referee`, `other` from torso colour (fitted per match) plus pitch position. **Not wired into `GSRPipeline`**; it exists for the athlete/ball detector (`runs/yolo26s_athlete_ball`, which has no roles). Scored by `gsr_pipeline/evaluate_athlete_roles.py`. Tunables in `configs/default.yaml`'s `athlete_roles` section were tuned on one sequence. See `docs/path_b/DETECTOR_AND_DATA_BLOCKERS.md` §5.
- `gsr_pipeline/calibration_live.py` — `LiveCalibrator`, which both `process_sequence` and `process_stream` now calibrate through. It runs NBJW every `calibration.calib_every_k` frames or on a trigger, and propagates H by camera motion (`gsr_pipeline/camera_motion.py`) in between. `inline` or `async` mode. The default is K=5 inline, chosen on SNGS-021 only; K=1 inline is the old per-frame behaviour and the accuracy reference. It adds `h_source` and `h_age_frames` DataFrame columns. Choose K with `python -m gsr_pipeline.evaluate_calib_propagation`. NBJW's camera-parameter voting is off by default (`calibration.compute_cam_params`); it fed only predictions.json's `camera_parameters` and was most of calibration's cost. `scripts/export_nbjw.py` exports the NBJW networks to ONNX and builds their TRT engines, for `calibration.backend`. See `docs/path_b/plans/IMPLEMENTATION_PLAN.md` §7 build status.
- `gsr_pipeline/nbjw_fast_decode.py` — faster drop-in for the vendored NBJW heatmap decode (separable max-pool, numpy instead of Python loops, inline two-point line fit). It gives the same keypoints, and is on by default via `calibration.fast_decode`. **Do not edit `nbjw_calib/utils/utils_heatmap.py`**: it is the reference that `scripts/parity_pipeline.py` and `tests/test_nbjw_fast_decode.py` compare against. Every "exact" change in `docs/path_b/plans/OPTIMIZATION_PLAN.md` must pass `scripts/parity_pipeline.py` before it ships.
- `scripts/train_yolo26_gsr.py`, `scripts/infer_athlete_video.py` — standalone, Kaggle-friendly scripts (no `gsr_pipeline` import): YOLO26 fine-tuning + evaluation, and athlete/ball inference with per-class confidence thresholds to an annotated mp4. Need `ultralytics>=8.4.0` for YOLO26; `requirements.txt` only says `>=8.0.0`, and the system Python 3.13 here has 8.3.174, which lacks YOLO26.
- `gsr_pipeline/configs/default.yaml` — the single runtime configuration file; all module constructors are instantiated with `**cfg.get("<section>", {})`.
- `gsr_pipeline/__init__.py` — package docstring explains design rationale vs. the original baseline; exports `GSRPipeline` only.
- `gsr_pipeline/nbjw_calib/setup.py` — the vendored calibration sub-package must be `pip install -e`'d separately (per README installation steps); it is NOT installed automatically by `requirements.txt`.
- `yolov8n_gsr.pt` — committed fine-tuned detector weights (repo root), referenced by `configs/default.yaml`'s `detector.model_path`.

## Development Setup

From README.md (verified against `requirements.txt` and directory layout):

```bash
# 1. Create and activate a venv (repo convention names it .gsr_venv)
python -m venv .gsr_venv
.gsr_venv\Scripts\activate        # Windows
source .gsr_venv/bin/activate      # Linux/macOS

# 2. Install pipeline dependencies
pip install -r requirements.txt

# 3. Install the vendored calibration sub-package (required — not in requirements.txt)
cd gsr_pipeline/nbjw_calib
pip install -e .
cd ../..
```

Calibration model checkpoints (`data/pretrained_models/calibration/SV_kp`, `SV_lines`) are auto-downloaded from Zenodo at first run by `calibrate.py`'s `_ensure_download()` if not already present locally.

## Common Commands

Only commands actually present in README.md / source are listed; no test, lint, or build tooling exists in this repo (see Testing and Coding Conventions below).

**Run inference on a sequence:**
```bash
python -m gsr_pipeline.run --sequence_dir data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021 --output_dir data/outputs/SNGS-021 --max_frames 50
```
Other `run.py` flags: `--config` (default `gsr_pipeline/configs/default.yaml`), `--save_video`, `--no-refine`, `--device cpu|cuda`.

**Evaluate against ground truth (GS-HOTA):**
```bash
python -m gsr_pipeline.evaluate --gt_path path/to/Labels-GameState.json --pred_path data/outputs/SNGS-021/predictions.json
```

**Per-class detector accuracy** (recall/precision per class, person role confusion, ball frame-recall and longest miss — isolates the detector, unlike GS-HOTA):
```bash
python -m gsr_pipeline.evaluate_detection --sequence_dir data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021 --output_json data/outputs/SNGS-021_detection_eval.json
```
Detector only, so it runs on CPU in a few minutes for the 750-frame sample.

**Score athlete role/team classification** (fits `athlete_roles.py` on the first `--fit_frames` frames, scores the held-out rest; uses ground-truth boxes and pitch positions, so it isolates the classifier from detector and calibration errors):
```bash
python -m gsr_pipeline.evaluate_athlete_roles --sequence_dir data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021
```

**Athlete/ball inference video** (per-class thresholds; needs `ultralytics>=8.4.0`; device picked automatically):
```bash
python scripts/infer_athlete_video.py --source <path>/SNGS-033 --athlete_conf 0.4 --ball_conf 0.10
```

**Re-generate visualization only** (from an existing `predictions.json`, without rerunning the pipeline):
```bash
python -m gsr_pipeline.visualize_only  # see argparse flags in the script itself
```

**Compute possession/turnover events from an existing `predictions.json`:**
```bash
python -m gsr_pipeline.possession_demo --json_path data/outputs/SNGS-021_smoke/predictions.json --output_path data/outputs/SNGS-021_smoke/possession_events.json
```
Writes `{"possession": [...], "turnover": [...]}`. Tunables live in `configs/default.yaml`'s `possession` section (`proximity_x_m`, `proximity_y_m`, `hold_frames_threshold`) — see that section's comments for why x/y thresholds differ.

**Train/fine-tune the YOLO detector** (standalone, not part of the inference pipeline):
```bash
python -m gsr_pipeline.train_yolo --mode {convert_one,convert_all,build_yaml,train,all}
```

**Run the cv-service API** (loads all models at startup, may take ~a minute):
```bash
CV_SERVICE_API_KEY=some-secret python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
`CV_SERVICE_API_KEY` is required — the app fails closed (500 on every request) rather than silently allowing unauthenticated access if it's unset. See `docs/architecture/STREAMING_ARCHITECTURE.md` §3 for endpoint details.

**Validate the API layer end-to-end** (requires the server above already running):
```bash
python -m api.validate_api --api_key some-secret --mp4 data/outputs/SNGS-021_eval75/annotated.mp4
```

**Validate the WS event feed end-to-end** (requires the server above already running; needs the `websockets` package, already pulled in transitively by `uvicorn[standard]`):
```bash
python -m api.validate_ws --api_key some-secret --mp4 data/outputs/SNGS-021_eval75/annotated.mp4
```

**Validate `LiveBroadcastFrameSource`'s reconnect logic** (synthetic failure injection, no real RTMP/RTSP/HLS feed needed or used):
```bash
python -m gsr_pipeline.validate_live_broadcast
```

**Path B detector (`docs/path_b/plans/IMPLEMENTATION_PLAN.md`, Phase 1a):**
```bash
# Export the ONNX (release step, once per model version; needs ultralytics>=8.4)
python scripts/export_athlete_detector.py                    # 736x1280, rectangular
python scripts/export_athlete_detector.py --imgsz 1280 1280  # square, for the A/B

# Build the TensorRT engine ahead of time (optional since 2026-10-15: the pipeline builds a
# missing engine itself at startup, see detector_b/calibration.auto_build_engine; exits 2 without CUDA)
python scripts/build_trt_engine.py

# Parity gate: runner vs Ultralytics, box for box, 5 sequences, both input sizes
python scripts/parity_gate.py --frames_per_sequence 30 --backend ort_cpu --device cpu

# Score the 2-class detector. --start_frame 168 skips SNGS-021's kickoff, which every
# model misses and which otherwise swamps the ball metrics.
python -m gsr_pipeline.evaluate_detection --path b --backend ort_cpu --device cpu \
    --sequence_dir data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021 --start_frame 168
```

There is **no** lint, format, type-check, build, or Docker command anywhere in this repository — do not invent one. For tests, see the Testing section: a narrow `pytest tests/` suite exists for Path B's runtime logic only.

## Coding Conventions

Observed, not prescribed elsewhere in the repo:

- **Module organization**: one class per pipeline stage, named `<Thing>Detector`/`<Thing>ReID`/`<Thing>Classifier`/`<Thing>Tracker`/`<Thing>Aggregator`, each exposing a `process_frame`/`process_sequence`/`update`/`compute` method that takes and returns a `pandas.DataFrame` (or augments one in place logically, though not literally in-place — new DataFrames are typically built and returned).
- **Config passing**: sections of the YAML config are unpacked into constructors via `**cfg.get("<section>", {})` — constructor keyword args must match YAML keys exactly.
- **Logging**: `logging.getLogger(__name__)` per module, `log.info`/`log.warning` for status; `logging.basicConfig` is configured once, centrally, in `run.py`'s `setup_logging()`. Utility/training scripts instead use plain `print()`.
- **Error handling**: optional heavy dependencies (`torchreid`, `easyocr`) are imported lazily inside `__init__`/functions and raise a chained `ImportError` with an install hint rather than failing at module import time. Per-row/per-frame fallible operations (pitch projection, keypoint drawing) are wrapped in narrow `try/except` that log/skip rather than aborting the whole run. CLI scripts (`evaluate.py`) validate paths with explicit `if not Path(...).exists(): print(...); return` rather than raising.
- **Defensive validation**: `utils.require_columns()` raises `KeyError` listing missing vs. available columns — the repo's one explicit "fail fast with a clear message" pattern.
- **CLI conventions**: every entry-point script uses `argparse` with `--flag` style, runnable via `python -m gsr_pipeline.<module>`.
- **Constants over config**: one-off/training scripts (`train_yolo.py`, `subset_dataset.py`) hardcode paths and settings as module-level constants rather than using `configs/default.yaml` — this differs from the main inference pipeline's config-driven convention. Do not assume all scripts share one config source.
- **Docstrings**: module-level docstring at top of most `gsr_pipeline/*.py` files stating purpose in one line; not consistently present in the small utility scripts.
- **Git commit convention**: `type: short description`, lowercase, e.g. `feat:`, `fix:`, `docs:`, `refactor:`, `data:`, `move:` (verified via `git log`).

## Testing

**A `pytest` suite exists as of 2026-09-26 (Path B Phase 1a)** — approved by the user on 2026-09-25, `docs/path_b/plans/IMPLEMENTATION_PLAN.md` §12. It is deliberately narrow:

```bash
python -m pytest tests/ -q        # 59 tests, ~20s, no weights / no GPU / no data
```

`tests/` covers **pure logic only**: backend tier selection with fake backends, artifact/sidecar contract refusals, letterbox geometry, NMS + box-decode post-processing (`test_backend_selection.py`, `test_postprocess.py`), `LiveCalibrator` triggers, propagation and the NBJW-storm cooldown (`test_calibration_live.py`), fast NBJW decode vs the vendored reference (`test_nbjw_fast_decode.py`), and ByteTrack stepping on empty frames (`test_track.py`). Anything needing weights, a GPU or a dataset stays in `validate_*.py` / `evaluate_*.py`, outside pytest's default run — that split is intentional, so the suite stays runnable anywhere in seconds.

Everything outside Path B's runtime still has **no automated tests**. For those, verification remains manual:
- Running `gsr_pipeline.run` on a sequence from `data/datasets/SoccerNetGS/gamestate-2024/` (a local download; nothing is committed) and inspecting output.
- Running `gsr_pipeline.evaluate` to check GS-HOTA against ground truth.

So "tests pass" means the Path B runtime's logic passes — it is not evidence that the pipeline as a whole works. Do not extend pytest to stages outside Path B's runtime without asking.

## Configuration

- **`gsr_pipeline/configs/default.yaml`** is the sole runtime config file, covering `detector`, `reid`, `tracker`, `calibration`, `jersey`, `team`, `refiner`, `output`, `possession`, `pitch_view_gate`, `team_live`, and `reid_worker_pool` sections. `run.py --device` overrides device settings across `detector`/`reid`/`calibration`/`jersey` sections at runtime. `possession`, `pitch_view_gate`, and `team_live` hold tunables for their respective modules — no values are hardcoded in the modules themselves, per repo convention. `team_live` in particular carries several explicitly provisional numeric defaults (`confidence_threshold`, `bootstrap_window_s`, etc.) that need tuning against real broadcast footage, which doesn't exist in this repo yet — see the config file's inline comments.
- **Environment variables**: `CV_SERVICE_API_KEY`, required by the `api/` service (it fails closed without it). `gsr_pipeline` itself reads none. Downloading the GSR splits needs no SoccerNet password (`docs/guides/BALL_TEMPORAL_GUIDE.md` §3).
- **Model weights**: `yolov8n_gsr.pt` (committed, repo root) and calibration checkpoints under `data/pretrained_models/calibration/` (gitignored, auto-downloaded on first run).
- **TensorRT engines** (`<onnx stem>_fp16.engine` beside each ONNX, with a `.json` sidecar): built automatically at startup when missing, controlled by `detector_b.auto_build_engine` and `calibration.auto_build_engine` (default true; `gsr_pipeline/runtime/trt_runner.ensure_engine`). Needs CUDA and the `tensorrt` package; the first build takes minutes. An engine that exists but was built for another GPU/TRT version is refused (fallback to ORT GPU), never rebuilt automatically.
- No secrets, API keys, or credentials are present in tracked files.

## Architecture Rules

Invariants an agent must preserve unless explicitly asked to change them:

1. **DataFrame contract**: every pipeline stage takes and returns a `pandas.DataFrame` with a stable, growing set of columns (`image_id`, `track_id`, `class_id`, `role`, `team`, `jersey_number`, `bbox_*`, embeddings, pitch coordinates, etc.). Renaming or restructuring these columns will silently break every downstream stage — there is no schema validation beyond `utils.require_columns()`.
2. **Config-driven construction**: pipeline module constructors accept `**kwargs` matched to `configs/default.yaml` section keys. Adding a required constructor arg without a corresponding YAML default will break `GSRPipeline.__init__`.
3. **Stage order in `pipeline.py`** (detect → reid/role → track → calibrate → jersey → [sequence-level] tracklet_agg → team → refine) encodes real dependencies (e.g., team classification needs aggregated tracklets; refinement needs team/role already assigned) — do not reorder without understanding why each stage needs its predecessor's output.
4. **`nbjw_calib` is a separately-installed vendored package**, not a regular in-repo module — it must stay pip-installable via its own `setup.py`, and `calibrate.py` imports from it as an external package (`from nbjw_calib...` style, not relative).
5. **`yolov8n_gsr.pt` is a fine-tuned custom model** with a specific 4-class label mapping (`player=0, goalkeeper=1, referee=2, ball=3`) baked into `detect.py` constants — this class mapping must stay consistent with the checkpoint; do not assume standard COCO classes.
6. **Output format follows the official SoccerNet GSR JSON schema** (`output.save_results_json`) and the GS-HOTA metric implementation follows the official formula (`evaluate.py`) — changes here affect compatibility with SoccerNet's official evaluation tooling and should be treated as a public contract.
**Path B is primary (decided 2026-10-12): Path A is removed in Phase 9, `docs/path_b/plans/IMPLEMENTATION_PLAN.md` §12.2. Don't extend Path A.** **Path B status (2026-10-11):** Rules 5 and 7 below describe **Path A**, which is still what `GSRPipeline` loads. Path B's detector (`gsr_pipeline/detect_athlete.py`, 2 classes: `athlete`, `ball`, via `gsr_pipeline/runtime/`) is parity-gated on ORT CPU and ORT GPU, but is **not wired into the pipeline**. It is reachable only through `evaluate_detection.py --path b` and the Kaggle runner `notebooks/path_b_eval.py`, which runs Path B batch end to end to GS-HOTA. Per-phase status: `docs/path_b/plans/IMPLEMENTATION_PLAN.md` §12.1. Latency work: `docs/path_b/plans/OPTIMIZATION_PLAN.md`. When it does land (plan Phase 2), Rule 7 stops applying to the live path: Path B's detector emits no roles at all, and `athlete_roles.py` recovers them downstream. Config for it lives in `configs/default.yaml`'s `detector_b` section, separate from `detector` so Path A stays runnable as the baseline until plan Phase 9.

7. **Role classification (player/goalkeeper/referee/ball) is owned by the detector, not inferred downstream.** `reid.OSNetReID._assign_roles` is a direct pass-through of `detect.py`'s `class_id`; `team.TeamClassifier` clusters only `role_detection == "player"` tracklets into 2 team-side clusters and never re-derives goalkeeper or referee role. Do not reintroduce spatial/appearance-based goalkeeper or referee heuristics downstream of the detector — if goalkeeper/referee detection needs improving, retrain/fix the detector, not `team.py`. The one exception is `athlete_roles.py`, which exists because the athlete/ball detector outputs no roles; it applies only if that detector replaces `yolov8n_gsr.pt`, and is not wired in today.

## Change Guidelines

- Read `gsr_pipeline/pipeline.py` first for any change touching more than one stage — it is the single place that shows the full data flow and config wiring.
- Prefer extending `configs/default.yaml` over hardcoding new constants, to stay consistent with the existing config-driven convention (except in the already-hardcoded training/utility scripts, where hardcoded constants are the established pattern — don't mix conventions across files without reason).
- When modifying a DataFrame-producing stage, check downstream consumers of any column you rename/remove (grep for the column name across `gsr_pipeline/`) since there is no schema enforcement.
- Match existing logging style (`log.info`/`log.warning` via module-level `logging.getLogger(__name__)`) in pipeline modules; `print()` is acceptable only in the existing standalone CLI/utility scripts.
- Since there is no test suite, manually verify pipeline changes by running `gsr_pipeline.run` against the committed sample sequence and, where relevant, `gsr_pipeline.evaluate` against its ground truth, before considering a change complete.
- `git status` currently shows `gsr_pipeline/track.py` (plus this session's edits) as locally modified but uncommitted — per `docs/architecture/PIPELINE_EVOLUTION.md` §6, this is a verified, intentional fix (`BYTETracker(args, frame_rate=...)` broke against `ultralytics==8.4.114`, patched with a `TypeError` fallback), not stray work. Still worth a `git status`/`git diff` check before starting new work so you know what's already staged versus what you're about to add.

## Forbidden / Dangerous Changes

Do not make these changes without explicit user instruction, given the effects noted:

- **Changing the GS-HOTA formula in `evaluate.py`** or the JSON schema in `output.save_results_json` — breaks compatibility with the official SoccerNet-GSR evaluation protocol referenced throughout `docs/architecture/GSR_PIPELINE.md` and README.
- **Changing `detect.py`'s class-ID mapping** (`player/goalkeeper/referee/ball`) without also retraining/re-mapping `yolov8n_gsr.pt` — the model and code must stay in sync; changing one without the other silently corrupts every downstream role/team decision.
- **Replacing or retraining `yolov8n_gsr.pt`** — this is a committed artifact others may depend on for reproducibility; treat as a deliberate, reviewed decision, not an incidental change.
- **Modifying `gsr_pipeline/nbjw_calib/`'s vendored model code** — it is a third-party calibration package pulled in as-is; upstream-diverging edits should be minimized and clearly flagged.
- **Removing/downgrading dependencies in `requirements.txt`** without checking all importers — several are optional/lazily-imported (`torchreid`, `easyocr`) with custom error handling; removing them silently changes failure behavior.
- **Committing files under `data/` or `runs/`** other than the Path B detector in `runs/yolo26s_athlete_ball/` and the ball specialist in `runs/ball_temporal/c5c/` — they are deliberately gitignored (large datasets, outputs and training artifacts).
- **Introducing network calls to unreviewed hosts** — the one existing network dependency (Zenodo checkpoint download in `calibrate.py`) is deliberate and documented; do not add silent new external downloads.

## Validation Requirements

Given the absence of CI, tests, or linters, a change should be considered complete only when:

1. The modified module(s) import and run without error (`python -c "import gsr_pipeline"` or targeted script execution).
2. `python -m gsr_pipeline.run --sequence_dir data/datasets/SoccerNetGS/gamestate-2024/valid/<seq> --output_dir <tmp> --max_frames <small N>` completes successfully if the change touches pipeline stages, detection, tracking, calibration, jersey, or team logic.
3. If the change affects output format or metrics, `python -m gsr_pipeline.evaluate` runs successfully against the produced `predictions.json` and a corresponding `Labels-GameState.json`.
4. Manually confirm no DataFrame column contract was broken — grep for any renamed/removed column across `gsr_pipeline/`.

There is no automated CI gate; the agent (or user) is the only verification step, so be conservative and explicit about what was and wasn't tested.

## Documentation Requirements

When a change alters any of the following, update the corresponding documentation in the same change:

- **Pipeline stage order, config schema, or module interfaces** → update this AGENTS.md's Architecture section and `README.md`'s "Configuration"/"Usage" sections.
- **Output JSON schema or GS-HOTA metric behavior** → update `README.md`'s "Evaluation Metrics" section and note the deviation in `docs/architecture/GSR_PIPELINE.md` if it affects reported results.
- **Installation/setup steps** (new dependencies, new sub-package install steps) → update `README.md`'s "Installation & Setup" section and `requirements.txt`.
- **New environment variables or config keys** → document them in this file's Configuration section and, if user-facing, in `README.md`.
- **CLI flags** → update the relevant script's `--help` (argparse) description and the corresponding README usage snippet.

## Uncertainties / Open Questions (flagged, not resolved)

- **`.gsr_venv/` naming**: this is the project's conventional venv directory name (used in README and gitignored), distinct from the top-level BallKnowledge monorepo's own `ballknowledge_venv/` — agents should be careful which venv is active when running commands in this subdirectory.
- **Python version**: no `.python-version`, `pyproject.toml`, or CI config pins a version — Unknown; infer from installed interpreter/venv if needed.
- **`data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021`**: the "Run Inference" example command in README/AGENTS.md references this path, but no dataset is tracked in this repo (see `.gitignore`) — the `valid/` split must be downloaded first (`docs/guides/BALL_TEMPORAL_GUIDE.md` §3) before that exact example command will work.
- **Batch-only architecture vs. the service's real-time goal**: tracklet aggregation, batch team classification, and refinement still require the full sequence. The live path sidesteps them rather than making them incremental: `process_stream` skips them, and `LiveTeamAssigner` replaces batch team classification. Live `POSSESSION`/`TURNOVER` remain unbuilt because `possession.py` still runs only over a completed sequence.
- **Relationship to the rest of the BallKnowledge monorepo**: resolved and built. cv-service is consumed over its HTTP + WS API by `CVPipelineSource`, which lives in `data_ingestion` (a separate git repository from this one, `github.com/SabihAli/ml-project`) and publishes through `data_ingestion`'s Kafka layer. A change to this repo's WS message shape (`api/event_serializer.py`) or detector classes must be matched in that file.
- **Ball detection is the weakest input to every ball-based event**: on `SNGS-021`, `evaluate_detection.py` measures 52.4% ball recall with gaps up to 6.7s (players 95.4%, referees 84.6%). See `docs/architecture/PIPELINE_EVOLUTION.md` §6 and §7.1 before building live possession or any classifier-tier event. On the 15 held-out test sequences the Path B detector finds 59.4% of balls; the temporal specialist `c5c` (`docs/architecture/BALL_TEMPORAL_ARCH.md`) raises that to 69.5% with precision up 1.2 points in simulation, but is not yet exported or wired into the pipeline.

### Resolved since initial documentation

The following were identified as bugs/stale docs and fixed in `gsr_pipeline/team.py`, `detect.py`, `reid.py`, `configs/default.yaml`, `README.md`, and `IMPLEMENTATION.md` (now `docs/architecture/GSR_PIPELINE.md`):
- `team.py` previously clustered "player"-only tracklets into 3 KMeans clusters and mislabeled the smallest as "referee" — real referees were already excluded from the clustering input via `role_detection`, so this cluster was actually a subset of real players being wrongly relabeled. Fixed to 2 clusters (team-side only); `configs/default.yaml`'s `team.n_clusters` changed from `3` to `2`.
- `team.py`'s "Final Goalkeeper Pass" (spatial x-position heuristic reassigning `role="goalkeeper"`) was removed — redundant and occasionally wrong (false positives on outfield players deep in their own box) now that the detector directly classifies goalkeepers.
- `detect.py` and `reid.py` module/function docstrings referencing stale COCO class ids (`0=person, 32=ball`) and a nonexistent aspect-ratio-based role heuristic were corrected to describe the actual 4-class fine-tuned detector and direct class_id → role pass-through.
- `IMPLEMENTATION.md`'s (now `GSR_PIPELINE.md`) Section 2.3 was updated to describe the current detector-based role classification; a note was added flagging that the reported GS-HOTA numbers in Section 4 predate this change and have not been re-measured against it.
- README's "Setup Calibration Plugin" step had a duplicated `cd ../..` line (would have `cd`-ed one directory above the repo root) — removed the extra line.
- README's "Run Inference" usage section claimed `--output_dir` saves `annotated_video.mp4`, but `run.py:70` actually writes `annotated.mp4` (and only when `--save_video`/`output.save_video` is set) — corrected.
- `team.py` previously left every goalkeeper tracklet with `team = None` indefinitely (only `role_detection == "player"` tracklets were ever assigned a side). Fixed: goalkeepers are still excluded from the KMeans *fit* (their kit usually doesn't resemble either outfield team's, which would have polluted the two team clusters), but each goalkeeper tracklet is now assigned a side via nearest-centroid distance against the already-fitted clusters, using the identical color+ReID feature construction (`TeamClassifier._combined_feature_matrix`). Referees remain unassigned (they don't belong to a team). This closes the gap flagged in `docs/architecture/REQUIREMENTS.md`'s FR-006/FR-005 sections re: goalkeeper team in possession/turnover payloads.

**Not a conflict, but note the difference:** `docs/architecture/PIPELINE_EVOLUTION.md` §6 reports a separately-verified run on the `SNGS-021` sample (75 frames, kickoff segment): DetA 50.48%, AssA 95.43%, GS-HOTA 69.10%. This is **not inconsistent** with `GSR_PIPELINE.md`'s reported Mean GS-HOTA of 30.72% — that figure is a mean across the full official SoccerNet-GSR test set, while the 69.10% figure is a single, explicitly-called-out "not representative of full-match difficulty" segment. Don't conflate the two numbers or treat one as superseding the other.
