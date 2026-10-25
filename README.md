# cv_service: SoccerNet Game State Reconstruction

The computer-vision service of BallKnowledge. It turns broadcast football video into game state:
every player, goalkeeper, referee and the ball, with track IDs, pitch positions in metres, roles,
teams and jersey numbers. It runs as a batch pipeline over recorded clips, or live behind an
HTTP + WebSocket API that `data_ingestion` consumes.

The pipeline (`gsr_pipeline`) is a lightweight reimplementation of the SoccerNet GSR baseline:
YOLO detection, ByteTrack, NBJW pitch calibration, OSNet appearance, EasyOCR jerseys, and the
official GS-HOTA metric. How it works and how well: `docs/architecture/GSR_PIPELINE.md`.

The pipeline is **Path B** (since 2026-10-25, batch and live):
- a 2-class YOLO26s athlete/ball detector (`runs/yolo26s_athlete_ball/`);
- the temporal ball specialist **c5c** (`runs/ball_temporal/c5c/`) on frames where the detector
  missed the ball: on the 15 held-out test sequences, ball recall 59.4% → 69.5%, precision
  85.3% → 86.5%;
- roles and teams from kit colour (`gsr_pipeline/live_roles.py`).

Status per phase: `docs/path_b/plans/IMPLEMENTATION_PLAN.md` §12.1. The old 4-class YOLOv8n
pipeline (Path A) is in `archive/path_a/` and is not used.

Agent-facing repo guidance: `AGENTS.md`.

---

## Setup

```bash
python -m venv .gsr_venv
source .gsr_venv/bin/activate            # Linux/macOS
# .gsr_venv\Scripts\activate             # Windows

# On a CUDA machine, install PyTorch for your CUDA version first (https://pytorch.org).
pip install -r requirements.txt
pip install -e gsr_pipeline/nbjw_calib   # vendored pitch-calibration package (required)
```

Calibration checkpoints download from Zenodo on first run into `data/pretrained_models/calibration/`.
GPU hosts that run Path B on TensorRT install `onnxruntime-gpu` and `tensorrt` in place of
`onnxruntime` (see `requirements.txt`).

Data: SoccerNet GSR (`gamestate-2024`). No dataset is in git; see
`docs/guides/BALL_TEMPORAL_GUIDE.md` §3 for downloading the `train` and `valid` splits. The examples below assume
they are under `data/datasets/SoccerNetGS/gamestate-2024/`.

---

## Usage

### Batch: run a sequence, then score it

```bash
python -m gsr_pipeline.run --sequence_dir data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021 --output_dir data/outputs/SNGS-021 --max_frames 50
python -m gsr_pipeline.evaluate --gt_path data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021/Labels-GameState.json --pred_path data/outputs/SNGS-021/predictions.json
```

`run.py` flags: `--config` (default `gsr_pipeline/configs/default.yaml`), `--save_video` (writes
`annotated.mp4`), `--no-refine`, `--device cpu|cuda`. `evaluate.py` reports DetA, AssA and
GS-HOTA.

### Detector alone, per class

```bash
python -m gsr_pipeline.evaluate_detection --sequence_dir data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021 --output_json data/outputs/SNGS-021_detection_eval.json
```

Recall and precision per class (people on IoU, the ball on centre distance), the role confusion
table, and ball frame recall with the longest miss. Add `--path b --backend ort_cpu --device cpu`
for the Path B detector, and `--start_frame 168` to skip SNGS-021's kickoff.

### Path B tools

```bash
python scripts/export_athlete_detector.py          # ONNX export (ultralytics>=8.4.0)
python scripts/build_trt_engine.py                 # TensorRT engine ahead of time (optional: the pipeline
                                                   # builds a missing engine itself on first launch)
python scripts/parity_gate.py --frames_per_sequence 30 --backend ort_cpu --device cpu
python notebooks/path_b_eval.py --data_root <valid dir> --output_dir <out> --limit 3 --max_frames 100   # CUDA host
```

### Live service (HTTP + WebSocket)

```bash
CV_SERVICE_API_KEY=some-secret python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`POST /streams`, `POST /streams/{id}/stop`, `GET /health`, and a WS event feed at
`/streams/{id}/events`. The key is required: the service refuses every request without it. End
to end checks, with the server running: `python -m api.validate_api ...` and
`python -m api.validate_ws ...` (see `AGENTS.md`). Design:
`docs/architecture/STREAMING_ARCHITECTURE.md`.

### Possession and turnovers (batch, from predictions.json)

```bash
python -m gsr_pipeline.possession_demo --json_path data/outputs/SNGS-021/predictions.json --output_path data/outputs/SNGS-021/possession_events.json
```

### Ball specialist work

Step-by-step: `docs/guides/BALL_TEMPORAL_GUIDE.md` (local GPU) or
`docs/guides/BALL_TEMPORAL_KAGGLE_GUIDE.md` (Kaggle). Plan: `docs/path_b/plans/BALL_TEMPORAL_PLAN.md`.
The current model (`c5c`, a 3-frame heatmap specialist), its results and history:
`docs/architecture/BALL_TEMPORAL_ARCH.md`.

### Tests

```bash
python -m pytest tests/ -q
```

Pure-logic unit tests for Path B's runtime (no weights, GPU or data). They don't cover the
whole pipeline.

---

## Configuration

All runtime settings live in `gsr_pipeline/configs/default.yaml`, one section per stage
(`detector`, `ball_specialist`, `reid`, `tracker`, `calibration`, `jersey`, `refiner`,
`output`, `possession`, `live_roles`, `athlete_roles`, `reid_worker_pool`,
`live_broadcast`, `frame_source`, `ws`, `streaming`).
The inline comments explain each tunable and where its value came from.

---

## Documentation

| folder | doc | authoritative for |
|---|---|---|
| `docs/architecture/` | `GSR_PIPELINE.md` | how the pipeline works, the metric, results |
| | `PIPELINE_EVOLUTION.md` | event taxonomy, revision semantics, Kafka mapping, open design issues |
| | `STREAMING_ARCHITECTURE.md` | the live path and the API; build status in §8 |
| | `BALL_TEMPORAL_ARCH.md` | the temporal ball specialist: model, training, results |
| | `REQUIREMENTS.md` | requirements, with evidence levels |
| `docs/path_b/plans/` | `IMPLEMENTATION_PLAN.md` | the Path B pipeline; per-phase status in §12.1 |
| | `OPTIMIZATION_PLAN.md` | Path B latency work; build status in §7 |
| | `BALL_TEMPORAL_PLAN.md` | the ball specialist; work order in §9 |
| `docs/path_b/` | `DETECTOR_AND_DATA_BLOCKERS.md` | why Path B; ball-detection options; the role/team classifier |
| | `PERF_LOG.md` | dated performance and accuracy measurements |
| `docs/guides/` | `PATH_B_EVAL_GUIDE.md` | scoring the Path B pipeline end to end with GS-HOTA, jersey counted (local GPU) |
| | `BALL_TEMPORAL_GUIDE.md`, `BALL_TEMPORAL_KAGGLE_GUIDE.md` | the ball specialist work, local GPU and Kaggle |
| `docs/results/` | `PATH_B_EVAL_T4_2026-10-05.md` | Path B end to end on a Kaggle T4 |

Conventions: each plan has a status line and a build-status section, kept current as work lands.
Measurements go into `PERF_LOG.md` as dated entries. Where a doc and the code disagree, the code
wins and the doc gets fixed.

---

## Layout

```text
gsr_pipeline/          the pipeline: stages, live path, Path B runtime (runtime/), evaluators
  configs/default.yaml   all runtime settings
  nbjw_calib/            vendored calibration package (own setup.py)
api/                   FastAPI HTTP + WS service
scripts/               export, TensorRT, parity, training and inference tools
  ball_temporal/         ball specialist scripts (current): cache, dataset, model, training
  eval/ball_det/         ball specialist evaluation (test gates)
notebooks/path_b_eval.py   Path B end-to-end evaluation runner
tests/                 pytest, pure logic
docs/                  see Documentation
runs/yolo26s_athlete_ball/   Path B detector: best.pt, ONNX exports and sidecars (tracked)
runs/ball_temporal/c5c/      ball specialist: best.pt, ONNX + sidecar, training args and curves (tracked)
data/                  gitignored: datasets/SoccerNetGS/, outputs/, pretrained_models/
archive/path_a/        the retired Path A pipeline (4-class YOLOv8n), not imported
```

## License

Follows the licensing of the SoccerNet GSR baseline and the models it uses (YOLO, TorchReID).
