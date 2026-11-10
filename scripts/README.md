# scripts/

Standalone tools, grouped by the plan or task they serve. None of them is imported by
`gsr_pipeline/` at runtime. Run every script from the `cv_service/` root
(`python scripts/<group>/<file>.py ...`).

| folder | plan / task | contents |
|---|---|---|
| `path_b/detector/` | Path B detector (`docs/path_b/plans/IMPLEMENTATION_PLAN.md`, `OPTIMIZATION_PLAN.md`) | `export_athlete_detector.py`, `build_trt_engine.py`, `parity_gate.py`, `train_yolo26_gsr.py`, `infer_athlete_video.py` |
| `path_b/calibration/` | NBJW calibration export and parity | `export_nbjw.py`, `parity_pipeline.py` |
| `ball_specialist_v1/` | v1 YOLO crop specialist (superseded by `ball_temporal/`, kept as a record) | label stats, crop dataset, training, evaluation, label snapping |
| `ball_temporal/` | Ball specialist v2 (`docs/path_b/plans/BALL_TEMPORAL_PLAN.md`) | main-detector cache, camera-motion cache, 3-frame dataset, heatmap model, training, ONNX export, `gsr_io.py` (splits) |
| `ball_eval/` | Ball system evaluation, shared by every ball plan | `eval_ball_temporal.py` (gates 1-3; `simulate` mirrors `BallFuser.update`), `ball_system.py` (tracker simulation), `diagnose_ball_misses.py` |
| `ball_improvement/` | `docs/plans/BALL_IMPROVEMENT_PLAN.md` | `build_ball_notebook.py` -> `notebooks/kaggle_ball_improvement.ipynb` |
| `pipeline_improvement_1/phase0/` | `docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md` Phase 0 | `phase0_report.py` (miss buckets B0-B7, ceilings, stage/size metrics, split integrity), `build_phase0_notebook.py` -> `notebooks/pipeline_improvement_1/phase0_baseline.ipynb` |
| `pipeline_improvement_1/phase1/` | same plan, Phase 1 tasks 1-2 | `phase1_sweep.py` (detector-ball plausibility gate `gsr_pipeline/ball_gate.py` + c5c threshold, chosen on `select`, frozen for `test`), `build_phase1_notebook.py` -> `notebooks/pipeline_improvement_1/phase1_gate_threshold.ipynb` |

New work for a plan phase goes in `<plan>/<phase>/`. Kaggle notebooks are generated from the
repo files by a `build_*_notebook.py` script (code goes in as `%%writefile` cells), so edit the
repo and regenerate rather than editing a notebook by hand. Uploads for a notebook are zipped
into `data/notebook_uploads/` (gitignored).
