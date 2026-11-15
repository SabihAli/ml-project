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
| `pipeline_improvement_1/phase3/` | same plan, Phase 3 (specialist) | `convert_sntracking.py` (SoccerNet-Tracking test = GSR test clips -> GSR layout, symlinked frames), `phase3a_mine.py` (Phase 1 replay with c5c heatmap peaks on every split; mined hard positives/negatives from train + pool; multi-peak selection on select/test), `build_phase3a_notebook.py` -> `notebooks/pipeline_improvement_1/phase3a_mining.ipynb`; `profile_latency.py` (live-path latency profile on the target GPU: e2e + per stage + candidate ONNX models vs the §1.4 budget) |
| `pipeline_improvement_1/phase3b/` | same plan, Phase 3b (offline teacher) | `train_teacher_yolo.py` (`build`: hard-link mirror + ball-only YOLO labels; `train`: YOLO26 at 1920, init from the live detector with its `ball` head row; `eval`: one ball per frame @15 px, threshold on `select`, frozen on `test`), `train_teacher_yolo_motion.py` (same, 5 channels: BGR(t) + grey(t-1) + grey(t+1), stacked at read time; `dump`: top-k candidates per frame), `trajectory_pass.py` (offline temporal support + gap fill over the candidates, CPU; knobs on `select`), `train_teacher_heatmap.py` (T1: 5-frame t-2..t+2 heatmap net, ResNet encoder + U-Net decoder, stride-2 heatmap + offset; trains on native-scale crops, infers full frames; `dump` writes candidates in the same format), `fuse_candidates.py` (merges two teachers' candidate dumps: matched within 15 px, weighted score; one folder per weight, chosen on `select` via `trajectory_pass.py`); `train_teacher_yolo_motion.py --dts 1,2` gives 7 channels (grey t-2..t+2) |
| `pipeline_improvement_1/phase4/` | same plan, Phase 4 (robust ball tracker) | `phase4_sweep.py` (`gsr_pipeline/ball_tracker_robust.py` driving main + c5c: knob search on `select`, frozen `test` vs Phase 1, ablations, recall by condition), `build_phase4_notebook.py` -> `notebooks/pipeline_improvement_1/phase4_tracker.ipynb`; `sort_compare.py` (boxmot ByteTrack / OC-SORT / BoT-SORT driving main + c5c vs Phase 1), `build_sort_notebook.py` -> `notebooks/pipeline_improvement_1/phase4b_sort.ipynb` |

New work for a plan phase goes in `<plan>/<phase>/`. Kaggle notebooks are generated from the
repo files by a `build_*_notebook.py` script (code goes in as `%%writefile` cells), so edit the
repo and regenerate rather than editing a notebook by hand. Uploads for a notebook are zipped
into `data/notebook_uploads/` (gitignored).
