# Ball improvement plan, items 1–3: Kaggle guide

How to run `docs/plans/BALL_IMPROVEMENT_PLAN.md` items 1–3 (miss diagnosis, ball tracker, the
c5d specialist) on a Kaggle notebook, end to end, from one generated notebook.

---

## 1. One-time setup

1. **Upload the c5c checkpoint.** Kaggle doesn't have it yet. Add `runs/ball_temporal/c5c/best.pt`
   (58 MB) to a dataset under the name **`c5c_best.pt`**, e.g. a new version of
   `sabihalii/ball-detector-scripts`. The runner finds it anywhere under `/kaggle/input`.
2. **Generate the notebook** from the repo (after any change to a kit file, regenerate):

   ```bash
   python scripts/kaggle/build_ball_notebook.py
   ```

   This writes `notebooks/kaggle_ball_improvement.ipynb`. The code travels inside it as
   `%%writefile` cells, so nothing else is uploaded. Its first cell lists the git commit and each
   kit file's sha256.
3. **Import it on Kaggle:** New Notebook → File → Import Notebook → the `.ipynb`.
4. **Notebook settings:** Accelerator **GPU T4 ×2**, Internet **on**.
5. **Add Input:** `sabihalii/soccernetgsr`, `sabihalii/ball-detector-scripts` (with `c5c_best.pt`).
   Optionally an earlier run's output, to reuse `main_cache.zip` or resume training.

---

## 2. Run order

| cell | stages | typical time | what to look at |
|---|---|---|---|
| Settings | – | – | set `SMOKE = True` for a ~15 min dry run first; check the paths |
| 0. Kit files | – | seconds | one `Writing ...` line per file |
| 1. Setup | `env,data` | 5–20 min | GPU list, package versions, free disk, copy progress (GB, MB/s, ETA), the split counts (`test` must be 15) |
| 2. Cache + motion | `cache,motion` | 30–60 min | `[cache-gpu0]` / `[cache-gpu1]` lines per sequence; the cache summary (test recall ≈ 59%) |
| 3. Diagnosis | `diagnose` | 10–20 min | per-sequence recall and top causes; the report with the live/batch ceiling |
| 4. Tracker | `track` | 1–2 h | one line per `select` trial (live, measured, batch+interp recall/precision); gate T |
| 5. c5d dataset | `c5d_data` | 30–60 min | `[i/N] SNGS-xxx: n samples`; the stats table |
| 6. c5d training | `c5d_train` | the rest of the session minus 2.5 h | per 100 steps: loss, lr, s/it, epoch ETA, GPU memory; per epoch: `select` score, `*best*`, run ETA |
| 7. Gates + export | `c5d_eval,export` | 1–2 h | gates 1–3 for c5d, gate S, ONNX parity PASS/FAIL |
| 8. Deliverables | `deliver` | minutes | `SUMMARY.md` rendered in the cell |
| 9. Cleanup | `cleanup` | 1 min | run before **Save Version** |

A heartbeat line every 60 s shows time in the stage, session time left, GPU use and free disk.
Everything printed is also in `/kaggle/working/ball_improvement/logs/` (`pipeline.log` and one
file per stage).

**Re-running a cell continues where it stopped.** Finished sequences, caches, the dataset build and
training epochs are all skipped or resumed.

---

## 3. Deliverables

`/kaggle/working/deliverables/` and `deliverables.zip`:

| path | contents |
|---|---|
| `SUMMARY.md` | headline recall/precision table, gates T / 1 / 3 / S, the diagnosis ceiling, next steps |
| `PERF_LOG_entry.md` | a dated entry to paste into `docs/path_b/PERF_LOG.md` |
| `reports/diagnosis/` | `report_diagnosis.md`, `misses.csv`, `displacement.csv`, `frames.jsonl`, `audit/` (sheets + `audit.csv` to fill) |
| `reports/track_c5c/`, `reports/track_c5d/` | tracker report, `results_track.json`, `search_log.csv` (every `select` trial) |
| `reports/eval_c5d/` | gates 1–3 report and `results.json` |
| `reports/parity_b*.json` | ONNX vs PyTorch parity |
| `configs/tracker_config.json` | the tracker and batch-path settings chosen on `select` |
| `models/` | c5d `best.pt`, ONNX + sidecar (batch 1; batch 3 too if the 3-window variant won), `history.csv`, `curves.png`, `args.json`, `DONE` or `INCOMPLETE` |
| `main_cache.zip` | reusable next time via `MAIN_CACHE_ZIP` |
| `logs/` | every stage's log |
| `MANIFEST.json` | bytes and sha256 of every file above |

If a stage fails, the deliverables are packaged automatically with what exists, and `SUMMARY.md`
lists which stage failed and why.

---

## 4. If training runs out of session time

Training stops on its own before the session budget (12 h minus `--reserve_hours`, 2.5 h by
default) and marks the run `INCOMPLETE`.

1. Run **Cleanup**, then **Save Version → Quick Save**.
2. New session: Add Input → this notebook's output. Run Settings and the kit cells.
3. Edit `PREV` in the **Resume** cell to `/kaggle/input/<output>/ball_improvement` and run it. It
   restores the state, caches, reports and `last.pt`, rebuilds the crop dataset (deterministic),
   and continues training.
4. Run cells 7–9.

---

## 5. Troubleshooting

| symptom | fix |
|---|---|
| `expected 15 test sequences` | the input's `valid/` is incomplete; fix the dataset |
| `c5c checkpoint not found` | upload `c5c_best.pt` (§1.1) or set `C5C_CKPT` |
| `NOT copying (not enough space)` | expected on a full dataset; stages read from `/kaggle/input` instead. `--copy always` forces it |
| CUDA out of memory in training | lower `BATCH` (per GPU) in Settings, e.g. 16 |
| `torch.distributed` errors | train on one GPU: `RUN("c5d_train", "--gpus", "1")` |
| a stage keeps failing | its log: `/kaggle/working/ball_improvement/logs/<stage>.log` |

---

## 6. What this does not do

- **Item 4** (main detector retrain) is not in this notebook. The plan gates it on the diagnosis;
  decide from `SUMMARY.md`.
- **Wiring into the pipeline.** The tracker (`gsr_pipeline/ball_tracker.py`) and c5d are measured
  here, not wired into `GSRPipeline`. That is a repo change, made after the gates pass.
