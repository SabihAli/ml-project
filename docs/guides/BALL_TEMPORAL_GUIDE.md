# Temporal ball specialist: local GPU guide (RTX 3090)

Step-by-step instructions for running the ball work in `docs/path_b/plans/BALL_TEMPORAL_PLAN.md` on a
single local NVIDIA GPU (written for an RTX 3090, 24 GB). An agent or a person can follow it top
to bottom. Every command is meant to be run from the **repo root** (the folder containing
`gsr_pipeline/`, `scripts/` and `docs/`) unless it says otherwise.

**What this guide covers today:** plan §9 steps 1–4: the main-detector cache (§4), the temporal
crop dataset (§5b), training (§5c) and the test gates (§5d). Steps 5–6 (export, tracker) are not
built yet; §6 says what to do when you reach that point.

**Rules for an agent following this guide:**
- Run the checks after each step. If a check fails, stop and report the output; don't work
  around it.
- Don't edit the scripts to make a step pass. Report the failure.
- Run long jobs in the foreground, so the progress output is visible.
- Record results only where §5 says.

---

## 0. Requirements

| item | needed |
|---|---|
| GPU | NVIDIA, CUDA-capable, ≥ 12 GB (RTX 3090 24 GB assumed) |
| NVIDIA driver | recent enough for CUDA 12.x (`nvidia-smi` shows "CUDA Version: 12.x" or later) |
| Python | 3.10, 3.11 or 3.12 |
| disk | room for the SoccerNet GSR `train` and `valid` zips **and** their extracted copies; check the zip sizes after downloading and keep at least that much free again |
| network | to download the dataset (once) and Python packages |
| SoccerNet access | none for the GSR splits (no password needed); a password is only needed for raw broadcast video, which this guide doesn't use |

Shell: the commands are bash (Linux, or Git Bash/WSL on Windows). Where Windows differs, both
forms are given.

---

## 1. Get the code and the weights

### 1.1 Clone

```bash
git clone https://github.com/SabihAli/ml-project.git cv_service
cd cv_service
```

### 1.2 Check the clone has this phase's files

```bash
ls scripts/ball_temporal/gsr_io.py scripts/ball_temporal/cache_main_detections.py docs/path_b/plans/BALL_TEMPORAL_PLAN.md docs/guides/BALL_TEMPORAL_GUIDE.md
```

All four must exist. **If any is missing**, the work hasn't been committed and pushed from the
development machine. Stop and ask for it to be pushed. Don't recreate the files.

### 1.3 Main detector weights

The scripts need the main Path B athlete/ball detector,
`runs/yolo26s_athlete_ball/best.pt` (about 20 MB). It may not be in git (only its
ONNX exports are tracked). Check:

```bash
ls -la runs/yolo26s_athlete_ball/best.pt
```

If it's missing, get it by **one** of these, then re-run the check:
- copy it from the development machine's `cv_service/runs/yolo26s_athlete_ball/best.pt`;
- or download the Kaggle dataset `sabihalii/ball-detector-scripts` (it contains `best.pt`):
  ```bash
  pip install kaggle   # needs ~/.kaggle/kaggle.json (Kaggle account → API token)
  kaggle datasets download sabihalii/ball-detector-scripts -p /tmp/bds --unzip
  mkdir -p runs/yolo26s_athlete_ball
  cp /tmp/bds/best.pt runs/yolo26s_athlete_ball/best.pt
  ```

Don't use the ONNX files in that folder instead; the scripts load the `.pt`.

---

## 2. Python environment

```bash
python -m venv .ball_venv
source .ball_venv/bin/activate          # Linux / WSL / Git Bash
# .ball_venv\Scripts\activate           # Windows PowerShell or cmd

python -m pip install --upgrade pip
# PyTorch with CUDA first, so Ultralytics doesn't pull a CPU-only build:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "ultralytics>=8.4.0" opencv-python numpy
```

`.ball_venv/` is gitignored.

**Checks:**

```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
python -c "from ultralytics import YOLO; m = YOLO('runs/yolo26s_athlete_ball/best.pt'); print(m.names)"
```

Expect:
- `cuda True` and the GPU's name;
- an Ultralytics version of 8.4 or later (older versions can't load YOLO26);
- class names that include `ball` (the detector has `athlete` and `ball`).

If `cuda` is `False`, the PyTorch build doesn't match the driver. Install the wheel for the CUDA
version `nvidia-smi` reports (e.g. `cu121`, `cu124`, `cu126`) and re-check.

---

## 3. Get the dataset

The scripts need the SoccerNet GSR `train` and `valid` splits (`gamestate-2024`), extracted into
their own folder, **`data/gsr/`**.

> Extract into `data/gsr/` as shown, not into `data/datasets/SoccerNetGS/`. That folder may hold
> a partial set of sequences from earlier work, and the scripts would pick those up instead of
> the full splits.

### 3.1 Download (once)

```bash
pip install SoccerNet
python - <<'EOF'
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory="data/gsr_download")
d.downloadDataTask(task="gamestate-2024", split=["train", "valid"])
EOF
find data/gsr_download -name "*.zip" -exec ls -la {} \;
```

The GSR splits need no password. If the downloader asks for one anyway, stop and ask the user;
don't guess. Expect a `train.zip` and a `valid.zip`.

### 3.2 Extract

```bash
mkdir -p data/gsr/train data/gsr/valid
unzip -q "$(find data/gsr_download -name train.zip | head -1)" -d data/gsr/train
unzip -q "$(find data/gsr_download -name valid.zip | head -1)" -d data/gsr/valid
```

On Windows without `unzip`, use PowerShell:
`Expand-Archive <path>\train.zip -DestinationPath data\gsr\train` (and the same for `valid`).

**Check:**

```bash
find data/gsr -name Labels-GameState.json | wc -l
```

Expect **115** (57 train + 58 valid). The script in §4.1 checks the split assignment too.

Once §4 is done, the zips under `data/gsr_download/` can be deleted to free space. `data/` is
gitignored.

---

## 4. Step 1: main-detector cache

What it does: runs the main detector over every frame of all 115 sequences (about 86k frames)
and saves, per sequence, every ball detection with confidence ≥ 0.05 (at most 5 per frame) next
to the ground-truth balls. The next step uses it to find **hard frames** (GSR has a ball, the
main detector missed it) and to place crops where the runtime would.

Script: `scripts/ball_temporal/cache_main_detections.py`, with `gsr_io.py` beside it.
Output: `data/outputs/ball_temporal/main_cache/<split>/SNGS-xxx.json` (gitignored).

### 4.1 Splits

```bash
python scripts/ball_temporal/cache_main_detections.py list --src data/gsr
```

Expect exactly:

| split | sequences | meaning |
|---|---|---|
| `train` | 57 | GSR train; the main detector trained on these, so easy examples only |
| `pool` | 33 | valid, not select or test; the source of hard examples |
| `select` | 10 | valid; model selection |
| `test` | 15 | `021 022 023 026 030 033 036 040 044 047 050 054 080 086 090` |

If the counts differ, stop and report the output.

### 4.2 Smoke test (one sequence, 32 frames)

```bash
python scripts/ball_temporal/cache_main_detections.py run --src data/gsr --out data/outputs/ball_temporal/main_cache_smoke --weights runs/yolo26s_athlete_ball/best.pt --seqs SNGS-021 --max_frames 32 --device 0 --half
python -c "import json,glob; d=json.load(open(glob.glob('data/outputs/ball_temporal/main_cache_smoke/*/SNGS-021.json')[0])); print(d['meta']); fr=d['frames']; print(len(fr), 'frames'); [print(k, v) for k, v in list(fr.items())[:5]]"
```

Checks:
- the run line reports `32 frames` and a **ms/frame** figure. Record it; it gives the time estimate;
- each frame has `balls` (lists of `cx, cy, w, h, conf`, highest confidence first) and `gt`
  (`cx, cy, size`);
- in frames where `gt` isn't empty, the first ball is usually within a few pixels of it.

Then delete the smoke output:

```bash
rm -rf data/outputs/ball_temporal/main_cache_smoke
```

### 4.3 Full run

```bash
python scripts/ball_temporal/cache_main_detections.py run --src data/gsr --out data/outputs/ball_temporal/main_cache --weights runs/yolo26s_athlete_ball/best.pt --device 0 --half --batch 32 --decode_threads 4
```

- Runs in the foreground and prints one line per sequence:
  `[gpu0] k/115 split/SNGS-xxx: N frames in Ts (X ms/frame), elapsed M min`.
- **Time estimate:** smoke-test ms/frame × 86,000. It hasn't been measured on a 3090 yet.
- **If it's interrupted**, run the same command again. Finished sequences print `cached` and are
  skipped; a sequence in progress is redone from its start.
- **If the GPU sits well under 80% busy** (check with `nvidia-smi` in another terminal), JPEG
  decoding is the bottleneck. Stop it and re-run with `--decode_threads 8`. Finished sequences
  are kept.
- If it runs out of GPU memory, use `--batch 16`.

**Check:**

```bash
for s in train pool select test; do echo "$s $(ls data/outputs/ball_temporal/main_cache/$s 2>/dev/null | grep -c '\.json$')"; done
```

Expect `train 57`, `pool 33`, `select 10`, `test 15`.

### 4.4 Summary

```bash
python scripts/ball_temporal/cache_main_detections.py summary --out data/outputs/ball_temporal/main_cache
```

It prints a table and saves it as `data/outputs/ball_temporal/main_cache/summary.md`. Per split:
frames, frames with a GT ball, the main detector's recall (top ball within 15 px at conf ≥ 0.15),
and hard frames.

**Sanity checks:**
- `test` recall should be close to **59%** (the main detector's ball frame recall on these 15
  sequences, measured on Kaggle 2026-10-12). If it's off by more than about 2 points, stop and
  report: something differs from that run (weights, image size, Ultralytics version).
- `train` recall should be clearly **higher** than `pool`, `select` and `test`. That's why hard
  examples come only from `pool` (plan decision A). If they're about the same, say so in the
  report; it would change that decision.
- The `pool` hard-frame count is the size of the hard-example set before oversampling.

---

## 5. Record the results

1. **`docs/path_b/PERF_LOG.md`:** add a dated entry at the top of the log, in the style of the
   existing entries:
   - hardware tag `rtx-3090` (add it to the log's tag list if it isn't there);
   - Ultralytics and PyTorch versions;
   - the smoke-test ms/frame, the full run's wall time and frames;
   - the summary table from §4.4.
2. **`docs/path_b/plans/BALL_TEMPORAL_PLAN.md`:** in §9, mark step 1 as done, with the date and a
   pointer to that `PERF_LOG.md` entry.
3. Report back the summary table, and any check that didn't match.

Don't commit `data/outputs/` (it's gitignored). Keep `data/outputs/ball_temporal/main_cache/`; step 2
reads it.

---

## 5b. Step 2: temporal crop dataset

What it does: for every ball-labelled frame it would sample, it cuts a 640×640 window from frames
t−2, t−1 and t, placed where the runtime would put it (on the main detector's last sighting in the
previous 15 frames), and saves the three crops side by side as one JPEG with a point label in
`manifest.csv`. Hard frames come from `pool` (plan §6). Runs on the CPU and needs step 1's cache.

Script: `scripts/ball_temporal/build_ball_temporal_dataset.py`. Output:
`data/outputs/ball_temporal/dataset/` (about 16 GB, gitignored).

Smoke test, then look at the preview sheets:

```bash
python scripts/ball_temporal/build_ball_temporal_dataset.py build --src data/gsr --out data/outputs/ball_temporal/dataset_smoke --seqs SNGS-024 SNGS-060 --workers 2
python scripts/ball_temporal/build_ball_temporal_dataset.py preview --out data/outputs/ball_temporal/dataset_smoke --n 4
```

In `dataset_smoke/preview/page_*.jpg`, green circles (frame t) must sit on the ball, and
negatives must contain no ball. Then delete `dataset_smoke` and run the full build (about 6 min
with 8 workers on the dev machine; re-running resumes per sequence):

```bash
python scripts/ball_temporal/build_ball_temporal_dataset.py build --src data/gsr --out data/outputs/ball_temporal/dataset --workers 8
```

It prints a table per split and saves it as `stats.md`. Expect about 46k samples and about 7.5k
`hard` samples in `pool`. Record the table in `PERF_LOG.md` as in §5.

---

## 5c. Step 3: train the heatmap specialist

Model: `scripts/ball_temporal/heatmap_model.py` (ResNet-18 encoder, U-Net decoder, stride-2 heatmap
+ offset, CenterNet focal loss; plan §5). Training: `scripts/ball_temporal/train_ball_temporal.py`,
on `train` + `pool` from step 2, validated on `select` each epoch. The first run downloads the
ImageNet ResNet-18 weights (about 45 MB) from download.pytorch.org.

Smoke test (a few minutes; checks the whole loop and writes a checkpoint):

```bash
python scripts/ball_temporal/train_ball_temporal.py --out runs/ball_temporal/smoke --channels 9 --epochs 1 --limit 6400
```

Then train both input versions, one after the other:

```bash
python scripts/ball_temporal/train_ball_temporal.py --out runs/ball_temporal/c9b --channels 9
python scripts/ball_temporal/train_ball_temporal.py --out runs/ball_temporal/c5b --channels 5
```

- One epoch is about 48.8k samples (hard `pool` positives counted 3 times). On the RTX 3090 it
  takes about 19 min (0.7 s/step at batch 32, GPU-bound), so 30 epochs is at most about 9.5 h per
  run; early stopping (patience 8) usually ends it sooner.
- Each epoch prints the `select` **score**: distance recall (peak within 15 px) averaged over the
  thresholds at which 1%…10% of negative crops (`neg_hard`, `neg_random`) fire, overall, hard
  and easy. `outside` crops are left out of training and of that count (they often contain a
  real, unlabelled ball); `outside fire` is printed for reference. `best.pt` is the epoch
  with the best score. Recall at a single 5% threshold and at 0.5 are printed too, for reference
  only: the single-threshold recall swung 16–41% between epochs on noise and stopped the first
  runs early (2026-10-13). `--resume` on a run from before that change re-scores its `best.pt`
  and `last.pt` first.
- Outputs per run: `best.pt`, `last.pt`, `history.csv`, `curves.png`. `runs/` is gitignored.
- **If it's interrupted**, add `--resume` to the same command.
- **If it runs out of GPU memory**, use `--batch 16 --lr 2.5e-4`. Batch 32 uses about 16 GB.
- Don't use `channels_last`: it made this network about 4× slower on the 3090.

Record each finished run (best epoch, select recall overall/hard/easy, wall time) in
`PERF_LOG.md` as in §5.

---

## 5d. Step 4: test gates 1–3

Script: `scripts/eval/ball_det/eval_ball_temporal.py`. It scores a checkpoint on the step 2 test
crops (gates 1–2) and replays every frame of the `select` and `test` sequences as the runtime runs
it (gate 3): the specialist runs only when the main detector misses, on a window centred on the
last sighting, for up to 15 frames. The system threshold is chosen on `select` and frozen for
`test`. A few minutes per checkpoint on the RTX 3090.

```bash
python scripts/eval/ball_det/eval_ball_temporal.py all --ckpt runs/ball_temporal/c5b/best.pt --src data/gsr --out data/outputs/ball_temporal/eval/c5b
```

- Output: `report.md` (gate tables, recall by frames since the last sighting, the `select`
  threshold sweep), `results.json`, `test_crops.csv`.
- `--chain main` centres the window on main-detector sightings only (as the training crops were
  built); the default `fused` re-centres on specialist hits, as the runtime does.
- `--test_sweep` adds the test curve at every threshold. It is a diagnostic: never choose the
  threshold from it.

Record the gate tables in `PERF_LOG.md` as in §5.

---

## 6. Step status and next steps

Plan §9, steps 2–6 (steps 2–4 have sections above; 5–6 are not built):

| step | what | status |
|---|---|---|
| 2 | `build_ball_temporal_dataset.py`: 3-frame crops, point labels, hard examples from `pool` | **built**, §5b; done 2026-10-13 |
| 3 | heatmap model and training script (single 3090) | **built**, §5c; done 2026-10-15 (`c9b`, `c5b`) |
| 4 | evaluation: crop-level gates and the full-system gate | **built**, §5d; measured 2026-10-15; `select`-only threshold rule pending (plan §10 item 14) |
| 5 | ONNX export, parity, TensorRT timing (gate 4: ≤ 5 ms on an RTX 30-series) | not built |
| 6 | `BallTracker` | not built; after step 5 (the specialist is measured, §5d) |

When you reach this point, stop. Each step is added to this guide as it's built.

**Single-GPU notes for steps 3–5** (the plan was written for Kaggle's 2×T4; on one 3090):
- train on one GPU (`--device 0`); a 24 GB card fits a larger batch than one T4, so the per-step
  batch can double, with the learning rate scaled to match;
- run evaluation passes one at a time, with no sharding;
- the 3090 is the RTX 30-series target, so step 5's latency can be measured directly against
  gate 4.

---

## 7. Troubleshooting

| symptom | cause and fix |
|---|---|
| `model has no 'ball' class` | the wrong weights file; use `runs/yolo26s_athlete_ball/best.pt` |
| `KeyError`/`AttributeError` loading the `.pt`, or "unknown model" | Ultralytics older than 8.4; `pip install -U "ultralytics>=8.4.0"` |
| `torch.cuda.is_available()` is `False` | the PyTorch build doesn't match the CUDA driver; §2 |
| `list` shows fewer than 15 `test` | the `valid` split is incomplete; re-extract `valid.zip` (§3.2) |
| `list` shows only a few sequences | `--src` points at a partial local set (for example `data/datasets/SoccerNetGS/`); use `--src data/gsr` |
| warning that `half` is deprecated | harmless; Ultralytics 8.4 warns about it |
| a sequence fails with `cannot decode frame` | a corrupt JPEG in the extracted data; re-extract that split |
