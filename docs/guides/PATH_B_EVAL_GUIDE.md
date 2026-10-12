# Evaluating the Path B pipeline end to end, with jersey numbers (local GPU)

Step-by-step instructions for scoring the current Path B pipeline with GS-HOTA, **jersey
numbers counted**. An agent or a person can follow it top to bottom. Run every command from the
**repo root** (the folder containing `gsr_pipeline/`, `notebooks/` and `docs/`).

**What gets scored.** `notebooks/path_b_eval.py` runs, per sequence:

```
athlete/ball detector (YOLO26s, ONNX Runtime GPU)
  -> ByteTrack on athletes
  -> LiveCalibrator (NBJW every 5 frames, camera-motion propagation between) -> pitch positions
  -> EasyOCR jersey numbers on every 5th frame
  -> AthleteRoleTeamClassifier (fit on the first 250 frames), then a per-track vote
     for role, team and jersey
  -> predictions.json -> GS-HOTA (gsr_pipeline/evaluate.py, unchanged)
```

**Read the numbers with these limits in mind:**
- **The jersey path has never been run.** Every recorded Path B score so far has jersey ignored
  (`--jersey` was off). §5's smoke test exists to catch failures in it. If it fails, stop and
  report; don't patch around it.
- **GS-HOTA with jersey is not comparable to the earlier 58.95%**, which ignored jersey. With
  jersey counted, a player only scores when the OCR number matches the ground truth, so the
  score is expected to drop clearly. That is not a bug.
- **EasyOCR here is the old recognizer** (`gsr_pipeline/jersey.py`), not the Phase 8 design in
  `docs/path_b/plans/IMPLEMENTATION_PLAN.md`.
- **No tracklet refinement** (it needs ReID, which Path B drops).
- **The ball** is the single highest-confidence detection per frame under one constant track
  id. No ball tracker exists yet.
- **Role-classifier tunables were fitted on SNGS-021 only**; every other sequence is a
  held-out test of them.

**Rules for an agent following this guide:**
- Run the check after each step. If a check fails, stop and report the output.
- Don't edit the scripts or config to make a step pass.
- Run long jobs in the foreground so progress is visible.
- Record results only where §8 says.

---

## 1. Requirements

| item | needed |
|---|---|
| GPU | NVIDIA, CUDA 12.x driver, ≥ 12 GB (written for an RTX 3090 24 GB) |
| cuDNN | 9.x, on the library path, for ONNX Runtime GPU (a `pip install nvidia-cudnn-cu12` provides it if the system has none) |
| Python | 3.10, 3.11 or 3.12 |
| disk | the extracted `valid` split (about the size of its zip), plus about 1 GB for outputs |
| network | for packages, the NBJW calibration checkpoints (Zenodo, downloaded on first run) and EasyOCR's models (downloaded on first use) |

---

## 2. Code

```bash
git clone https://github.com/SabihAli/ml-project.git cv_service
cd cv_service
ls notebooks/path_b_eval.py runs/yolo26s_athlete_ball/best_736x1280.onnx runs/yolo26s_athlete_ball/best_736x1280.onnx.json gsr_pipeline/configs/default.yaml docs/guides/PATH_B_EVAL_GUIDE.md
```

All five must exist. If any is missing, the work hasn't been pushed from the development
machine; stop and ask. (If you already cloned for `BALL_TEMPORAL_GUIDE.md`, reuse that clone.)

The detector runs from the tracked ONNX file, so `best.pt` is **not** needed, except for the
optional parity and reference checks in §7.

---

## 3. Environment

```bash
python -m venv .gsr_venv
source .gsr_venv/bin/activate          # Linux / WSL / Git Bash
# .gsr_venv\Scripts\activate           # Windows PowerShell or cmd

python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
pip install -e gsr_pipeline/nbjw_calib
```

The `onnxruntime` / `onnxruntime-gpu` swap matters: with both installed, ONNX Runtime can pick
the CPU build, and the detector then falls back to the CPU tier without an error.

**Checks:**

```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -c "import onnxruntime as o; print('onnxruntime', o.__version__, o.get_available_providers())"
python -c "import easyocr, yacs, sklearn, scipy; print('easyocr', easyocr.__version__)"
python -c "import nbjw_calib; print('nbjw_calib ok')"
```

Expect:
- `cuda True` and the GPU's name;
- `CUDAExecutionProvider` in the ONNX Runtime provider list (if it's missing, fix it before
  going on: the CPU tier is far too slow for this evaluation);
- no import errors.

---

## 4. Data

The runner needs the SoccerNet GSR **`valid`** split (58 sequences). If you haven't downloaded
and extracted it, do `docs/guides/BALL_TEMPORAL_GUIDE.md` §3 first (it extracts to
`data/gsr/valid/`). Don't use `data/datasets/SoccerNetGS/`; it may hold only a few sequences from earlier work.

`--data_root` must be the folder whose **immediate children are the `SNGS-xxx` sequence
folders**. Find it:

```bash
python - <<'EOF'
from pathlib import Path
labels = list(Path("data/gsr/valid").rglob("Labels-GameState.json"))
roots = sorted({str(p.parent.parent) for p in labels})
print(len(labels), "sequences; data_root candidates:", roots)
EOF
```

Expect `58 sequences` and exactly **one** candidate (usually `data/gsr/valid`). In the commands
below, `data/gsr/valid` stands for that value; replace it if yours differs. If there are
several candidates or the count isn't 58, stop and report.

---

## 5. Smoke test (1 sequence, 50 frames, jersey on)

```bash
python notebooks/path_b_eval.py --data_root data/gsr/valid --output_dir data/outputs/path_b_eval_smoke --limit 1 --max_frames 50 --stages detection gshota --jersey --jersey_every 5 --profile_frames 0
```

Do **not** pass `--calib_backend` or `--backend`; the defaults are right.

**Checks (all must pass):**
1. The output includes the line `detector backend: ort_gpu`. If it says `ort_cpu`, ONNX Runtime
   isn't using the GPU; go back to §3.
2. No line contains `FAILED`. If one does, stop and report that line and the traceback in
   `data/outputs/path_b_eval_smoke/SNGS-*/result.json` (`error`).
3. The per-sequence line shows `GS-HOTA persons ...` and an `ms/frame` figure. **Note the
   ms/frame**; it sets the time estimate for §7.
4. The summary says `Jersey in IdSim: **used**.`
5. OCR produced numbers:

   ```bash
   python - <<'EOF'
   import json, glob
   p = glob.glob("data/outputs/path_b_eval_smoke/SNGS-*/predictions.json")[0]
   dets = [d for f in json.load(open(p))["predictions"] for d in f["detections"]]
   people = [d for d in dets if d["role"] in ("player", "goalkeeper")]
   with_num = [d for d in people if d["jersey_number"]]
   print(f"{len(with_num)} of {len(people)} player/goalkeeper detections have a jersey number")
   print("examples:", sorted({d["jersey_number"] for d in with_num})[:15])
   EOF
   ```

   Expect a non-zero count and plausible shirt numbers (1–99). **Zero means the jersey path
   isn't working; stop and report.**

Then remove the smoke output:

```bash
rm -rf data/outputs/path_b_eval_smoke
```

---

## 6. Run A: the earlier 3 sequences, to check this machine against the T4 results

The T4 report of 2026-10-05 used `--limit 3 --max_frames 100`: the first 3 sequences in sorted
order, 100 frames each. It doesn't list their names; its last line is SNGS-023, which fits
SNGS-021 to 023. This run repeats that **without jersey**, so the result should land close to
its numbers:

```bash
python notebooks/path_b_eval.py --data_root data/gsr/valid --output_dir data/outputs/path_b_eval_repro --limit 3 --max_frames 100 --stages detection roles gshota
```

**Check:** `GS-HOTA persons` for the mean should be close to **58.95%** and `with_ball` close
to **57.75%** (T4 run: `docs/results/EVAL_2.md`). Within about 1 point is fine; the detector is
deterministic but the GPU and library versions differ. If it's more than 3 points away, stop
and report; something differs from that run.

Then the same 3 sequences **with jersey**:

```bash
python notebooks/path_b_eval.py --data_root data/gsr/valid --output_dir data/outputs/path_b_eval_jersey3 --limit 3 --max_frames 100 --stages detection roles gshota --jersey --jersey_every 5
```

This is the first jersey-counted number on the same frames. Expect it **below** Run A's.

---

## 7. Run B: the whole `valid` split, jersey counted

All 58 sequences, every frame (about 43,500 frames):

```bash
python notebooks/path_b_eval.py --data_root data/gsr/valid --output_dir data/outputs/path_b_eval_full --stages detection roles gshota --jersey --jersey_every 5
```

- Runs in the foreground; prints one block per sequence and a final summary.
- **Time:** the smoke test's ms/frame × 43,500. EasyOCR on every 5th frame makes this slower
  than the T4 figure of 85.9 ms/frame; nothing measures how much slower yet. Plan for hours,
  not minutes.
- **If it's interrupted**, run the same command again. Each finished sequence is cached in
  `data/outputs/path_b_eval_full/<SNGS-xxx>/result.json` and is skipped; use `--force` only to
  recompute everything.
- A sequence that fails is recorded in the summary's `failed` list and the run continues. Report
  any failures.

Optional extras (each adds time; run only when asked):
- `--stages parity detection roles gshota --parity_backends ort_gpu`: the detector parity gate
  against the Ultralytics `.pt`. Needs `runs/yolo26s_athlete_ball/best.pt`.
- `--reference`: also scores the Ultralytics `.pt` detector (same `best.pt` requirement).

---

## 8. Read and record the results

Each run writes `summary.md` and `summary.json` in its output folder, plus
`<SNGS-xxx>/predictions.json` and `result.json` per sequence.

**Read the summary's:**
- **GS-HOTA table** (persons / with_ball), and the line `Jersey in IdSim: **used**`;
- **Detection** (athlete and ball recall and precision, ball recall by size);
- **Role/team** accuracy, on ground-truth boxes;
- **Latency** per stage, and the NBJW/propagation split.

**Record:**
1. Copy each summary into the repo, adding the run's date to each filename:

   ```bash
   cp data/outputs/path_b_eval_full/summary.md docs/results/PATH_B_EVAL_RTX3090_valid_jersey_<date>.md
   cp data/outputs/path_b_eval_jersey3/summary.md docs/results/PATH_B_EVAL_RTX3090_3seq_jersey_<date>.md
   cp data/outputs/path_b_eval_repro/summary.md docs/results/PATH_B_EVAL_RTX3090_3seq_nojersey_<date>.md
   ```

2. **`docs/path_b/PERF_LOG.md`:** a dated entry with the hardware tag `rtx-3090`, the library
   versions (`torch`, `onnxruntime-gpu`), the three GS-HOTA tables, the ms/frame, and which
   sequences failed, if any. Say plainly that these are the first jersey-counted numbers.
3. **`docs/architecture/GSR_PIPELINE.md` §5.2:** add the full-`valid` jersey-counted result.
4. Report back: the three GS-HOTA tables, the jersey statistics from the §5 check, and any
   check that didn't match.

Don't commit `data/outputs/` (gitignored).

---

## 9. Troubleshooting

| symptom | cause and fix |
|---|---|
| `detector backend: ort_cpu` | ONNX Runtime has no GPU provider; §3 (uninstall `onnxruntime`, install `onnxruntime-gpu`, check cuDNN 9) |
| `libcudnn.so.9: cannot open shared object file` | cuDNN 9 isn't on the library path; `pip install nvidia-cudnn-cu12` and add its `lib` folder to `LD_LIBRARY_PATH` |
| `no sequences with Labels-GameState.json under ...` | wrong `--data_root`; §4 |
| NBJW checkpoint download fails | no network to Zenodo; the files go to `data/pretrained_models/calibration/` (`SV_kp`, `SV_lines`); download them manually from the URLs in `gsr_pipeline/calibrate.py` |
| EasyOCR hangs on first use | it is downloading its models (`~/.EasyOCR/model/`); wait, or check the network |
| `CUDA out of memory` in the OCR step | EasyOCR and the detector share the GPU; close other GPU jobs |
| zero jersey numbers in the §5 check | the jersey path isn't producing output; stop and report |
| GS-HOTA with jersey is far below the no-jersey number | expected: IdSim requires the number to match the ground truth (see the limits at the top) |
