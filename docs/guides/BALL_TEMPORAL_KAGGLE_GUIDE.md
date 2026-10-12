# Temporal ball specialist: Kaggle guide

Notebook steps for `docs/path_b/plans/BALL_TEMPORAL_PLAN.md`. For a local GPU, use
`docs/guides/BALL_TEMPORAL_GUIDE.md` instead. This guide covers **step 1, the
main-detector cache** (plan §6, §9). Later steps get their own sections as they're built.

Paste the cells top to bottom. Every path is written out; there are no variables to set.

---

## 0. Before you start

**Notebook settings:** GPU T4 ×2, internet on (for `pip`).

**Inputs** (Add Input):

| input | path | used for |
|---|---|---|
| `sabihalii/soccernetgsr` | `/kaggle/input/datasets/sabihalii/soccernetgsr` | GSR frames and labels (`train/`, `valid/`) |
| `sabihalii/ball-detector-scripts` | `/kaggle/input/datasets/sabihalii/ball-detector-scripts` | only its `best.pt`, the main athlete/ball detector |

**Scripts, created in the notebook with `%%writefile`** (no dataset upload), from the repo's
`cv_service/scripts/ball_temporal/`:

| file | does |
|---|---|
| `gsr_io.py` | GSR readers and the train / pool / select / test splits |
| `cache_main_detections.py` | runs the main detector and caches its ball detections; `list`, `run`, `summary` |

---

## 1. Setup

```bash
# Cell 1: GPUs, Ultralytics with YOLO26, folders, main detector weights
!nvidia-smi -L
!pip install -q "ultralytics>=8.4.0"
!python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
!mkdir -p /kaggle/working/kit/ball_temporal
!cp /kaggle/input/datasets/sabihalii/ball-detector-scripts/best.pt /kaggle/working/kit/best.pt
!ls -la /kaggle/working/kit
```

Expect two `Tesla T4` lines, a version of 8.4 or later, and `best.pt` at about 20 MB.

---

## 2. Write the scripts

**Cell 2.** First line `%%writefile /kaggle/working/kit/ball_temporal/gsr_io.py`, then paste the
whole of `cv_service/scripts/ball_temporal/gsr_io.py`.

**Cell 3.** First line `%%writefile /kaggle/working/kit/ball_temporal/cache_main_detections.py`,
then paste the whole of `cv_service/scripts/ball_temporal/cache_main_detections.py`.

`%%writefile` must be the very first line of each cell.

---

## 3. Check the splits

```bash
# Cell 4
!python /kaggle/working/kit/ball_temporal/cache_main_detections.py list --src /kaggle/input/datasets/sabihalii/soccernetgsr
```

Expect:

| split | sequences |
|---|---|
| `train` | 57 |
| `pool` | 33 |
| `select` | 10 |
| `test` | 15: 021 022 023 026 030 033 036 040 044 047 050 054 080 086 090 |

If `test` has fewer than 15, the `valid` folder in the input dataset is incomplete; stop and fix
the input. Paste the `select` line back so it's recorded in the plan.

---

## 4. Smoke test (one sequence, 32 frames, GPU 0)

```bash
# Cell 5
!python /kaggle/working/kit/ball_temporal/cache_main_detections.py run --src /kaggle/input/datasets/sabihalii/soccernetgsr --out /kaggle/working/main_cache_smoke --weights /kaggle/working/kit/best.pt --seqs SNGS-021 --max_frames 32 --device 0 --half
!python -c "import json,glob; d=json.load(open(glob.glob('/kaggle/working/main_cache_smoke/*/SNGS-021.json')[0])); print(d['meta']); fr=d['frames']; print(len(fr), 'frames'); [print(k, v) for k, v in list(fr.items())[:5]]"
```

Check:
- `32 frames`, and the line ends with a ms/frame figure. Note it; it sets the time estimate.
- Each frame has `balls` (up to 5 entries of `cx, cy, w, h, conf`, highest first) and `gt`.
- Where `gt` is non-empty, the top ball's `cx, cy` is usually within a few pixels of it.

Then remove the smoke output:

```bash
# Cell 6
!rm -rf /kaggle/working/main_cache_smoke
```

---

## 5. Full run (all 115 sequences, both GPUs)

```bash
# Cell 7
!(python /kaggle/working/kit/ball_temporal/cache_main_detections.py run --src /kaggle/input/datasets/sabihalii/soccernetgsr --out /kaggle/working/main_cache --weights /kaggle/working/kit/best.pt --device 0 --half --shard 0 --nshards 2 & python /kaggle/working/kit/ball_temporal/cache_main_detections.py run --src /kaggle/input/datasets/sabihalii/soccernetgsr --out /kaggle/working/main_cache --weights /kaggle/working/kit/best.pt --device 1 --half --shard 1 --nshards 2 & wait)
```

- Both processes print into the cell live. Each line starts with `[gpu0]` or `[gpu1]` and shows
  the sequence, its ms/frame and the elapsed time.
- **Time:** about 86k frames split over two T4s. Multiply the smoke test's ms/frame by about
  43,000 for a rough estimate. The time hasn't been measured yet.
- **If the session stops**, run the same cell again. Finished sequences print `cached` and are
  skipped.

Check that everything is there:

```bash
# Cell 8
!for s in train pool select test; do echo "$s $(ls /kaggle/working/main_cache/$s 2>/dev/null | grep -c '\.json$')"; done
```

Expect `train 57`, `pool 33`, `select 10`, `test 15`.

---

## 6. Summary

```bash
# Cell 9
!python /kaggle/working/kit/ball_temporal/cache_main_detections.py summary --out /kaggle/working/main_cache
```

It prints, and saves to `/kaggle/working/main_cache/summary.md`, one row per split:

- frames, and frames with a GT ball;
- the main detector's recall (top ball within 15 px at conf ≥ 0.15);
- hard frames: GT ball, but no detection at conf ≥ 0.15 within 15 px.

What to look for:
- **`test` recall should be close to 59%**, the figure from the v1 evaluation. If it's far off,
  something differs from that run; stop and check.
- **`train` recall should be clearly higher than `pool`, `select` and `test`.** That's the
  memorisation that decision A works around. If they're about the same, say so: hard examples
  could then come from `train` too.
- **The `pool` hard-frame count** is the size of the hard-example set before ×3 oversampling.

Paste the table back; it goes into `PERF_LOG.md` and the plan.

---

## 7. Keep the cache for the next step

The next step's builder reads `/kaggle/working/main_cache`. It is a few tens of MB of JSON.

```bash
# Cell 10
!cd /kaggle/working && zip -qr main_cache.zip main_cache && ls -la main_cache.zip
```

Then do one of these:
- **Save Version → Quick Save** on the notebook. In a later notebook, Add Input → this
  notebook's output gives `main_cache.zip` without uploading anything.
- Or download `main_cache.zip` from the output panel.
