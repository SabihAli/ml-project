"""
build_ball_notebook.py -- generate notebooks/kaggle_ball_improvement.ipynb, the Kaggle notebook
for docs/plans/BALL_IMPROVEMENT_PLAN.md items 1-3, from the repo's own files.

The notebook carries the code as %%writefile cells (one per kit file, written to
/kaggle/working/kit/cv_service/<same path as in the repo>), so nothing needs uploading except the
data and two checkpoints. Regenerate after any change to a kit file; the notebook's first cell
records the git commit and every file's sha256, so a stale notebook is visible.

    python scripts/kaggle/build_ball_notebook.py            # -> notebooks/kaggle_ball_improvement.ipynb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT = "/kaggle/working/kit/cv_service"

# Everything the stages import, by repo path.
KIT_FILES = [
    "scripts/ball_temporal/gsr_io.py",
    "scripts/ball_temporal/cache_main_detections.py",
    "scripts/ball_temporal/camera_motion_cache.py",
    "scripts/ball_temporal/heatmap_model.py",
    "scripts/ball_temporal/build_ball_temporal_dataset.py",
    "scripts/ball_temporal/train_ball_temporal.py",
    "scripts/ball_temporal/export_ball_specialist.py",
    "scripts/eval/ball_det/eval_ball_temporal.py",
    "scripts/eval/ball_det/ball_system.py",
    "scripts/eval/ball_det/diagnose_ball_misses.py",
    "gsr_pipeline/ball_tracker.py",
    "notebooks/kaggle/ball_pipeline.py",
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *KIT_FILES], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + (" + uncommitted changes" if dirty else "")
    except Exception:
        return "unknown"


INTRO = """
# Ball detection to 80%: items 1-3 on Kaggle

Runs `docs/plans/BALL_IMPROVEMENT_PLAN.md` items 1-3 end to end: **miss diagnosis**, the **ball
tracker** (tuned on `select`, gated on `test`), and the **c5d specialist** (dataset, 2-GPU training
warm-started from c5c, gates, ONNX export). Every stage streams its log into the cell, writes
`/kaggle/working/ball_improvement/logs/<stage>.log`, and is resumable: re-running a cell skips
finished work. The last cells collect everything into `/kaggle/working/deliverables/` and
`deliverables.zip`.

**Settings:** Accelerator **GPU T4 x2**, Internet **on** (pip).

**Inputs** (Add Input):

| input | what | used for |
|---|---|---|
| `sabihalii/soccernetgsr` | SoccerNet-GSR frames + labels (`train/`, `valid/`, optionally GSR `test/`) | everything |
| `sabihalii/ball-detector-scripts` | its `best.pt`: the main athlete/ball detector | main cache |
| a dataset holding **`c5c_best.pt`** | `runs/ball_temporal/c5c/best.pt` from the repo, renamed | diagnosis, tracker, c5d warm start |
| (optional) an earlier run's output | `main_cache.zip` or `ball_improvement/` | skip the cache / resume training |

The c5c checkpoint is not on Kaggle yet: upload `runs/ball_temporal/c5c/best.pt` (58 MB) as a new
version of `ball-detector-scripts` (or a new dataset) under the name `c5c_best.pt`. The runner finds it
anywhere under `/kaggle/input`.

**Time** (one 12 h session, 2xT4): setup + cache ~1 h, diagnosis ~20 min, tracker ~1-2 h, c5d dataset
~1 h, training gets what's left minus 2.5 h, eval + export + deliverables ~2 h. Training stops itself
before the session ends; see **Resume** at the bottom.

**Disk:** `/kaggle/working` keeps ~20 GB of saved output. The data stage copies the GSR dataset there
only if it fits next to the caches and the crop dataset (`--copy auto`); otherwise it reads it in place
from `/kaggle/input` and says so. Run **Cleanup** before Save Version.

**Run the cells top to bottom.** For a ~15-minute dry run first, set `SMOKE = True` in the next cell.
"""

SETTINGS = '''
# Settings. Edit paths here if your inputs are named differently; leave None to auto-detect.
import os, subprocess, sys
from pathlib import Path

SMOKE = False                       # True: 1 sequence per split, 64 frames, a 256-sample epoch -> ~15 min
GSR_INPUT = "/kaggle/input/datasets/sabihalii/soccernetgsr"
MAIN_WEIGHTS = None                 # e.g. "/kaggle/input/ball-detector-scripts/best.pt"
C5C_CKPT = None                     # e.g. "/kaggle/input/ball-detector-scripts/c5c_best.pt"
MAIN_CACHE_ZIP = None               # e.g. "/kaggle/input/<earlier-run>/deliverables/main_cache.zip"
EPOCHS, LR, BATCH = 20, 1.5e-4, 32  # c5d training (warm start from c5c; batch is per GPU)

KIT = "__KIT__"
WORK = "/kaggle/working/" + ("ball_smoke" if SMOKE else "ball_improvement")
BASE = ["--gsr_input", GSR_INPUT, "--work", WORK, "--epochs", str(1 if SMOKE else EPOCHS),
        "--lr", str(LR), "--batch", str(BATCH)]
for flag, val in (("--main_weights", MAIN_WEIGHTS), ("--c5c_ckpt", C5C_CKPT), ("--main_cache_zip", MAIN_CACHE_ZIP)):
    if val:
        BASE += [flag, val]
if SMOKE:
    BASE += ["--max_seqs", "1", "--max_frames", "64", "--limit", "256", "--search_passes", "1", "--copy", "never"]


def RUN(stages, *extra):
    """Run pipeline stages; every log line appears here as it is printed."""
    cmd = [sys.executable, "-u", f"{KIT}/notebooks/kaggle/ball_pipeline.py", "--stages", stages, *BASE, *extra]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError(f"stages {stages} stopped (exit {p.returncode}); fix the cause above and re-run this cell")


def SHOW(rel):
    """Render a report from the run folder."""
    from IPython.display import Markdown, display
    path = Path(WORK) / rel
    display(Markdown(path.read_text(encoding="utf-8")) if path.exists() else Markdown(f"*{path} not written yet*"))


print("kit:", KIT, "| work:", WORK, "| smoke:", SMOKE)
'''

STAGE_CELLS = [
    ("## 1. Setup: environment and data",
     "Installs `ultralytics>=8.4`, `onnx`, `onnxruntime`; copies the weights, and the GSR dataset if it fits; "
     "checks the splits (15 test sequences required).",
     'RUN("env,data")'),
    ("## 2. Main-detector cache and camera motion",
     "Up to 5 ball candidates per frame at conf >= 0.05, every split, one process per GPU (lines tagged "
     "`cache-gpu0` / `cache-gpu1`). Then camera motion for `select` and `test`. ~30-60 min.",
     'RUN("cache,motion")\nSHOW("reports/main_cache_summary.md")'),
    ("## 3. Item 1: miss diagnosis",
     "Replays main + c5c as wired, tags every miss by cause, measures ball displacement, estimates the "
     "ceiling per item, and writes 200 audit sheets (`reports/diagnosis/audit/`).",
     'RUN("diagnose")\nSHOW("reports/diagnosis/report_diagnosis.md")'),
    ("## 4. Item 2: ball tracker (+ item 3 window variants)",
     "Coordinate search on `select` (tracker knobs, then window centre / 3 windows / run-below, then the "
     "batch path), frozen and reported on `test`. Gate T at the end. ~1-2 h.",
     'RUN("track")\nSHOW("reports/track_c5c/report_track.md")'),
    ("## 5. Item 3: c5d dataset",
     "Window centre as chosen in step 4; pool hard positives >= 2 frames from the last sighting repeated x2.",
     'RUN("c5d_data")\nSHOW("reports/c5d_dataset_stats.md")'),
    ("## 6. Item 3: c5d training",
     "2-GPU DDP via torchrun, warm start from c5c. Per-100-step lines with loss, lr, s/it, epoch ETA and GPU "
     "memory; per-epoch `select` score and run ETA. Stops before the session budget runs out.",
     'RUN("c5d_train")'),
    ("## 7. Item 3: c5d gates and export",
     "Gates 1-3 with a new threshold chosen on `select`, the tracker re-run with c5d (gate S), then the ONNX "
     "export and parity check.",
     'RUN("c5d_eval,export")\nSHOW("reports/eval_c5d/report.md")\nSHOW("reports/track_c5d/report_track.md")'),
    ("## 8. Deliverables",
     "`/kaggle/working/deliverables/`: `SUMMARY.md`, `PERF_LOG_entry.md`, `reports/`, `configs/tracker_config.json`, "
     "`models/` (best.pt, ONNX + sidecar, history, curves), `main_cache.zip`, `logs/`, `MANIFEST.json` "
     "(sha256 per file); and `deliverables.zip`. A failed stage packages what exists automatically.",
     'RUN("deliver")\nfrom IPython.display import Markdown, display\n'
     'display(Markdown(Path("/kaggle/working/deliverables/SUMMARY.md").read_text(encoding="utf-8")))'),
    ("## 9. Cleanup before Save Version",
     "Deletes the GSR copy and the crop dataset (both rebuilt on resume) so the output fits in 20 GB.",
     'RUN("cleanup")'),
]

RESUME = """
## Resume in a new session (only if training stopped at the time budget)

1. In the finished session: run **Cleanup**, then **Save Version** (Quick Save).
2. New session: Add Input -> this notebook's output. Run the Settings cell and the kit cells.
3. Set `PREV` below to that input's `ball_improvement` folder and run the cell. It restores the state,
   caches, reports and `last.pt`, rebuilds the crop dataset, and continues training. Then run cells 7-9.
"""

RESUME_CODE = '''
PREV = "/kaggle/input/<this-notebook-output>/ball_improvement"   # edit
RUN("restore,env,data,c5d_data,c5d_train", "--restore_from", PREV)
'''


def build(out: Path):
    files = []
    for rel in KIT_FILES:
        p = ROOT / rel
        files.append((rel, p.read_text(encoding="utf-8"), hashlib.sha256(p.read_bytes()).hexdigest()[:12]))
    stamp = (f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/kaggle/build_ball_notebook.py` from commit "
             f"`{git_commit()}`.\n\n| kit file | sha256 |\n|---|---|\n"
             + "\n".join(f"| `{rel}` | `{h}` |" for rel, _, h in files))
    dirs = sorted({Path(r).parent.as_posix() for r in KIT_FILES})
    cells = [md(INTRO + "\n" + stamp), code(SETTINGS.replace("__KIT__", KIT_ROOT)),
             md(f"## 0. Kit files\n\nThese cells write the code to `{KIT_ROOT}`. Run them all; don't edit them "
                "here (edit the repo and regenerate the notebook)."),
             code(f"import os\nfor d in {dirs!r}:\n    os.makedirs(f'{KIT_ROOT}/{{d}}', exist_ok=True)\n"
                  "print('kit folders ready')")]
    for rel, text, _ in files:
        cells.append(code(f"%%writefile {KIT_ROOT}/{rel}\n{text}"))
    for title, body, src in STAGE_CELLS:
        cells += [md(f"{title}\n\n{body}"), code(src)]
    cells += [md(RESUME), code(RESUME_CODE)]
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(cells)} cells, {len(files)} kit files)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "notebooks" / "kaggle_ball_improvement.ipynb")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
