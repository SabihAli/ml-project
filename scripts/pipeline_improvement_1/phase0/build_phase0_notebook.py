"""
build_phase0_notebook.py -- generate notebooks/pipeline_improvement_1/phase0_baseline.ipynb, the
Kaggle notebook for Phase 0 of docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md.

The notebook carries every code file as a %%writefile cell (written to
/kaggle/working/kit/cv_service/<repo path>), so the only uploads are the data and the two model
checkpoints (data/notebook_uploads/phase0_models.zip). Regenerate after changing any kit file;
the first cell records the git commit and each file's sha256, so a stale notebook is visible.

Stages: env, data, cache (select + test only), motion, diagnose (notebooks/kaggle/ball_pipeline.py),
then phase0_report.py (CPU), then a deliverables zip.

    python scripts/pipeline_improvement_1/phase0/build_phase0_notebook.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KIT_ROOT = "/kaggle/working/kit/cv_service"

KIT_FILES = [
    "scripts/ball_temporal/gsr_io.py",
    "scripts/ball_temporal/cache_main_detections.py",
    "scripts/ball_temporal/camera_motion_cache.py",
    "scripts/ball_temporal/heatmap_model.py",
    "scripts/ball_temporal/build_ball_temporal_dataset.py",
    "scripts/ball_temporal/train_ball_temporal.py",
    "scripts/ball_eval/eval_ball_temporal.py",
    "scripts/ball_eval/diagnose_ball_misses.py",
    "gsr_pipeline/ball_tracker.py",
    "notebooks/kaggle/ball_pipeline.py",
    "scripts/pipeline_improvement_1/phase0/phase0_report.py",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def git_commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *KIT_FILES], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + (" + uncommitted changes" if dirty else "")
    except Exception:
        return "unknown"


INTRO = """
# Pipeline Improvement Plan 1 -- Phase 0: baseline and miss breakdown

Replays the ball path exactly as wired (main detector top ball at conf >= 0.15, else c5c on a 640 window
at the last sighting for up to 15 frames, threshold 0.70) on `select` (10 sequences) and `test` (15
held-out sequences). It then sorts every miss into buckets B1-B7, works out the ceilings, reports 5/10/15 px
recall, per-stage and per-size recall and continuity, and checks the splits for leakage.
Nothing is trained.

**Settings:** Accelerator **GPU T4 x2** (one GPU is enough), Internet **on** (pip installs ultralytics>=8.4).

**Inputs** (Add Input), then set the two paths in the Settings cell:

| input | contents |
|---|---|
| SoccerNet-GSR dataset | `train/` + `valid/` sequence folders (or `train.zip` / `valid.zip`); `valid` must have all 15 test sequences |
| `phase0_models.zip` (uploaded as a dataset) | `main_best.pt` (= `runs/yolo26s_athlete_ball/best.pt`), `c5c_best.pt` (= `runs/ball_temporal/c5c/best.pt`), `MANIFEST.json` |

**Time:** setup ~5-15 min, cache (select + test) ~15-25 min, motion ~5-10 min, diagnosis ~10-20 min,
report ~5 min.

Run the cells top to bottom. For a ~10-minute dry run first, set `SMOKE = True`.
"""

SETTINGS = '''
# Settings. Input paths confirmed.
import os, subprocess, sys
from pathlib import Path

SMOKE = False   # True: 1 sequence per split, 64 frames per sequence

GSR_INPUT  = "/kaggle/input/datasets/sabihalii/soccernetgsr"         # folder holding train/ and valid/ (or the zips)
MODELS_DIR = "/kaggle/input/datasets/sabihalii/phase0-models"       # folder holding main_best.pt and c5c_best.pt

KIT  = "__KIT__"
WORK = Path("/kaggle/working/" + ("phase0_smoke" if SMOKE else "phase0"))


def _model(name):
    hits = sorted(Path(MODELS_DIR).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {MODELS_DIR}: check MODELS_DIR")
    return str(hits[0])


BASE = ["--gsr_input", GSR_INPUT, "--work", str(WORK), "--main_weights", _model("main_best.pt"),
        "--c5c_ckpt", _model("c5c_best.pt"), "--cache_splits", "select,test", "--copy", "never"]
if SMOKE:
    BASE += ["--max_seqs", "1", "--max_frames", "64", "--expect_test", "0"]


def STREAM(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError(f"stopped (exit {p.returncode}); fix the cause above and re-run this cell")


def RUN(stages):
    STREAM([sys.executable, "-u", f"{KIT}/notebooks/kaggle/ball_pipeline.py", "--stages", stages, *BASE])


def SHOW(path):
    from IPython.display import Markdown, display
    p = Path(path)
    display(Markdown(p.read_text(encoding="utf-8")) if p.exists() else Markdown(f"*{p} not written yet*"))


print("kit:", KIT, "| work:", WORK, "| smoke:", SMOKE)
print("main:", _model("main_best.pt"), "| c5c:", _model("c5c_best.pt"))
'''

STAGE_CELLS = [
    ("## 1. Setup: environment, data, weights",
     "pip installs `ultralytics>=8.4`, copies the weights and checks the splits (15 test, 10 select). "
     "The dataset is read in place; it is not copied.",
     'RUN("env,data")'),
    ("## 2. Main-detector cache (select + test) and camera motion",
     "Up to 5 ball candidates per frame at conf >= 0.05 (YOLO26s, imgsz 1280, FP16), one process per GPU. "
     "Then camera motion for select + test.",
     'RUN("cache,motion")\nSHOW(WORK / "reports" / "main_cache_summary.md")'),
    ("## 3. Replay main + c5c as wired, and the existing cause diagnosis",
     "`scripts/ball_eval/diagnose_ball_misses.py`: every frame's output, every specialist call (window, raw "
     "confidence), cause flags, ball displacement, and 200 audit sheets of test misses.",
     'RUN("diagnose")\nSHOW(WORK / "reports" / "diagnosis" / "report_diagnosis.md")'),
    ("## 4. Phase 0 report",
     "CPU only: the B1-B7 miss breakdown and ceilings, 5/10/15 px recall per stage, recall by size, "
     "continuity, and the split-integrity checks (game/clip overlap, near-duplicate frames).",
     'STREAM([sys.executable, "-u", f"{KIT}/scripts/pipeline_improvement_1/phase0/phase0_report.py",\n'
     '        "--src", GSR_INPUT, "--diag", str(WORK / "reports" / "diagnosis"),\n'
     '        "--out", str(WORK / "reports" / "phase0")] + (["--frames_per_hash", "0"] if SMOKE else []))\n'
     'SHOW(WORK / "reports" / "phase0" / "report_phase0.md")'),
    ("## 5. Deliverables",
     "Packs reports, caches, logs and state into `/kaggle/working/phase0_deliverables.zip` with a "
     "sha256 manifest. Download that one file (Output tab) or Save Version.",
     '''import hashlib, json, zipfile
out = Path("/kaggle/working/phase0_deliverables.zip")
pick = [WORK / "reports", WORK / "main_cache", WORK / "motion_cache", WORK / "logs", WORK / "state.json"]
manifest = {}
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for root in pick:
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for f in files:
            rel = f.relative_to(WORK).as_posix()
            zf.write(f, rel)
            manifest[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    zf.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
print(f"{out}: {len(manifest)} files, {out.stat().st_size / 1e6:.1f} MB")'''),
]


def build(out: Path):
    files = [(rel, (ROOT / rel).read_text(encoding="utf-8"), hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:12])
             for rel in KIT_FILES]
    stamp = (f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/pipeline_improvement_1/phase0/"
             f"build_phase0_notebook.py` from commit `{git_commit()}`.\n\n| kit file | sha256 |\n|---|---|\n"
             + "\n".join(f"| `{rel}` | `{h}` |" for rel, _, h in files))
    dirs = sorted({Path(r).parent.as_posix() for r in KIT_FILES})
    cells = [md(INTRO + "\n" + stamp), code(SETTINGS.replace("__KIT__", KIT_ROOT)),
             md(f"## 0. Kit files\n\nThese cells write the code to `{KIT_ROOT}`. Run them all; don't edit them here "
                "(edit the repo and regenerate the notebook)."),
             code(f"import os\nfor d in {dirs!r}:\n    os.makedirs(f'{KIT_ROOT}/{{d}}', exist_ok=True)\nprint('kit folders ready')")]
    cells += [code(f"%%writefile {KIT_ROOT}/{rel}\n{text}") for rel, text, _ in files]
    for title, body, src in STAGE_CELLS:
        cells += [md(f"{title}\n\n{body}"), code(src)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(cells)} cells, {len(files)} kit files)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "notebooks" / "pipeline_improvement_1" / "phase0_baseline.ipynb")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
