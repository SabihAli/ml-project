"""
build_phase1_notebook.py -- generate notebooks/pipeline_improvement_1/phase1_gate_threshold.ipynb,
the Kaggle notebook for Phase 1 tasks 1-2 of docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md (the
detector-ball plausibility gate and the c5c threshold sweep).

Code goes in as %%writefile cells (written to /kaggle/working/kit/cv_service/<repo path>). Uploads:
the GSR dataset and phase0-models (both already on Kaggle from Phase 0) and
data/notebook_uploads/phase1_caches.zip (Phase 0's main-detector cache). Regenerate after
changing any kit file.

    python scripts/pipeline_improvement_1/phase1/build_phase1_notebook.py
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
    "scripts/ball_temporal/heatmap_model.py",
    "scripts/ball_temporal/build_ball_temporal_dataset.py",
    "scripts/ball_eval/eval_ball_temporal.py",
    "gsr_pipeline/ball_gate.py",
    "scripts/pipeline_improvement_1/phase1/phase1_sweep.py",
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
        return sha or "no git"
    except Exception:
        return "unknown"


INTRO = """
# Pipeline Improvement Plan 1 -- Phase 1: plausibility gate + c5c threshold

Replays main + c5c exactly as `BallFuser.update` runs it, on Phase 0's cached main detections, with
the new detector-ball plausibility gate (`gsr_pipeline/ball_gate.py`) and a c5c threshold sweep.
Everything is chosen on `select` and frozen for `test`; a change is kept only if test recall gains
>= 0.5 pp at <= 1 pp precision loss. Then the Phase 0 miss breakdown is re-run on the chosen
configuration.

**Settings:** Accelerator **GPU T4** (one GPU is used). Internet not needed.

**Inputs** (Add Input), then confirm the paths in the Settings cell:

| input | contents |
|---|---|
| SoccerNet-GSR dataset (same as Phase 0) | `train/` + `valid/` |
| phase0-models (same as Phase 0) | `main_best.pt`, `c5c_best.pt` |
| `phase1_caches.zip` (new dataset) | `main_cache/select/*.json`, `main_cache/test/*.json` from Phase 0 |

**Time:** sweep ~20-40 min (19 gate configs + 40 threshold runs on `select`, 4 on `test`; the
specialist is memoised per window), report ~5 min. `SMOKE = True` for a ~5-minute dry run.
"""

SETTINGS = '''
# Settings. Input paths confirmed.
import subprocess, sys
from pathlib import Path

SMOKE = False   # True: 1 sequence per split, 100 frames per sequence

GSR_INPUT   = "/kaggle/input/datasets/sabihalii/soccernetgsr"
MODELS_DIR  = "/kaggle/input/datasets/sabihalii/phase0-models"
CACHE_INPUT = "/kaggle/input/datasets/sabihalii/phase1-caches"     # folder holding main_cache/ (or select/ + test/)

KIT  = "__KIT__"
WORK = Path("/kaggle/working/" + ("phase1_smoke" if SMOKE else "phase1"))
WORK.mkdir(parents=True, exist_ok=True)


def _find(root, name):
    hits = sorted(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}: check the path")
    return hits[0]


C5C = str(_find(MODELS_DIR, "c5c_best.pt"))
CACHE = str(_find(CACHE_INPUT, "test").parent)      # the folder that holds select/ and test/


def STREAM(cmd):
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise RuntimeError(f"stopped (exit {p.returncode}); fix the cause above and re-run this cell")


def SHOW(path):
    from IPython.display import Markdown, display
    p = Path(path)
    display(Markdown(p.read_text(encoding="utf-8")) if p.exists() else Markdown(f"*{p} not written yet*"))


import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (turn the accelerator on)")
print("c5c:", C5C, "| cache:", CACHE, "| work:", WORK, "| smoke:", SMOKE)
print("cache files:", {s: len(list((Path(CACHE) / s).glob("*.json"))) for s in ("select", "test")}, "(expect 10 / 15)")
'''

STAGE_CELLS = [
    ("## 1. Sweep: gate grid, thresholds, frozen test",
     "`scripts/pipeline_improvement_1/phase1/phase1_sweep.py`. Stage A: 18 gate configurations + gate off at "
     "threshold 0.70 on `select`. Stage B: thresholds 0.40-0.85 for gate off and the best 3 gates. Then the four "
     "frozen picks (baseline, threshold only, gate only, gate + threshold) on `test`, with and without SNGS-050.",
     'STREAM([sys.executable, "-u", f"{KIT}/scripts/pipeline_improvement_1/phase1/phase1_sweep.py",\n'
     '        "--src", GSR_INPUT, "--cache", CACHE, "--ckpt", C5C, "--out", WORK]\n'
     '       + (["--max_seqs", "1", "--max_frames", "100", "--top_gates", "1"] if SMOKE else []))\n'
     'SHOW(WORK / "report_phase1.md")'),
    ("## 2. Miss breakdown for the chosen configuration",
     "The Phase 0 report (buckets B0-B7, ceilings, per-stage and per-size recall) on the chosen configuration's "
     "replay, to compare with Phase 0. Split-integrity checks are skipped (done in Phase 0).",
     '''import json
chosen = json.loads((WORK / "phase1.json").read_text())["chosen"]
STREAM([sys.executable, "-u", f"{KIT}/scripts/pipeline_improvement_1/phase0/phase0_report.py",
        "--src", GSR_INPUT, "--diag", WORK / f"frames_{chosen}", "--out", WORK / f"buckets_{chosen}",
        "--frames_per_hash", "0"])
SHOW(WORK / f"buckets_{chosen}" / "report_phase0.md")'''),
    ("## 3. Deliverables",
     "Packs everything in the work folder into `/kaggle/working/phase1_deliverables.zip` with a sha256 manifest. "
     "Download that one file (Output tab) or Save Version.",
     '''import hashlib, json, zipfile
out = Path("/kaggle/working/phase1_deliverables.zip")
manifest = {}
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(p for p in WORK.rglob("*") if p.is_file()):
        rel = f.relative_to(WORK).as_posix()
        zf.write(f, rel)
        manifest[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    zf.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
print(f"{out}: {len(manifest)} files, {out.stat().st_size / 1e6:.1f} MB")'''),
]


def build(out: Path):
    files = [(rel, (ROOT / rel).read_text(encoding="utf-8"), hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:12])
             for rel in KIT_FILES]
    stamp = (f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/pipeline_improvement_1/phase1/"
             f"build_phase1_notebook.py` (commit `{git_commit()}`).\n\n| kit file | sha256 |\n|---|---|\n"
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
    ap.add_argument("--out", type=Path, default=ROOT / "notebooks" / "pipeline_improvement_1" / "phase1_gate_threshold.ipynb")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
