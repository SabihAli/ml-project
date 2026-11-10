"""
build_phase4_notebook.py -- generate notebooks/pipeline_improvement_1/phase4_tracker.ipynb, the
Kaggle notebook for Phase 4 of docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md (the robust ball tracker,
gsr_pipeline/ball_tracker_robust.py).

Code goes in as %%writefile cells (written to /kaggle/working/kit/cv_service/<repo path>). Inputs:
the GSR dataset and phase0-models, both already on Kaggle. No new upload: the notebook re-caches
the main detector on select + test with athlete boxes (--athletes) and recomputes camera motion.

    python scripts/pipeline_improvement_1/phase4/build_phase4_notebook.py
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
    "scripts/ball_eval/eval_ball_temporal.py",
    "gsr_pipeline/ball_gate.py",
    "gsr_pipeline/ball_tracker_robust.py",
    "scripts/pipeline_improvement_1/phase1/phase1_sweep.py",
    "scripts/pipeline_improvement_1/phase4/phase4_sweep.py",
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
        return (sha or "no git") + (" + uncommitted changes" if dirty else "")
    except Exception:
        return "unknown"


INTRO = """
# Pipeline Improvement Plan 1 -- Phase 4: robust ball tracker

The tracker (`gsr_pipeline/ball_tracker_robust.py`) replaces the "last sighting + 15 frames" rule:
an IMM filter (calm + manoeuvre models) with camera-motion compensation, an occlusion model that
attaches a hidden ball to the athlete covering it, cut handling (no motion estimate -> one strong
detection restarts the track), and the c5c window centred on the tracker's prediction. Its knobs
are chosen on `select`, then the tracker, its ablations and the Phase 1 system run on `test`.

**Settings:** Accelerator **GPU T4 x2**, Internet **on** (pip installs ultralytics>=8.4).

**Inputs** (both already added for Phase 0/1; no new upload):

| input | contents |
|---|---|
| SoccerNet-GSR dataset | `train/` + `valid/` |
| phase0-models | `main_best.pt`, `c5c_best.pt` |

**Time:** setup ~3 min, cache with athletes ~16 min (2 GPUs), camera motion ~5 min, tracker search
~40-80 min, report ~3 min. `SMOKE = True` for a ~10-minute dry run (1 sequence per split, 150 frames).
"""

SETTINGS = '''
# Settings. Input paths confirmed (the same datasets as Phase 0/1).
import os, subprocess, sys
from pathlib import Path

SMOKE = False

GSR_INPUT  = "/kaggle/input/datasets/sabihalii/soccernetgsr"
MODELS_DIR = "/kaggle/input/datasets/sabihalii/phase0-models"

KIT  = "__KIT__"
WORK = Path("/kaggle/working/" + ("phase4_smoke" if SMOKE else "phase4"))
CACHE, MOTION = WORK / "main_cache", WORK / "motion_cache"
WORK.mkdir(parents=True, exist_ok=True)


def _find(root, name):
    hits = sorted(Path(root).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}: check the path")
    return str(hits[0])


MAIN, C5C = _find(MODELS_DIR, "main_best.pt"), _find(MODELS_DIR, "c5c_best.pt")


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


print("main:", MAIN, "| c5c:", C5C, "| work:", WORK, "| smoke:", SMOKE)
'''

STAGE_CELLS = [
    ("## 1. Environment",
     "Installs `ultralytics>=8.4` (YOLO26) and lists the GPUs and splits.",
     '''STREAM([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4"])
import torch
NGPU = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(NGPU)])
sys.path.insert(0, f"{KIT}/scripts/ball_temporal")
from gsr_io import discover
REFS = discover(GSR_INPUT)
SMOKE_SEQS = [next(r.name for r in REFS if r.split == s) for s in ("select", "test")]
print({s: sum(r.split == s for r in REFS) for s in ("select", "test")}, "(expect 10 / 15) | smoke seqs:", SMOKE_SEQS)'''),
    ("## 2. Main-detector cache with athlete boxes (select + test)",
     "Ball candidates (up to 5, conf >= 0.05) and athlete boxes (up to 40, conf >= 0.3) per frame, YOLO26s at "
     "imgsz 1280, FP16, one process per GPU.",
     '''base = [sys.executable, "-u", f"{KIT}/scripts/ball_temporal/cache_main_detections.py", "run", "--src", GSR_INPUT,
        "--out", CACHE, "--weights", MAIN, "--splits", "select", "test", "--athletes", "--half"]
if SMOKE:
    base += ["--seqs", *SMOKE_SEQS, "--max_frames", "150"]
n = max(NGPU, 1)
procs = [subprocess.Popen([str(c) for c in base + ["--device", str(i), "--shard", str(i), "--nshards", str(n)]])
         for i in range(n)]
codes = [p.wait() for p in procs]
assert not any(codes), f"cache failed: exit codes {codes}"
print({s: len(list((CACHE / s).glob("*.json"))) for s in ("select", "test")})'''),
    ("## 3. Camera motion (select + test)",
     "A 2x3 similarity per frame (frame t-1 -> t) from sparse optical flow with people masked; null when it "
     "cannot be estimated (a cut, a close-up), which the tracker treats as a camera break.",
     '''STREAM([sys.executable, "-u", f"{KIT}/scripts/ball_temporal/camera_motion_cache.py", "run", "--src", GSR_INPUT,
        "--out", MOTION, "--splits", "select", "test", "--workers", str(os.cpu_count() or 4)]
       + (["--seqs", *SMOKE_SEQS] if SMOKE else []))'''),
    ("## 4. Tracker search on select, frozen evaluation on test",
     "`scripts/pipeline_improvement_1/phase4/phase4_sweep.py`: coordinate descent over 12 knobs (2 passes) on "
     "`select`, then on `test`: Phase 1, the tracker (measured and +predicted), and ablations without the occlusion "
     "model, without camera motion, and with one motion model.",
     '''STREAM([sys.executable, "-u", f"{KIT}/scripts/pipeline_improvement_1/phase4/phase4_sweep.py",
        "--src", GSR_INPUT, "--cache", CACHE, "--motion", MOTION, "--ckpt", C5C, "--out", WORK]
       + (["--max_seqs", "1", "--max_frames", "150", "--passes", "1"] if SMOKE else []))
SHOW(WORK / "report_phase4.md")'''),
    ("## 5. Miss breakdown for the tracker (test)",
     "The Phase 0 report on the tracker's measured outputs, to compare buckets with Phase 0 and Phase 1.",
     '''STREAM([sys.executable, "-u", f"{KIT}/scripts/pipeline_improvement_1/phase0/phase0_report.py",
        "--src", GSR_INPUT, "--diag", WORK / "frames_tracker", "--out", WORK / "buckets_tracker",
        "--splits", "test", "--frames_per_hash", "0"])
SHOW(WORK / "buckets_tracker" / "report_phase0.md")'''),
    ("## 6. Deliverables",
     "Packs the work folder (reports, search log, caches) into `/kaggle/working/phase4_deliverables.zip` with a "
     "sha256 manifest. Download that one file (Output tab) or Save Version.",
     '''import hashlib, json, zipfile
out = Path("/kaggle/working/phase4_deliverables.zip")
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
    stamp = (f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/pipeline_improvement_1/phase4/"
             f"build_phase4_notebook.py` (commit `{git_commit()}`).\n\n| kit file | sha256 |\n|---|---|\n"
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
    ap.add_argument("--out", type=Path, default=ROOT / "notebooks" / "pipeline_improvement_1" / "phase4_tracker.ipynb")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
