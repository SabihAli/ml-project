"""
build_sort_notebook.py -- generate notebooks/pipeline_improvement_1/phase4b_sort.ipynb, the Kaggle
notebook for the Phase 4 follow-up (scripts/pipeline_improvement_1/phase4/sort_compare.py):
boxmot ByteTrack / OC-SORT / BoT-SORT driving main + c5c, against Phase 1.

Inputs: the GSR dataset, phase0-models, and phase4-caches (data/notebook_uploads/phase4_caches.zip:
the Phase 4 main cache with athlete boxes + camera motion, so nothing is re-cached).

    python scripts/pipeline_improvement_1/phase4/build_sort_notebook.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_phase4_notebook as b4  # noqa: E402

b4.KIT_FILES = [
    "scripts/ball_temporal/gsr_io.py",
    "scripts/ball_temporal/heatmap_model.py",
    "scripts/ball_temporal/build_ball_temporal_dataset.py",
    "scripts/ball_eval/eval_ball_temporal.py",
    "gsr_pipeline/ball_gate.py",
    "gsr_pipeline/ball_tracker_robust.py",
    "scripts/pipeline_improvement_1/phase1/phase1_sweep.py",
    "scripts/pipeline_improvement_1/phase4/phase4_sweep.py",
    "scripts/pipeline_improvement_1/phase4/sort_compare.py",
]

b4.INTRO = """
# Pipeline Improvement Plan 1 -- Phase 4b: SORT-family trackers vs Phase 1

Off-the-shelf trackers from `boxmot` (ByteTrack, OC-SORT, BoT-SORT without re-id) take the main
detector's balls plus the c5c specialist's peak as detections; one ball track is output per frame.
Each tracker's knobs are chosen on `select`, then all run frozen on `test` against Phase 1, with the
same scorer and kill criterion as Phase 4. BoT-SORT is also run on `test` with camera compensation.

**Settings:** Accelerator **GPU T4 x2**, Internet **on** (pip installs boxmot).

**Inputs:**

| input | contents |
|---|---|
| SoccerNet-GSR dataset | `train/` + `valid/` |
| phase0-models | `main_best.pt`, `c5c_best.pt` |
| phase4-caches (new) | `main_cache/`, `motion_cache/` from the Phase 4 run |

**Time:** setup ~2 min; the three trackers run in parallel (two on GPU 0, one on GPU 1), ~40-70 min.
`SMOKE = True` for a ~10-minute dry run (1 sequence per split, 150 frames, 1 pass).
"""

b4.SETTINGS = '''
# Settings. CACHE_INPUT: confirm the path once the phase4-caches dataset is added.
import os, subprocess, sys
from pathlib import Path

SMOKE = False

GSR_INPUT   = "/kaggle/input/datasets/sabihalii/soccernetgsr"
MODELS_DIR  = "/kaggle/input/datasets/sabihalii/phase0-models"
CACHE_INPUT = "/kaggle/input/datasets/sabihalii/phase4-caches"

KIT  = "__KIT__"
WORK = Path("/kaggle/working/" + ("phase4b_smoke" if SMOKE else "phase4b"))
WORK.mkdir(parents=True, exist_ok=True)


def _find(root, name, kind="file"):
    hits = sorted(p for p in Path(root).rglob(name) if (p.is_dir() if kind == "dir" else p.is_file()))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}: check the path")
    return hits[0]


MAIN, C5C = _find(MODELS_DIR, "main_best.pt"), _find(MODELS_DIR, "c5c_best.pt")
CACHE, MOTION = _find(CACHE_INPUT, "main_cache", "dir"), _find(CACHE_INPUT, "motion_cache", "dir")


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


print("c5c:", C5C, "| cache:", CACHE, "| motion:", MOTION, "| work:", WORK, "| smoke:", SMOKE)
print({s: len(list((CACHE / s).glob("*.json"))) for s in ("select", "test")}, "(expect 10 / 15)")
'''

b4.STAGE_CELLS = [
    ("## 1. Environment",
     "Installs `boxmot==26.0.0` and lists the GPUs.",
     '''STREAM([sys.executable, "-m", "pip", "install", "-q", "boxmot==26.0.0"])
import torch
NGPU = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(NGPU)])'''),
    ("## 2. Search on select, frozen evaluation on test",
     "One process per tracker (logs in `WORK/<tracker>/sort_log.txt`; this cell prints them when done).",
     '''import time
procs = {}
for i, name in enumerate(["bytetrack", "ocsort", "botsort"]):
    out = WORK / name
    out.mkdir(exist_ok=True)
    cmd = [sys.executable, "-u", f"{KIT}/scripts/pipeline_improvement_1/phase4/sort_compare.py", "--src", GSR_INPUT,
           "--cache", CACHE, "--motion", MOTION, "--ckpt", C5C, "--out", out, "--trackers", name,
           "--device", f"cuda:{i % max(NGPU, 1)}"]
    if SMOKE:
        cmd += ["--max_seqs", "1", "--max_frames", "150", "--passes", "1"]
    procs[name] = subprocess.Popen([str(c) for c in cmd], stdout=open(out / "stdout.txt", "w"), stderr=subprocess.STDOUT)
t0 = time.time()
while any(p.poll() is None for p in procs.values()):
    time.sleep(60)
    print(f"{(time.time() - t0) / 60:.0f} min:", {n: p.poll() for n, p in procs.items()}, flush=True)
for n in procs:
    print(f"===== {n} =====")
    print((WORK / n / "stdout.txt").read_text()[-3000:])
assert not any(p.returncode for p in procs.values()), {n: p.returncode for n, p in procs.items()}
for n in procs:
    SHOW(WORK / n / "report_sort.md")'''),
    ("## 3. Deliverables",
     "Packs the work folder into `/kaggle/working/phase4b_deliverables.zip` with a sha256 manifest.",
     '''import hashlib, json, zipfile
out = Path("/kaggle/working/phase4b_deliverables.zip")
manifest = {}
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(p for p in WORK.rglob("*") if p.is_file()):
        rel = f.relative_to(WORK).as_posix()
        zf.write(f, rel)
        manifest[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    zf.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
print(f"{out}: {len(manifest)} files, {out.stat().st_size / 1e6:.1f} MB")'''),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=b4.ROOT / "notebooks" / "pipeline_improvement_1" / "phase4b_sort.ipynb")
    b4.build(ap.parse_args().out)


if __name__ == "__main__":
    main()
