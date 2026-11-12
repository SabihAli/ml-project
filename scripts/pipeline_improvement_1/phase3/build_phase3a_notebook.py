"""
build_phase3a_notebook.py -- generate notebooks/pipeline_improvement_1/phase3a_mining.ipynb, the Kaggle
notebook for Phase 3a of docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md: restore the 49 GSR-test clips to
`pool` (SoccerNet-Tracking test, converted), cache the main detector on every split, replay Phase 1
with heatmap peaks, mine the specialist's failures, test multi-peak selection, check split integrity.

Disk: no frames are copied to /kaggle/working (converted clips symlink their img1/ to the input;
zips, if any, are extracted to /tmp). Outputs are caches + jsonl, a few hundred MB.

    python scripts/pipeline_improvement_1/phase3/build_phase3a_notebook.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4"))
import build_phase4_notebook as b4  # noqa: E402

b4.KIT_FILES = [
    "scripts/ball_temporal/gsr_io.py",
    "scripts/ball_temporal/cache_main_detections.py",
    "scripts/ball_temporal/heatmap_model.py",
    "scripts/ball_temporal/build_ball_temporal_dataset.py",
    "scripts/ball_eval/eval_ball_temporal.py",
    "gsr_pipeline/ball_gate.py",
    "scripts/pipeline_improvement_1/phase1/phase1_sweep.py",
    "scripts/pipeline_improvement_1/phase0/phase0_report.py",
    "scripts/pipeline_improvement_1/phase3/convert_sntracking.py",
    "scripts/pipeline_improvement_1/phase3/phase3a_mine.py",
]

b4.INTRO = """
# Pipeline Improvement Plan 1 -- Phase 3a: hard mining and multi-peak selection

1. Converts SoccerNet-Tracking `test` (= the 49 GSR `test` clips c5c trained on) to the GSR layout, so
   `pool` is 82 clips as in c5c's training (the Kaggle GSR copy has train/valid only).
2. Caches the main detector on train / pool / select / test.
3. Replays the live system (Phase 1) on every split, keeping c5c's top-5 heatmap peaks per call.
4. Mines the specialist's own failures from train + pool (hard positives B3/B5/B6, false c5c peaks,
   main-detector false balls) into `mined.csv`, for the Phase 3b dataset.
5. Multi-peak selection (highest peak inside the Phase 1 gate radius), chosen on `select`, frozen on `test`.
6. Split integrity with the new clips (same game / overlapping clip / near-duplicate frames).

**Settings:** Accelerator **GPU T4 x2**, Internet **on** (pip installs ultralytics>=8.4).

**Inputs:** soccernetgsr, soccernettracking, phase0-models.

**Disk:** nothing large is written to `/kaggle/working` (clips are symlinked); outputs are a few hundred MB.

**Time:** convert ~2 min, cache ~30-45 min (2 GPUs), replay ~20-30 min (2 GPUs), multi-peak ~15 min,
report + integrity ~10 min. `SMOKE = True` for a ~10-minute dry run (1 sequence per split, 150 frames).
"""

b4.SETTINGS = '''
# Settings. Input paths confirmed by the user.
import os, shutil, subprocess, sys
from pathlib import Path

SMOKE = False

GSR_INPUT   = "/kaggle/input/datasets/sabihalii/soccernetgsr"
TRACK_INPUT = "/kaggle/input/datasets/sabihalii/soccernettracking"
MODELS_DIR  = "/kaggle/input/datasets/sabihalii/phase0-models"

KIT  = "__KIT__"
WORK = Path("/kaggle/working/" + ("phase3a_smoke" if SMOKE else "phase3a"))
SNT  = Path("/kaggle/working/sntracking_gsr")          # labels + img1 symlinks only
SRC  = f"{GSR_INPUT},{SNT}"
CACHE = WORK / "main_cache"
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


def PARALLEL(cmds, logs):
    """One process per GPU; prints each log's last line every minute and fails if any process fails."""
    import time
    procs = [subprocess.Popen([str(c) for c in cmd], stdout=open(log, "w"), stderr=subprocess.STDOUT)
             for cmd, log in zip(cmds, logs)]
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        for log in logs:
            lines = Path(log).read_text().splitlines()
            print(Path(log).name, "|", lines[-1] if lines else "(no output yet)", flush=True)
    for log, p in zip(logs, procs):
        if p.returncode:
            print(Path(log).read_text()[-3000:])
    assert not any(p.returncode for p in procs), [p.returncode for p in procs]


def SHOW(path):
    from IPython.display import Markdown, display
    p = Path(path)
    display(Markdown(p.read_text(encoding="utf-8")) if p.exists() else Markdown(f"*{p} not written yet*"))


def DISK():
    u = shutil.disk_usage("/kaggle/working")
    print(f"/kaggle/working: {u.used / 1e9:.1f} GB used, {u.free / 1e9:.1f} GB free")


print("main:", MAIN, "| c5c:", C5C, "| work:", WORK, "| smoke:", SMOKE)
DISK()
'''

b4.STAGE_CELLS = [
    ("## 1. Environment",
     "Installs `ultralytics>=8.4` (YOLO26) and lists the GPUs.",
     '''STREAM([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4"])
import torch
NGPU = max(torch.cuda.device_count(), 1)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'''),
    ("## 2. SoccerNet-Tracking test -> GSR layout (pool 33 -> 82)",
     "Labels only; each clip's `img1/` is a symlink into the input. Expect train 57 / pool 82 / select 10 / test 15.",
     '''STREAM([sys.executable, "-u", f"{KIT}/scripts/pipeline_improvement_1/phase3/convert_sntracking.py",
        "--src", TRACK_INPUT, "--out", SNT, "--extract_dir", "/tmp/sntracking"])
sys.path.insert(0, f"{KIT}/scripts/ball_temporal")
from gsr_io import discover
REFS = discover(SRC)
print({s: sum(r.split == s for r in REFS) for s in ("train", "pool", "select", "test")},
      "| pool from tracking:", sum(r.name.startswith("SNMOT") for r in REFS))
SMOKE_SEQS = [next(r.name for r in REFS if r.split == s) for s in ("train", "pool", "select", "test")] + \\
             [next(r.name for r in REFS if r.name.startswith("SNMOT"))]
DISK()'''),
    ("## 3. Main-detector cache (all splits)",
     "Ball candidates (top 5, conf >= 0.05) per frame, YOLO26s at imgsz 1280, FP16, one process per GPU. Resumable.",
     '''base = [sys.executable, "-u", f"{KIT}/scripts/ball_temporal/cache_main_detections.py", "run", "--src", SRC,
        "--out", CACHE, "--weights", MAIN, "--splits", "train", "pool", "select", "test", "--half"]
if SMOKE:
    base += ["--seqs", *SMOKE_SEQS, "--max_frames", "150"]
PARALLEL([base + ["--device", str(i), "--shard", str(i), "--nshards", str(NGPU)] for i in range(NGPU)],
         [WORK / f"cache_gpu{i}.log" for i in range(NGPU)])
print({s: len(list((CACHE / s).glob("*.json"))) for s in ("train", "pool", "select", "test")})
DISK()'''),
    ("## 4. Phase 1 replay with heatmap peaks (all splits)",
     "`phase3a_mine.py replay`: per sequence, identical to the live system, plus c5c's top-5 peaks and the main candidates.",
     '''base = [sys.executable, "-u", f"{KIT}/scripts/pipeline_improvement_1/phase3/phase3a_mine.py", "replay",
        "--src", SRC, "--cache", CACHE, "--ckpt", C5C, "--out", WORK]
if SMOKE:
    base += ["--max_seqs", "1", "--max_frames", "150"]
PARALLEL([base + ["--device", f"cuda:{i}", "--shard", str(i), "--nshards", str(NGPU)] for i in range(NGPU)],
         [WORK / f"replay_gpu{i}.log" for i in range(NGPU)])'''),
    ("## 5. Multi-peak selection: chosen on select, frozen on test",
     "Grid over (thr_lo, hi); kill criterion as Phase 1 (+0.5 pp recall, <= 1 pp precision loss).",
     '''STREAM([sys.executable, "-u", f"{KIT}/scripts/pipeline_improvement_1/phase3/phase3a_mine.py", "multipeak",
        "--src", SRC, "--cache", CACHE, "--ckpt", C5C, "--out", WORK]
       + (["--max_seqs", "1", "--max_frames", "150"] if SMOKE else []))'''),
    ("## 6. Report and mined set",
     "Per-split Phase 1 numbers and buckets (pool split into GSR and tracking clips), `mined.csv`, multi-peak verdict.",
     '''STREAM([sys.executable, "-u", f"{KIT}/scripts/pipeline_improvement_1/phase3/phase3a_mine.py", "report",
        "--src", SRC, "--out", WORK])
SHOW(WORK / "report_phase3a.md")'''),
    ("## 7. Split integrity with the new clips",
     "`phase0_report.py` on the select/test replay: same game + overlapping clip time across splits, and near-duplicate "
     "frames (dHash, Hamming <= 4, every 25th frame) between {select, test} and {train, pool}.",
     '''STREAM([sys.executable, "-u", f"{KIT}/scripts/pipeline_improvement_1/phase0/phase0_report.py",
        "--src", SRC, "--diag", WORK, "--out", WORK / "integrity", "--splits", "select", "test",
        "--frames_per_hash", "100" if SMOKE else "25"])
txt = (WORK / "integrity" / "report_phase0.md").read_text()
print(txt[txt.find("## Split integrity"):])'''),
    ("## 8. Deliverables",
     "Packs the work folder (reports, mined set, replay frames, caches) into `/kaggle/working/phase3a_deliverables.zip`.",
     '''import hashlib, json, zipfile
out = Path("/kaggle/working/phase3a_deliverables.zip")
manifest = {}
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(p for p in WORK.rglob("*") if p.is_file()):
        rel = f.relative_to(WORK).as_posix()
        zf.write(f, rel)
        manifest[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    zf.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
print(f"{out}: {len(manifest)} files, {out.stat().st_size / 1e6:.1f} MB")
DISK()'''),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=b4.ROOT / "notebooks" / "pipeline_improvement_1" / "phase3a_mining.ipynb")
    b4.build(ap.parse_args().out)


if __name__ == "__main__":
    main()
