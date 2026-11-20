"""
relabel_gsr.py -- Phase 3c (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md), decided 2026-11-20: the students train on
teacher labels only. GSR's own train and pool labels miss balls and athletes, so GSR's train and pool sequences are
relabelled by both teachers, exactly as the SoccerNet windows (ball >= 0.4, 0.06..0.4 -> info.ignore_frames;
athletes at the athlete teacher's chosen confidence). select and test keep GSR's labels: they are the ground truth
every score is measured on.

  balls     <work>/<gsr folder>/<seq>/frames (hard links to img1) + labels.json, yolo/ (the Phase 3b ball teacher)
  athletes  athlete_teacher.py label --pseudo <work>             -> athletes.json
  write     make_pseudo_seqs.py --keep_names --out <out>         -> <out>/<gsr folder>/<seq>/{img1,Labels-GameState.json}
            and select / test as hard links to GSR (labels included)

<out> mirrors data/gsr's folders and sequence names, so gsr_io.discover gives every sequence the same split, and the
main-detector cache (keyed by split and name) is shared with data/gsr. Resumable per sequence.

    python scripts/pipeline_improvement_1/phase3c/relabel_gsr.py all
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "phase3b"))
sys.path.insert(0, str(ROOT / "scripts" / "ball_temporal"))
from gsr_io import discover  # noqa: E402

WORK = ROOT / "data" / "pseudo_labels" / "gsr_teacher"
OUT = ROOT / "data" / "gsr_teacher"
PY = sys.executable


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def folder(ref) -> str:
    return Path(ref.container).parent.name          # train | valid | test (data/gsr's folder)


def balls(a) -> None:
    import pseudo_label as pl
    refs = [r for r in discover(a.src) if r.split in ("train", "pool")]
    for i, r in enumerate(refs, 1):
        w = a.work / folder(r) / r.name
        if (w / "labels.json").exists():
            continue
        for p in sorted((Path(r.container) / "img1").glob("*.jpg")):
            link(p, w / "frames" / p.name)
        print(f"[{i}/{len(refs)}] {r.split} {r.name}", flush=True)
        pl.label_dir(w / "frames", w, a.device, a.batch)


def write(a) -> None:
    subprocess.run([PY, "-u", str(HERE / "make_pseudo_seqs.py"), "--pseudo", str(a.work), "--out", str(a.out),
                    "--keep_names"], check=True)
    n = 0
    for r in discover(a.src):
        if r.split not in ("select", "test"):
            continue
        d = a.out / folder(r) / r.name
        for p in sorted((Path(r.container) / "img1").glob("*.jpg")):
            link(p, d / "img1" / p.name)
        link(Path(r.container) / "Labels-GameState.json", d / "Labels-GameState.json")
        n += 1
    got = {r.name: r.split for r in discover(a.out)}
    want = {r.name: r.split for r in discover(a.src)}
    if got != want:
        raise SystemExit(f"split mismatch vs {a.src}: {sorted(set(got.items()) ^ set(want.items()))[:10]}")
    print(f"{n} select/test sequences linked with GSR labels; {len(got)} sequences, splits identical to {a.src}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["balls", "athletes", "write", "all"])
    ap.add_argument("--src", default=str(ROOT / "data" / "gsr"))
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    if a.step in ("balls", "all"):
        balls(a)
    if a.step in ("athletes", "all"):
        subprocess.run([PY, "-u", str(HERE / "athlete_teacher.py"), "label", "--pseudo", str(a.work)], check=True)
    if a.step in ("write", "all"):
        write(a)


if __name__ == "__main__":
    main()
