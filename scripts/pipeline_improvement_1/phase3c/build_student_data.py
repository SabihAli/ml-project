"""
build_student_data.py -- Phase 3c (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md): the 2-class (athlete 0, ball 1) YOLO
mirror for the athlete teacher and the detector student, from GSR and the pseudo-labelled windows.

Every sequence gsr_io.discover finds under --src (comma-separated roots; the pseudo windows are GSR-format
sequences made by make_pseudo_seqs.py, split `train`) is hard-linked into <data>/images/<role>/<seq>_<frame:06d>.jpg
with labels/<role>/<same>.txt. Every frame is linked (the 5-channel reader needs the neighbours); the lists name
every --gsr_stride-th GSR frame and every --pseudo_stride-th pseudo frame, never a frame in the sequence's
info.ignore_frames (a ball the teacher saw with too little confidence: neither a positive nor a safe negative).

Roles as every Phase 3b experiment: train + pool (+ pseudo) -> train, select -> val, test -> test.
Lists: train_gsr.txt (GSR only: the athlete teacher), train_all.txt (GSR + pseudo: the student), val.txt, test.txt;
dataset_gsr.yaml, dataset_all.yaml. Resumable per sequence (a sequence whose label dir is complete is skipped).

    python scripts/pipeline_improvement_1/phase3c/build_student_data.py --src data/gsr
    python scripts/pipeline_improvement_1/phase3c/build_student_data.py --src data/gsr,data/pseudo_gsr
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "ball_temporal"))
from gsr_io import SeqReader, discover, parse_labels  # noqa: E402

DATA = ROOT / "data" / "outputs" / "pipeline_improvement_1" / "phase3c" / "student_data"
ROLE = {"train": "train", "pool": "train", "select": "val", "test": "test"}
NAMES = "names:\n  0: athlete\n  1: ball\n"


def link(src: Path, dst: Path) -> None:
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def rows(objs, cls: int, W: int, H: int) -> list[str]:
    return [f"{cls} {cx / W:.6f} {cy / H:.6f} {max(w, 1.0) / W:.6f} {max(h, 1.0) / H:.6f}" for cx, cy, w, h, *_ in objs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "data" / "gsr"))
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--gsr_stride", type=int, default=3)
    ap.add_argument("--pseudo_stride", type=int, default=5)
    ap.add_argument("--val_stride", type=int, default=10)
    a = ap.parse_args()
    refs = [r for r in discover(a.src) if r.split in ROLE]
    lists = {"train_gsr": [], "train_all": [], "val": [], "test": []}
    stats = {}
    for i, r in enumerate(refs, 1):
        role, pseudo = ROLE[r.split], not r.name.startswith("SNGS")
        rd = SeqReader(r)
        try:
            d = rd.labels()
        finally:
            rd.close()
        im_dir, files, balls, persons, _, (W, H) = parse_labels(d)
        ignore = set(d.get("info", {}).get("ignore_frames", []))
        img_dir, lab_dir = a.data / "images" / role, a.data / "labels" / role
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        stride = a.pseudo_stride if pseudo else a.gsr_stride if role == "train" else a.val_stride if role == "val" else 1
        done = all((lab_dir / f"{r.name}_{f:06d}.txt").exists() for f in files)
        for k, f in enumerate(sorted(files)):
            stem = f"{r.name}_{f:06d}"
            if not done:
                link(Path(r.container) / im_dir / files[f], img_dir / f"{stem}.jpg")
                txt = rows(persons.get(f, []), 0, W, H) + rows(balls.get(f, []), 1, W, H)
                (lab_dir / f"{stem}.txt").write_text("\n".join(txt) + ("\n" if txt else ""))
            if k % stride or f in ignore:
                continue
            p = str((img_dir / f"{stem}.jpg").resolve())
            key = "pseudo" if pseudo else r.split
            s = stats.setdefault(key, [0, 0])
            s[0] += 1
            s[1] += bool(balls.get(f))
            if role == "train":
                lists["train_all"].append(p)
                if not pseudo:
                    lists["train_gsr"].append(p)
            else:
                lists[role].append(p)
        if i % 50 == 0 or i == len(refs):
            print(f"  [{i}/{len(refs)}] {r.name} ({r.split} -> {role}){' skipped, done' if done else ''}", flush=True)
    for k, v in lists.items():
        (a.data / f"{k}.txt").write_text("\n".join(v) + "\n")
    base = f"path: {a.data.resolve().as_posix()}\nval: val.txt\ntest: test.txt\nnc: 2\n{NAMES}"
    (a.data / "dataset_gsr.yaml").write_text(f"train: train_gsr.txt\n{base}")
    (a.data / "dataset_all.yaml").write_text(f"train: train_all.txt\n{base}")
    (a.data / "stats.json").write_text(json.dumps(stats, indent=1))
    for k, (n, nb) in sorted(stats.items()):
        print(f"{k}: {n} frames listed, {nb} with a ball")
    print({k: len(v) for k, v in lists.items()})


if __name__ == "__main__":
    main()
