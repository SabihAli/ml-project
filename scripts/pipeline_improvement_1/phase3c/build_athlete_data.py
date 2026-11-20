"""
build_athlete_data.py -- Phase 3c (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md): athlete-only (1 class) copy of the GSR
part of build_student_data.py's mirror, for the athlete teacher (athlete_teacher.py). Ball labels come from the ball
teacher alone, so the athlete teacher does not learn a ball class (decided 2026-11-20).

Images are hard links to the mirror's (no extra disk); labels keep only class-0 lines. Same frames as train_gsr.txt,
val.txt and test.txt.

<out>/images/<split>/*.jpg, <out>/labels/<split>/*.txt, <out>/{train,val,test}.txt, <out>/dataset.yaml (nc 1)

    python scripts/pipeline_improvement_1/phase3c/build_athlete_data.py
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P3C = ROOT / "data" / "outputs" / "pipeline_improvement_1" / "phase3c"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=P3C / "student_data")
    ap.add_argument("--out", type=Path, default=P3C / "athlete_data")
    a = ap.parse_args()
    for lst, name in (("train_gsr.txt", "train.txt"), ("val.txt", "val.txt"), ("test.txt", "test.txt")):
        out_list, n_box = [], 0
        for line in (a.src / lst).read_text().split():
            img = Path(line)
            split = img.parent.name
            dst = a.out / "images" / split / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(img, dst)
                except OSError:
                    shutil.copy2(img, dst)
            lab = a.src / "labels" / split / (img.stem + ".txt")
            rows = [r for r in (lab.read_text().splitlines() if lab.exists() else []) if r.split()[:1] == ["0"]]
            n_box += len(rows)
            ld = a.out / "labels" / split / (img.stem + ".txt")
            ld.parent.mkdir(parents=True, exist_ok=True)
            ld.write_text("\n".join(rows) + ("\n" if rows else ""))
            out_list.append(str(dst))
        (a.out / name).write_text("\n".join(out_list) + "\n")
        print(f"{name}: {len(out_list)} frames, {n_box} athlete boxes", flush=True)
    (a.out / "dataset.yaml").write_text(
        f"path: {a.out.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnc: 1\nnames:\n  0: athlete\n")


if __name__ == "__main__":
    main()
