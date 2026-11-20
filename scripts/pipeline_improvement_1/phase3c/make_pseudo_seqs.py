"""
make_pseudo_seqs.py -- Phase 3c (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md): each pseudo-labelled SoccerNet window as a
GSR-format sequence, so every GSR tool (gsr_io.discover, the YOLO mirror, c5c's main-detector cache and dataset
builder) reads it unchanged.

<out>/test/PSL-<nnnnn>/img1/*.jpg     hard links to <window>/frames/*.jpg (no extra disk)
<out>/test/PSL-<nnnnn>/Labels-GameState.json
    ball (category 4): frames whose ball-teacher score >= --ball_thr (default 0.4, decided 2026-11-19), box size from
                       the window's yolo/*.txt (T2's box, else 12 px)
    athletes (category 1): athlete-teacher boxes at >= the confidence chosen on select (athlete_teacher.py choose)
    info.ignore_frames: frames the ball teacher labelled at the evaluated floor (0.06) but below --ball_thr; the
                       ball is probably there, so these frames are neither positives nor safe negatives
<out>/index.json                       {window dir: PSL name}, stable across reruns (new windows get new numbers)

The `test` folder name puts the sequences in gsr_io's `pool` split (GSR's own test folder maps there): clips the
live detector never trained on, like GSR's test clips, so c5c's builder samples their hard frames (it does not in
`train`, decision A). For the YOLO mirror, pool is training data like train. Windows without athletes.json are skipped
(rerun after athlete_teacher.py label).

    python scripts/pipeline_improvement_1/phase3c/make_pseudo_seqs.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PSEUDO = ROOT / "data" / "pseudo_labels" / "phase3b_teacher"
OUT = ROOT / "data" / "pseudo_gsr"
EVAL = ROOT / "data" / "outputs" / "pipeline_improvement_1" / "phase3c" / "eval" / "athlete_teacher_a1_s1920_ft.json"
W, H = 1920, 1080


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pseudo", type=Path, default=PSEUDO)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--ball_thr", type=float, default=0.4)
    ap.add_argument("--athlete_eval", type=Path, default=EVAL, help="athlete_teacher.py choose output (its conf)")
    ap.add_argument("--athlete_conf", type=float, default=None, help="override the chosen athlete confidence")
    ap.add_argument("--keep_names", action="store_true",
                    help="write <out>/<window key> (e.g. train/SNGS-060) instead of <out>/test/PSL-n: relabel_gsr.py")
    a = ap.parse_args()
    aconf = a.athlete_conf if a.athlete_conf is not None else json.loads(a.athlete_eval.read_text())["conf"]
    idx_f = a.out / "index.json"
    index = json.loads(idx_f.read_text()) if idx_f.exists() else {}
    wins = sorted(p.parent for p in a.pseudo.glob("*/*/labels.json"))
    made = skipped = 0
    tot = {"frames": 0, "ball": 0, "ignore": 0, "athletes": 0}
    for w in wins:
        key = w.relative_to(a.pseudo).as_posix()
        if not (w / "athletes.json").exists():
            skipped += 1
            continue
        if a.keep_names:
            name, seq = w.name, a.out / key
        else:
            name = index.setdefault(key, f"PSL-{len(index) + 1:05d}")
            seq = a.out / "test" / name
        lab_f = seq / "Labels-GameState.json"
        if lab_f.exists():
            continue
        frames = json.loads((w / "labels.json").read_text())["frames"]
        aths = json.loads((w / "athletes.json").read_text())
        img = seq / "img1"
        img.mkdir(parents=True, exist_ok=True)
        images, anns, ignore = [], [], []
        for fn in sorted(frames):
            f = int(Path(fn).stem)
            dst = img / fn
            if not dst.exists():
                try:
                    os.link(w / "frames" / fn, dst)
                except OSError:
                    shutil.copy2(w / "frames" / fn, dst)
            images.append({"image_id": f, "file_name": fn, "width": W, "height": H})
            p = frames[fn]
            if p is not None and p[2] >= a.ball_thr:
                bw = bh = 12.0
                y = (w / "yolo" / f"{Path(fn).stem}.txt").read_text().split()
                if len(y) == 5:
                    bw, bh = float(y[3]) * W, float(y[4]) * H
                anns.append({"image_id": f, "category_id": 4, "track_id": 1, "score": p[2],
                             "bbox_image": {"x_center": p[0], "y_center": p[1], "w": bw, "h": bh}})
                tot["ball"] += 1
            elif p is not None:
                ignore.append(f)
            for k, (cx, cy, bw, bh, c) in enumerate(aths.get(fn, [])):
                if c >= aconf:
                    anns.append({"image_id": f, "category_id": 1, "track_id": 1000 + k, "score": c,
                                 "bbox_image": {"x_center": cx, "y_center": cy, "w": bw, "h": bh}})
                    tot["athletes"] += 1
        tot["frames"] += len(images)
        tot["ignore"] += len(ignore)
        doc = {"info": {"name": name, "im_dir": "img1", "frame_rate": 25, "source": key, "ignore_frames": ignore,
                        "ball_thr": a.ball_thr, "athlete_conf": aconf},
               "images": images, "annotations": anns}
        tmp = lab_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc))
        tmp.replace(lab_f)
        made += 1
    idx_f.parent.mkdir(parents=True, exist_ok=True)
    idx_f.write_text(json.dumps(index, indent=0))
    print(f"{made} sequences written, {skipped} windows without athletes.json yet; this run: {tot}")


if __name__ == "__main__":
    main()
