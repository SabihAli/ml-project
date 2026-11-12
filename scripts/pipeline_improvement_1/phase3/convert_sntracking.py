"""
convert_sntracking.py -- SoccerNet-Tracking clips (MOT format: gameinfo.ini, gt/gt.txt, img1/) to the
GSR layout gsr_io.discover() reads (Labels-GameState.json + img1/), for Phase 3 of
docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md.

Why: tracking `test` (49 clips, games 7/8/11) is the GSR `test` split that c5c's `pool` was trained
with (82 clips); the Kaggle GSR dataset has only train/valid, so `pool` there is 33 clips. Writing
these under <out>/test/ makes discover() put them in `pool` (src split gsrtest).

Disk: labels only (~1 MB per clip); img1/ is a symlink to the input (no frames copied), so the
20 GB Kaggle working limit is not touched. Input may be extracted folders or the split zip; a zip
is extracted to --extract_dir (outside /kaggle/working).

Labels kept: ball (category 4) and persons (category 1) as bbox_image centre/size. No pitch lines.

    python scripts/pipeline_improvement_1/phase3/convert_sntracking.py --src <tracking root> --out <dir>
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import zipfile
from pathlib import Path

BALL_CAT, PERSON_CAT = 4, 1


def _clips(src: Path, extract_dir: Path) -> list[Path]:
    clips = sorted({p.parent for p in src.rglob("gameinfo.ini") if (p.parent / "img1").is_dir()})
    if clips:
        return clips
    for zp in sorted(src.rglob("*.zip")):
        dest = extract_dir / zp.stem
        if not dest.exists():
            print(f"extracting {zp} -> {dest}", flush=True)
            with zipfile.ZipFile(zp) as z:
                z.extractall(dest)
        clips += sorted({p.parent for p in dest.rglob("gameinfo.ini") if (p.parent / "img1").is_dir()})
    return clips


def convert(clip: Path, out_root: Path) -> tuple[str, int]:
    cfg = configparser.ConfigParser()
    cfg.read_string(clip.joinpath("gameinfo.ini").read_text())
    s = cfg[cfg.sections()[0]]
    name = s["name"].strip()
    kind = {k.split("_")[1]: v.split(";")[0].strip() for k, v in s.items() if k.startswith("trackletid")}
    seq = configparser.ConfigParser()
    seq.read_string(clip.joinpath("seqinfo.ini").read_text())
    si = seq[seq.sections()[0]]
    W, H = int(si.get("imWidth", 1920)), int(si.get("imHeight", 1080))
    ext = si.get("imExt", ".jpg")
    files = sorted(p.name for p in (clip / "img1").iterdir() if p.suffix == ext)
    images = [{"image_id": int(Path(f).stem), "file_name": f, "width": W, "height": H} for f in files]
    anns, n_ball = [], 0
    for line in clip.joinpath("gt", "gt.txt").read_text().split():
        f, tid, x, y, w, h = line.split(",")[:6]
        k = kind.get(tid, "")
        if k == "ball":
            cat = BALL_CAT
            n_ball += 1
        elif k:
            cat = PERSON_CAT
        else:
            continue
        x, y, w, h = map(float, (x, y, w, h))
        anns.append({"image_id": int(f), "category_id": cat, "track_id": int(tid),
                     "bbox_image": {"x_center": x + w / 2, "y_center": y + h / 2, "w": w, "h": h}})
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)
    info = {"name": name, "im_dir": "img1", "game_id": s.get("gameID", "").strip(),
            "clip_start": s.get("clipStart", "").strip(), "clip_stop": s.get("clipStop", "").strip(),
            "game_time_start": s.get("gameTimeStart", "").strip(), "source": "SoccerNet-Tracking"}
    (d / "Labels-GameState.json").write_text(json.dumps({"info": info, "images": images, "annotations": anns}))
    link = d / "img1"
    if not link.exists():
        try:
            os.symlink((clip / "img1").resolve(), link, target_is_directory=True)
        except OSError:                     # Windows without symlink rights (local smoke tests): copy
            shutil.copytree(clip / "img1", link)
    return name, n_ball


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="tracking dataset root (folders or zips)")
    ap.add_argument("--out", type=Path, required=True, help="GSR-layout root; clips go to <out>/test/")
    ap.add_argument("--extract_dir", type=Path, default=Path("/tmp/sntracking"))
    ap.add_argument("--split_dir", default="test", help="only clips under a folder of this name")
    a = ap.parse_args()
    clips = [c for c in _clips(a.src, a.extract_dir) if a.split_dir in c.parts]
    tot = 0
    for c in clips:
        _, n = convert(c, a.out / "test")
        tot += n
    print(f"converted {len(clips)} clips, {tot} ball boxes -> {a.out / 'test'}")


if __name__ == "__main__":
    main()
