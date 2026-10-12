"""
build_ball_crop_dataset.py -- 640x640 ball-specialist crop dataset from SoccerNet-GSR.

Implements docs/path_b/plans/BALL_TEMPORAL_PLAN.md section 4. Standalone (no gsr_pipeline import),
so it runs on Kaggle as is. Needs numpy, opencv-python, pyyaml.

  * Every 3rd frame per sequence (frames 4, 7, 10, ...).
  * Positive crop: centred on the GT ball at t-3 (+-48 px jitter), cut from frame t, shift-
    clamped into the frame. Labels are the balls at t. If t-3 has no ball, the most recent
    ball in t-4 .. t-15 is used; none -> no positive for that frame.
  * Negatives (~25% of crops): hard (centred on a person box or a pitch-line point) and
    random, from ball-labelled frames only, never overlapping a ball.
  * A ball cut by the crop border is kept if >=50% of its box and its centre are inside;
    a ball cut worse than that drops the crop.
  * Splits by sequence: train = GSR train; test = the 15 held-out valid sequences;
    val = every other valid sequence.

Source: --src is searched recursively for sequence folders (Labels-GameState.json + img1/)
or, failing that, for the split zips (train.zip / valid.zip), read without extracting.

    python scripts/build_ball_crop_dataset.py --src /kaggle/input/datasets/sabihalii/soccernetgsr \
        --out /kaggle/working/ball_crops --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path, PurePosixPath

import numpy as np

CROP = 640
STEP = 3                 # keep every 3rd frame; also the gap to the centring ball
LOOKBACK = 15            # = max_miss_frames
JITTER = 48
NEG_PER_POS = 1 / 3      # negatives = 25% of all crops
HARD_SHARE = 0.75        # of negatives: 15% hard, 5% random
MIN_VISIBLE = 0.5
JPEG_QUALITY = 95
TEST_SEQS = {21, 22, 23, 26, 30, 33, 36, 40, 44, 47, 50, 54, 80, 86, 90}
PERSON_CATS = {1, 2, 3, 7}
BALL_CAT = 4
MANIFEST_FIELDS = ["split", "seq", "frame", "kind", "centre_source_frame", "jitter_x", "jitter_y",
                   "x0", "y0", "n_balls", "ball_px", "ball_cx_in_crop", "ball_cy_in_crop",
                   "clamped", "file"]


# --------------------------------------------------------------------------- sources

@dataclass(frozen=True)
class SeqRef:
    name: str          # SNGS-060
    src_split: str     # train | valid
    kind: str          # dir | zip
    container: str     # sequence dir, or zip path
    prefix: str        # inside-zip prefix of the sequence folder ("" for dir)

    @property
    def number(self) -> int:
        return int(self.name.split("-")[-1])

    @property
    def split(self) -> str:
        if self.src_split == "train":
            return "train"
        return "test" if self.number in TEST_SEQS else "val"


def _split_of(parts) -> str | None:
    low = [p.lower() for p in parts]
    if any("train" in p for p in low):
        return "train"
    if any("valid" in p for p in low):
        return "valid"
    return None


def discover(src: Path) -> list[SeqRef]:
    """Sequence folders under src, else sequences inside zips under src. One entry per name."""
    found: dict[str, SeqRef] = {}
    for lab in sorted(src.rglob("Labels-GameState.json")):
        seq = lab.parent
        split = _split_of(seq.relative_to(src).parts)
        if split and seq.name not in found and (seq / "img1").is_dir():
            found[seq.name] = SeqRef(seq.name, split, "dir", str(seq), "")
    if not found:
        for zp in sorted(src.rglob("*.zip")):
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    if not n.endswith("Labels-GameState.json"):
                        continue
                    pre = str(PurePosixPath(n).parent)
                    name = PurePosixPath(pre).name
                    split = _split_of(PurePosixPath(pre).parts) or _split_of([zp.name])
                    if split and name.startswith("SNGS") and name not in found:
                        found[name] = SeqRef(name, split, "zip", str(zp), pre)
    return sorted(found.values(), key=lambda s: s.number)


class SeqReader:
    """Reads one sequence's labels and frames from a folder or a zip (one per process)."""

    def __init__(self, ref: SeqRef):
        self.ref = ref
        self._zip = zipfile.ZipFile(ref.container) if ref.kind == "zip" else None

    def read(self, rel: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(f"{self.ref.prefix}/{rel}")
        return (Path(self.ref.container) / rel).read_bytes()

    def labels(self) -> dict:
        return json.loads(self.read("Labels-GameState.json"))

    def frame(self, im_dir: str, file_name: str):
        import cv2
        buf = np.frombuffer(self.read(f"{im_dir}/{file_name}"), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"{self.ref.name}: cannot decode {im_dir}/{file_name}")
        return img

    def close(self):
        if self._zip is not None:
            self._zip.close()


def parse_labels(d: dict):
    """-> im_dir, {frame: file_name}, {frame: [(cx,cy,w,h,track_id)]}, persons, line points, (W,H)."""
    info = d.get("info", {})
    im_dir = info.get("im_dir", "img1")
    W, H = 1920, 1080
    files, id2f = {}, {}
    for im in d["images"]:
        f = int(Path(im["file_name"]).stem)
        files[f] = im["file_name"]
        id2f[im["image_id"]] = f
        W, H = im.get("width", W), im.get("height", H)
    balls, persons, lines = {}, {}, {}
    for a in d["annotations"]:
        f = id2f.get(a["image_id"])
        if f is None:
            continue
        cat = a.get("category_id")
        if cat == BALL_CAT or cat in PERSON_CATS:
            b = a["bbox_image"]
            row = (float(b["x_center"]), float(b["y_center"]), float(b["w"]), float(b["h"]), a.get("track_id"))
            (balls if cat == BALL_CAT else persons).setdefault(f, []).append(row)
        elif a.get("supercategory") == "pitch":
            pts = [(p["x"] * W, p["y"] * H) for seg in a.get("lines", {}).values() for p in seg]
            lines[f] = [(x, y) for x, y in pts if 0 <= x < W and 0 <= y < H]
    return im_dir, files, balls, persons, lines, (W, H)


# --------------------------------------------------------------------------- geometry

def clamp_window(cx: float, cy: float, W: int, H: int) -> tuple[int, int, bool]:
    """Top-left of a CROP x CROP window centred at (cx, cy), shifted (never padded) into the frame."""
    x0 = int(round(cx - CROP / 2))
    y0 = int(round(cy - CROP / 2))
    cx0 = min(max(x0, 0), max(W - CROP, 0))
    cy0 = min(max(y0, 0), max(H - CROP, 0))
    return cx0, cy0, (cx0, cy0) != (x0, y0)


def cut(img, x0: int, y0: int):
    """CROP x CROP slice. Pads with 114 only if the frame is smaller than CROP (never on GSR)."""
    c = img[y0:y0 + CROP, x0:x0 + CROP]
    if c.shape[0] != CROP or c.shape[1] != CROP:
        pad = np.full((CROP, CROP, 3), 114, np.uint8)
        pad[:c.shape[0], :c.shape[1]] = c
        print(f"WARNING: frame smaller than {CROP}px, letterbox-padded", file=sys.stderr)
        return pad
    return c


def balls_in_window(balls, x0, y0):
    """-> (labels, dropped). labels are normalised YOLO boxes; dropped=True if a ball is cut ambiguously."""
    labels = []
    for cx, cy, w, h, _ in balls:
        l, t, r, b = cx - w / 2 - x0, cy - h / 2 - y0, cx + w / 2 - x0, cy + h / 2 - y0
        il, it, ir, ib = max(l, 0), max(t, 0), min(r, CROP), min(b, CROP)
        inter = max(ir - il, 0) * max(ib - it, 0)
        if inter <= 0:
            continue
        centre_in = 0 <= cx - x0 < CROP and 0 <= cy - y0 < CROP
        if inter / max(w * h, 1e-6) < MIN_VISIBLE or not centre_in:
            return [], True
        labels.append(((il + ir) / 2 / CROP, (it + ib) / 2 / CROP, (ir - il) / CROP, (ib - it) / CROP))
    return labels, False


def overlaps_any_ball(balls, x0, y0, margin=8):
    for cx, cy, w, h, _ in balls:
        if (cx + w / 2 + margin > x0 and cx - w / 2 - margin < x0 + CROP and
                cy + h / 2 + margin > y0 and cy - h / 2 - margin < y0 + CROP):
            return True
    return False


# --------------------------------------------------------------------------- per sequence

def process_sequence(job) -> list[dict]:
    import cv2
    ref, out, seed = job
    reader = SeqReader(ref)
    try:
        im_dir, files, balls, persons, lines, (W, H) = parse_labels(reader.labels())
        split = ref.split
        img_dir = Path(out) / "images" / split
        lab_dir = Path(out) / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for t in range(1 + STEP, max(files) + 1, STEP):
            bt = balls.get(t)
            if not bt or t not in files:
                continue          # frames without a ball label give no crops at all (plan 4.3)
            rng = np.random.default_rng([seed, ref.number, t])
            jobs = []             # (kind, x0, y0, labels, meta)

            # positive: centre on the most recent ball in t-3 .. t-15
            src = next((g for g in range(STEP, LOOKBACK + 1) if balls.get(t - g)), None)
            if src is not None:
                prev = balls[t - src]
                track_ids = {b[4] for b in bt}
                anchor = next((b for b in prev if b[4] in track_ids), prev[0])
                jx, jy = rng.uniform(-JITTER, JITTER, 2)
                x0, y0, clamped = clamp_window(anchor[0] + jx, anchor[1] + jy, W, H)
                labels, dropped = balls_in_window(bt, x0, y0)
                if not dropped:
                    jobs.append(("p" if labels else "e", x0, y0, labels,
                                 dict(centre_source_frame=t - src, jitter_x=round(jx, 1),
                                      jitter_y=round(jy, 1), clamped=int(clamped))))

            # negative, ~1 per 3 positives
            if rng.random() < NEG_PER_POS:
                cands = [(p[0], p[1]) for p in persons.get(t, [])] + lines.get(t, [])
                hard = rng.random() < HARD_SHARE and bool(cands)
                for _ in range(20):
                    if hard:
                        cx, cy = cands[rng.integers(len(cands))]
                        cx, cy = cx + rng.uniform(-64, 64), cy + rng.uniform(-64, 64)
                    else:
                        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
                    x0, y0, clamped = clamp_window(cx, cy, W, H)
                    if not overlaps_any_ball(bt, x0, y0):
                        jobs.append(("nh" if hard else "nr", x0, y0, [],
                                     dict(centre_source_frame="", jitter_x="", jitter_y="",
                                          clamped=int(clamped))))
                        break

            if not jobs:
                continue
            img = reader.frame(im_dir, files[t])
            for kind, x0, y0, labels, meta in jobs:
                stem = f"{ref.name}_{t:06d}_{kind}"
                ok, enc = cv2.imencode(".jpg", cut(img, x0, y0), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    raise RuntimeError(f"jpeg encode failed for {stem}")
                (img_dir / f"{stem}.jpg").write_bytes(enc.tobytes())
                (lab_dir / f"{stem}.txt").write_text(
                    "".join(f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}\n" for a, b, c, d in labels))
                first = labels[0] if labels else None
                rows.append(dict(split=split, seq=ref.name, frame=t, kind=kind, x0=x0, y0=y0,
                                 n_balls=len(labels),
                                 ball_px=round(max(first[2], first[3]) * CROP, 1) if first else "",
                                 ball_cx_in_crop=round(first[0] * CROP, 1) if first else "",
                                 ball_cy_in_crop=round(first[1] * CROP, 1) if first else "",
                                 file=f"images/{split}/{stem}.jpg", **meta))
        return rows
    finally:
        reader.close()


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="output dir (required unless --list)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seqs", nargs="*", help="only these sequence names (smoke tests)")
    ap.add_argument("--list", action="store_true", help="print the discovered sequences and exit")
    args = ap.parse_args()

    seqs = discover(args.src)
    if args.seqs:
        seqs = [s for s in seqs if s.name in set(args.seqs)]
    by_split = {k: [s.name for s in seqs if s.split == k] for k in ("train", "val", "test")}
    print(f"found {len(seqs)} sequences ({seqs[0].kind if seqs else '-'}): "
          + ", ".join(f"{k} {len(v)}" for k, v in by_split.items()))
    if args.list or not seqs:
        for s in seqs:
            print(f"  {s.name:9s} {s.split:5s} {s.kind} {s.container}")
        return
    if args.out is None:
        ap.error("--out is required")
    missing_test = TEST_SEQS - {s.number for s in seqs if s.split == "test"}
    if missing_test and not args.seqs:
        print(f"WARNING: held-out test sequences not found: {sorted(missing_test)}")

    args.out.mkdir(parents=True, exist_ok=True)
    t0, rows = time.time(), []
    with Pool(args.workers) as pool:
        jobs = [(s, str(args.out), args.seed) for s in seqs]
        for i, r in enumerate(pool.imap_unordered(process_sequence, jobs), 1):
            rows.extend(r)
            print(f"[{i}/{len(seqs)}] {r[0]['seq'] if r else '?'}: {len(r)} crops  ({time.time() - t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: (r["split"], r["seq"], r["frame"], r["kind"]))
    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    import yaml
    with open(args.out / "dataset.yaml", "w") as f:
        yaml.safe_dump({"path": str(args.out.resolve()), "train": "images/train", "val": "images/val",
                        "test": "images/test", "names": {0: "ball"}}, f, sort_keys=False)

    print("\nsplit  crops  positive  empty  neg_hard  neg_random")
    for k in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == k]
        c = lambda kind: sum(r["kind"] == kind for r in rs)
        print(f"{k:5s} {len(rs):6d} {c('p'):9d} {c('e'):6d} {c('nh'):9d} {c('nr'):11d}")
    print(f"\nwrote {args.out}/dataset.yaml and manifest.csv in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
