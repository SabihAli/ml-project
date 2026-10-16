"""
gsr_io.py -- SoccerNet-GSR readers and the sequence splits for the temporal ball specialist
(docs/path_b/plans/BALL_TEMPORAL_PLAN.md §3, §6).

Standalone (numpy, opencv-python only), so the scripts in this folder run on Kaggle as is.

Splits, by sequence (decision A, 2026-10-12):
  train   all GSR `train` sequences. The main detector was trained on them, so their frames give
          easy positives and negatives only, never hard examples.
  pool    `valid` sequences outside test and select, plus every GSR `test` sequence (added
          2026-10-15). The main detector never trained on either: the source of hard examples.
  select  10 `valid` sequences, evenly spaced through the non-test ones: model selection and
          early stopping.
  test    the 15 held-out `valid` sequences: final gates only.

Sources: a directory searched recursively for sequence folders (Labels-GameState.json + img1/),
or failing that the split zips (train.zip / valid.zip), read without extracting.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

TEST_SEQS = frozenset({21, 22, 23, 26, 30, 33, 36, 40, 44, 47, 50, 54, 80, 86, 90})
N_SELECT = 10
BALL_CAT = 4
PERSON_CATS = frozenset({1, 2, 3, 7})


@dataclass(frozen=True)
class SeqRef:
    name: str          # SNGS-060
    src_split: str     # train | valid | gsrtest (the GSR folder it came from)
    kind: str          # dir | zip
    container: str     # sequence dir, or zip path
    prefix: str        # inside-zip prefix of the sequence folder ("" for dir)
    split: str = ""    # train | pool | select | test, filled in by discover()

    @property
    def number(self) -> int:
        return int(self.name.split("-")[-1])


def _gsr_split_of(parts) -> str | None:
    low = [p.lower() for p in parts]
    if any("train" in p for p in low):
        return "train"
    if any("valid" in p for p in low):
        return "valid"
    if any(p == "test" or p.startswith("test.") for p in low):
        return "gsrtest"            # the GSR test split; not our held-out `test` (valid sequences)
    return None


def select_seqs(valid_numbers) -> set[int]:
    """N_SELECT sequences evenly spaced through the sorted non-test valid sequences."""
    rest = sorted(n for n in valid_numbers if n not in TEST_SEQS)
    if len(rest) <= N_SELECT:
        return set(rest)
    return {rest[int((i + 0.5) * len(rest) / N_SELECT)] for i in range(N_SELECT)}


def discover(src: Path) -> list[SeqRef]:
    """Sequence folders under src, else sequences inside zips under src. One entry per name."""
    found: dict[str, SeqRef] = {}
    for lab in sorted(Path(src).rglob("Labels-GameState.json")):
        seq = lab.parent
        gsr = _gsr_split_of(seq.relative_to(src).parts)
        if gsr and seq.name not in found and (seq / "img1").is_dir():
            found[seq.name] = SeqRef(seq.name, gsr, "dir", str(seq), "")
    if not found:
        for zp in sorted(Path(src).rglob("*.zip")):
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():
                    if not n.endswith("Labels-GameState.json"):
                        continue
                    pre = str(PurePosixPath(n).parent)
                    name = PurePosixPath(pre).name
                    gsr = _gsr_split_of(PurePosixPath(pre).parts) or _gsr_split_of([zp.name])
                    if gsr and name.startswith("SNGS") and name not in found:
                        found[name] = SeqRef(name, gsr, "zip", str(zp), pre)
    sel = select_seqs(r.number for r in found.values() if r.src_split == "valid")

    def split_of(r: SeqRef) -> str:
        if r.src_split == "train":
            return "train"
        if r.src_split == "gsrtest":
            return "pool"
        return "test" if r.number in TEST_SEQS else "select" if r.number in sel else "pool"

    refs = [SeqRef(r.name, r.src_split, r.kind, r.container, r.prefix, split_of(r)) for r in found.values()]
    return sorted(refs, key=lambda s: s.number)


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

    def close(self):
        if self._zip is not None:
            self._zip.close()


def decode(buf: bytes):
    """JPEG bytes -> BGR image. cv2.imdecode releases the GIL, so threads can share it."""
    import cv2
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode frame")
    return img


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
