"""
camera_motion_cache.py -- per-frame camera motion for every sequence, cached, for the ball
tracker's offline evaluation and the dataset builder (docs/plans/BALL_IMPROVEMENT_PLAN.md §2.3,
§3.2). Standalone apart from gsr_io.py (same folder). Needs numpy, opencv-python.

For frame t: a 2x3 similarity transform (rotation, uniform scale, translation) taking frame t-1
pixels to frame t pixels, estimated from sparse optical flow on a 960-wide grey image with
people and the ball masked out (GT boxes, expanded 20%), then scaled back to full resolution.
None when fewer than --min_tracked features survive or RANSAC inliers fall below --min_inliers
(a cut, a close-up): the tracker then treats the frame as having no motion information.

GT boxes are used for the mask, so this is slightly cleaner than the runtime, which masks
detections. The runtime estimate is gsr_pipeline/camera_motion.py; this file is the offline
stand-in and does not import it, so it runs in the Kaggle kit unchanged.

    run      compute and cache: <out>/<split>/<SNGS-xxx>.json = {"meta": ..., "frames": {"<t>": [6 floats] | null}}
    summary  share of frames with an estimate, median |translation| per split

    python scripts/ball_temporal/camera_motion_cache.py run --src data/datasets/SoccerNetGS \
        --out data/outputs/ball_temporal/motion_cache --splits select test --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsr_io import SeqReader, decode, discover, parse_labels  # noqa: E402

SPLITS = ("train", "pool", "select", "test")
WORK_W = 960


def _mask(shape, boxes, scale):
    m = np.full(shape, 255, np.uint8)
    for cx, cy, w, h, _ in boxes:
        w2, h2 = 0.6 * w * scale, 0.6 * h * scale
        x0, y0 = int(max(cx * scale - w2, 0)), int(max(cy * scale - h2, 0))
        x1, y1 = int(cx * scale + w2) + 1, int(cy * scale + h2) + 1
        m[y0:y1, x0:x1] = 0
    return m


def estimate(prev_g, cur_g, mask, scale, min_tracked=40, min_inliers=0.5):
    """Grey images at work resolution -> full-resolution 2x3 affine (frame t-1 -> t) or None."""
    import cv2
    pts = cv2.goodFeaturesToTrack(prev_g, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=mask)
    if pts is None or len(pts) < min_tracked:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, pts, None, winSize=(21, 21), maxLevel=3)
    good = st.reshape(-1) == 1
    if good.sum() < min_tracked:
        return None
    a, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if a is None or inl is None or inl.mean() < min_inliers:
        return None
    a = a.copy()
    a[:, 2] /= scale                      # translation back to full-resolution pixels
    return a


def process_sequence(job):
    import cv2
    ref, out_dir, min_tracked, min_inliers = job
    out = Path(out_dir) / ref.split / f"{ref.name}.json"
    if out.exists():
        return ref.name, None, True
    reader = SeqReader(ref)
    try:
        im_dir, files, balls, persons, _lines, (W, H) = parse_labels(reader.labels())
        scale = WORK_W / W
        size = (WORK_W, int(round(H * scale)))
        rec, prev, ok = {}, None, 0
        for t in sorted(files):
            g = cv2.cvtColor(cv2.resize(decode(reader.read(f"{im_dir}/{files[t]}")), size,
                                        interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            a = None
            if prev is not None and prev[0] == t - 1:
                m = _mask(prev[1].shape, persons.get(t - 1, []) + balls.get(t - 1, []), scale)
                a = estimate(prev[1], g, m, scale, min_tracked, min_inliers)
            rec[str(t)] = None if a is None else [round(float(v), 6) for v in a.reshape(-1)]
            ok += a is not None
            prev = (t, g)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({"meta": {"work_w": WORK_W, "min_tracked": min_tracked,
                                            "min_inliers": min_inliers}, "frames": rec}))
        tmp.replace(out)
        return ref.name, ok / max(len(files), 1), False
    finally:
        reader.close()


def load(cache_dir: Path, ref) -> dict:
    """{frame: 2x3 affine (np.ndarray) or None} for one sequence."""
    d = json.loads((Path(cache_dir) / ref.split / f"{ref.name}.json").read_text())["frames"]
    return {int(k): (None if v is None else np.array(v, float).reshape(2, 3)) for k, v in d.items()}


def stage_run(a):
    refs = [r for r in discover(a.src) if r.split in a.splits]
    if a.seqs:
        refs = [r for r in refs if r.name in a.seqs]
    print(f"camera motion: {len(refs)} sequences, {a.workers} workers -> {a.out}", flush=True)
    t0 = time.time()
    jobs = [(r, str(a.out), a.min_tracked, a.min_inliers) for r in refs]
    with Pool(a.workers) as pool:
        for i, (name, share, cached) in enumerate(pool.imap_unordered(process_sequence, jobs), 1):
            tail = "cached" if cached else f"{share:.0%} frames with an estimate"
            print(f"[motion {i}/{len(refs)}] {name}: {tail} ({time.time() - t0:.0f}s)", flush=True)


def stage_summary(a):
    lines = ["| split | sequences | frames | with an estimate | median translation px |", "|---|---|---|---|---|"]
    for s in SPLITS:
        files = sorted((a.out / s).glob("SNGS-*.json"))
        if not files:
            continue
        n = k = 0
        tr = []
        for fp in files:
            for v in json.loads(fp.read_text())["frames"].values():
                n += 1
                if v is not None:
                    k += 1
                    tr.append(float(np.hypot(v[2], v[5])))
        lines.append(f"| {s} | {len(files)} | {n} | {k / max(n, 1):.1%} | {np.median(tr) if tr else 0:.1f} |")
    text = "\n".join(lines)
    print(text)
    (a.out / "summary.md").write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "summary"])
    ap.add_argument("--src", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/outputs/ball_temporal/motion_cache"))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--seqs", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--min_tracked", type=int, default=40)
    ap.add_argument("--min_inliers", type=float, default=0.5)
    a = ap.parse_args()
    if a.stage == "run" and not a.src:
        ap.error("--src is required")
    {"run": stage_run, "summary": stage_summary}[a.stage](a)


if __name__ == "__main__":
    main()
