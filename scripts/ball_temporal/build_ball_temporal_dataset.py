"""
build_ball_temporal_dataset.py -- step 2 of docs/path_b/plans/BALL_TEMPORAL_PLAN.md §9: the
3-frame, point-labelled crop dataset for the temporal ball specialist (plan §4-§6).

Standalone apart from gsr_io.py (same folder). Needs numpy, opencv-python. Reads the main-detector
cache from step 1 (cache_main_detections.py).

  build    write the samples and manifest.csv; resumes per sequence
  stats    counts per split and kind, from manifest.csv
  preview  contact sheets of random samples with their labels drawn, for the spot-check by eye

Per ball-labelled frame t (frames without a ball label give no crops, plan §3):
  * Window: 640x640 native pixels, shift-clamped into the frame (never padded), centred like the
    runtime (plan §6.3): on the main detector's top ball (conf >= --pred_conf) in the most recent
    of frames t-1 .. t-15; failing that, on the GT ball in the most recent labelled frame of that
    range, +-48 px jitter; failing both, no positive crop.
  * Hard frame: GT ball, and no main ball at conf >= --pred_conf within 15 px of it. Marked in
    every split; hard *positives* are only sampled as such outside `train` (decision A).
  * Positives: every hard frame (repeat 3 in pool, 1 elsewhere); other frames every 3rd (t % 3 == 0).
    A ball cut by the window is labelled if its centre and >= 50% of its box are inside, otherwise
    the crop is dropped. If the window misses the ball entirely (centred on a false positive, or
    a long gap), the crop is kept as kind `outside`, with an empty label: the runtime sees it too.
  * Negatives, from positive-sampled frames only: hard (person or pitch-line point, +-64 px) for
    15% of crops and random for 5%; no ball in the window at t, t-1 or t-2.
  * Frames t-1, t-2 are cut with the window chosen at t. Before the sequence start they repeat
    frame t (n_prev < 2). The 5% "repeat t" training case is an augmentation in step 3, not here.

Output (<out>/):
  <split>/<SNGS-xxx>/<frame>_<kind>.jpg   one sample: crops t-2 | t-1 | t side by side, 640x1920
  manifest.csv                            one row per sample (columns: MANIFEST_FIELDS)
  parts/<SNGS-xxx>.csv                    per-sequence manifest, written last (= sequence done)
balls: "cx cy size;..." in crop pixels of frame t (size = max(w, h) of the GSR box), empty if none.
main_conf: positives, the highest main-detector confidence within 15 px of the GT ball (0 if none);
negatives and `outside`, the highest main-detector confidence inside the window (0 if none).

    python scripts/ball_temporal/build_ball_temporal_dataset.py build --src data/datasets/SoccerNetGS \
        --cache data/outputs/ball_temporal/main_cache --out data/outputs/ball_temporal/dataset --workers 8
"""

from __future__ import annotations

import argparse
import csv
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
CROP = 640
LOOKBACK = 15            # frames searched back for the window centre (= max_miss_frames)
JITTER = 48              # on GT-centred windows only
EASY_STEP = 3            # easy positives: every 3rd frame
HARD_REPEAT = 3          # hard positives in pool, via the manifest's repeat column
NEG_HARD_SHARE = 0.15    # of all crops
NEG_RANDOM_SHARE = 0.05
MIN_VISIBLE = 0.5
MATCH_PX = 15.0
MANIFEST_FIELDS = ["split", "seq", "frame", "kind", "hard", "repeat", "centre_source", "centre_gap",
                   "main_conf", "x0", "y0", "clamped", "n_prev", "n_balls", "ball_size", "ball_dx",
                   "ball_dy", "balls", "file"]


# --------------------------------------------------------------------------- geometry

def clamp_window(cx: float, cy: float, W: int, H: int) -> tuple[int, int, bool]:
    """Top-left of a CROP x CROP window centred at (cx, cy), shifted (never padded) into the frame."""
    x0, y0 = int(round(cx - CROP / 2)), int(round(cy - CROP / 2))
    cx0 = min(max(x0, 0), max(W - CROP, 0))
    cy0 = min(max(y0, 0), max(H - CROP, 0))
    return cx0, cy0, (cx0, cy0) != (x0, y0)


def balls_in_window(balls, x0, y0):
    """-> (points, dropped). points: [(cx, cy, size)] in crop px; dropped if a ball is cut ambiguously."""
    pts = []
    for cx, cy, w, h, _ in balls:
        l, t, r, b = cx - w / 2 - x0, cy - h / 2 - y0, cx + w / 2 - x0, cy + h / 2 - y0
        inter = max(min(r, CROP) - max(l, 0), 0) * max(min(b, CROP) - max(t, 0), 0)
        if inter <= 0:
            continue
        centre_in = 0 <= cx - x0 < CROP and 0 <= cy - y0 < CROP
        if inter / max(w * h, 1e-6) < MIN_VISIBLE or not centre_in:
            return [], True
        pts.append((cx - x0, cy - y0, max(w, h)))
    return pts, False


def overlaps_any_ball(balls, x0, y0, margin=8):
    return any(cx + w / 2 + margin > x0 and cx - w / 2 - margin < x0 + CROP and
               cy + h / 2 + margin > y0 and cy - h / 2 - margin < y0 + CROP
               for cx, cy, w, h, _ in balls)


def _cut(img, x0, y0):
    c = img[y0:y0 + CROP, x0:x0 + CROP]
    if c.shape[:2] != (CROP, CROP):
        raise ValueError(f"frame {img.shape[:2]} smaller than {CROP}px")
    return c


# --------------------------------------------------------------------------- per sequence

def _matched_conf(dets, gts) -> float:
    return max((d[4] for d in dets for g in gts if np.hypot(d[0] - g[0], d[1] - g[1]) <= MATCH_PX),
               default=0.0)


def _conf_in_window(dets, x0, y0) -> float:
    return max((d[4] for d in dets if x0 <= d[0] < x0 + CROP and y0 <= d[1] < y0 + CROP), default=0.0)


def _centre(t, dets, balls, pred_conf, rng):
    """Runtime-like window centre for frame t -> (cx, cy, source, gap) or None."""
    for g in range(1, LOOKBACK + 1):
        d = [b for b in dets.get(t - g, []) if b[4] >= pred_conf]
        if d:
            return d[0][0], d[0][1], "pred", g
    for g in range(1, LOOKBACK + 1):
        if balls.get(t - g):
            b = balls[t - g][0]
            jx, jy = rng.uniform(-JITTER, JITTER, 2)
            return b[0] + jx, b[1] + jy, "gt", g
    return None


def process_sequence(job):
    import cv2
    ref, cache_dir, out, seed, pred_conf, quality = job
    part = Path(out) / "parts" / f"{ref.name}.csv"
    if part.exists():
        with open(part, newline="") as f:
            return ref.name, list(csv.DictReader(f)), True
    cache = json.loads((Path(cache_dir) / ref.split / f"{ref.name}.json").read_text())["frames"]
    dets = {int(k): v["balls"] for k, v in cache.items()}       # highest confidence first
    reader = SeqReader(ref)
    try:
        im_dir, files, balls, persons, lines, (W, H) = parse_labels(reader.labels())
        seq_dir = Path(out) / ref.split / ref.name
        seq_dir.mkdir(parents=True, exist_ok=True)
        frames_cache: dict[int, np.ndarray] = {}

        def frame(f):
            if f not in frames_cache:
                frames_cache[f] = decode(reader.read(f"{im_dir}/{files[f]}"))
            return frames_cache[f]

        rows = []
        for t in sorted(files):
            bt = balls.get(t)
            if not bt:
                continue
            gts = [(b[0], b[1]) for b in bt]
            hard = _matched_conf([d for d in dets.get(t, []) if d[4] >= pred_conf], gts) == 0.0
            sample_hard = hard and ref.split != "train"
            if not sample_hard and t % EASY_STEP:
                continue
            rng = np.random.default_rng([seed, ref.number, t])
            jobs = []        # (kind, x0, y0, clamped, points, meta)

            c = _centre(t, dets, balls, pred_conf, rng)
            if c is not None:
                x0, y0, clamped = clamp_window(c[0], c[1], W, H)
                pts, dropped = balls_in_window(bt, x0, y0)
                if not dropped:
                    kind = ("hard" if sample_hard else "easy") if pts else "outside"
                    conf = _matched_conf(dets.get(t, []), gts) if pts else _conf_in_window(dets.get(t, []), x0, y0)
                    rep = HARD_REPEAT if kind == "hard" and ref.split == "pool" else 1
                    jobs.append((kind, x0, y0, clamped, pts,
                                 dict(centre_source=c[2], centre_gap=c[3], main_conf=conf, repeat=rep)))

            r = rng.random()
            neg_p = 1 / (1 - NEG_HARD_SHARE - NEG_RANDOM_SHARE) - 1     # negatives per positive
            if r < neg_p:
                cands = [(p[0], p[1]) for p in persons.get(t, [])] + lines.get(t, [])
                is_hard = rng.random() < NEG_HARD_SHARE / (NEG_HARD_SHARE + NEG_RANDOM_SHARE) and bool(cands)
                near = bt + balls.get(t - 1, []) + balls.get(t - 2, [])
                for _ in range(20):
                    if is_hard:
                        cx, cy = cands[rng.integers(len(cands))]
                        cx, cy = cx + rng.uniform(-64, 64), cy + rng.uniform(-64, 64)
                    else:
                        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
                    x0, y0, clamped = clamp_window(cx, cy, W, H)
                    if not overlaps_any_ball(near, x0, y0):
                        jobs.append(("neg_hard" if is_hard else "neg_random", x0, y0, clamped, [],
                                     dict(centre_source="", centre_gap="", repeat=1,
                                          main_conf=_conf_in_window(dets.get(t, []), x0, y0))))
                        break

            if not jobs:
                continue
            prev = [f for f in (t - 2, t - 1) if f in files]
            idx = [t - 2 if t - 2 in files else t, t - 1 if t - 1 in files else t, t]
            imgs = [frame(f) for f in idx]
            for f in [k for k in frames_cache if k < t - 2]:
                del frames_cache[f]
            for kind, x0, y0, clamped, pts, meta in jobs:
                name = f"{t:06d}_{kind}.jpg"
                tile = np.concatenate([_cut(im, x0, y0) for im in imgs], axis=1)
                ok, enc = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    raise RuntimeError(f"jpeg encode failed: {ref.name} {name}")
                (seq_dir / name).write_bytes(enc.tobytes())
                p0 = pts[0] if pts else None
                rows.append(dict(split=ref.split, seq=ref.name, frame=t, kind=kind, hard=int(hard),
                                 x0=x0, y0=y0, clamped=int(clamped), n_prev=len(prev), n_balls=len(pts),
                                 ball_size=round(p0[2], 1) if p0 else "",
                                 ball_dx=round(p0[0] - CROP / 2, 1) if p0 else "",
                                 ball_dy=round(p0[1] - CROP / 2, 1) if p0 else "",
                                 balls=";".join(f"{x:.1f} {y:.1f} {s:.1f}" for x, y, s in pts),
                                 file=f"{ref.split}/{ref.name}/{name}",
                                 **{**meta, "main_conf": round(meta["main_conf"], 3)}))
        part.parent.mkdir(parents=True, exist_ok=True)
        tmp = part.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(part)
        return ref.name, rows, False
    finally:
        reader.close()


# --------------------------------------------------------------------------- stages

def stage_build(a):
    refs = [r for r in discover(a.src) if r.split in a.splits]
    if a.seqs:
        refs = [r for r in refs if r.name in a.seqs]
    missing = [r.name for r in refs if not (a.cache / r.split / f"{r.name}.json").exists()]
    if missing:
        raise SystemExit(f"no main-detector cache for {len(missing)} sequences, e.g. {missing[:3]}; run step 1")
    print(f"{len(refs)} sequences, {a.workers} workers", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(r, str(a.cache), str(a.out), a.seed, a.pred_conf, a.jpeg_quality) for r in refs]
    t0, rows = time.time(), []
    with Pool(a.workers) as pool:
        for i, (name, r, cached) in enumerate(pool.imap_unordered(process_sequence, jobs), 1):
            rows.extend(r)
            print(f"[{i}/{len(refs)}] {name}: {len(r)} samples{' (cached)' if cached else ''}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    order = {s: i for i, s in enumerate(SPLITS)}
    rows.sort(key=lambda r: (order[r["split"]], r["seq"], int(r["frame"]), r["kind"]))
    with open(a.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out / 'manifest.csv'} ({len(rows)} samples) in {(time.time() - t0) / 60:.1f} min\n")
    stage_stats(a)


def stage_stats(a):
    with open(a.out / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    kinds = ("hard", "easy", "outside", "neg_hard", "neg_random")
    lines = ["| split | samples | " + " | ".join(kinds) + " | hard frames sampled (any kind) | "
             "with repeat | GT-centred |", "|---" * (len(kinds) + 5) + "|"]
    for s in SPLITS:
        rs = [r for r in rows if r["split"] == s]
        if not rs:
            continue
        n = {k: sum(r["kind"] == k for r in rs) for k in kinds}
        hard_any = sum(r["hard"] == "1" and r["kind"] in ("hard", "outside") for r in rs)
        rep = sum(int(r["repeat"]) for r in rs)
        gt = sum(r["centre_source"] == "gt" for r in rs)
        lines.append(f"| {s} | {len(rs)} | " + " | ".join(str(n[k]) for k in kinds)
                     + f" | {hard_any} | {rep} | {gt} |")
    text = "\n".join(lines)
    print(text)
    (a.out / "stats.md").write_text(text + "\n")


def stage_preview(a):
    import cv2
    with open(a.out / "manifest.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in a.splits]
    rng = np.random.default_rng(a.seed)
    kinds = sorted({r["kind"] for r in rows})
    pick = []
    for k in kinds:
        rs = [r for r in rows if r["kind"] == k]
        pick += [rs[i] for i in rng.choice(len(rs), min(a.n, len(rs)), replace=False)]
    outdir = a.out / "preview"
    outdir.mkdir(exist_ok=True)
    per_page = 8
    for p in range(0, len(pick), per_page):
        tiles = []
        for r in pick[p:p + per_page]:
            img = cv2.imread(str(a.out / r["file"]))
            for k in range(3):
                cv2.line(img, (CROP * k, 0), (CROP * k, CROP), (255, 255, 255), 2)
            for b in filter(None, r["balls"].split(";")):
                x, y, s = map(float, b.split())
                for k in range(3):     # same window position on every frame; the label is for t
                    col = (0, 255, 0) if k == 2 else (0, 160, 255)
                    cv2.circle(img, (int(x) + CROP * k, int(y)), int(max(s, 8)) + 6, col, 2)
            cv2.drawMarker(img, (CROP * 2 + CROP // 2, CROP // 2), (255, 0, 255), cv2.MARKER_CROSS, 20, 1)
            txt = (f"{r['split']} {r['seq']} f{r['frame']} {r['kind']} hard={r['hard']} "
                   f"src={r['centre_source']}{r['centre_gap']} conf={r['main_conf']} n_prev={r['n_prev']}")
            cv2.putText(img, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        path = outdir / f"page_{p // per_page:02d}.jpg"
        cv2.imwrite(str(path), np.concatenate(tiles, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "stats", "preview"])
    ap.add_argument("--src", type=Path, help="GSR source: sequence dirs or the split zips")
    ap.add_argument("--cache", type=Path, default=Path("data/outputs/ball_temporal/main_cache"))
    ap.add_argument("--out", type=Path, default=Path("data/outputs/ball_temporal/dataset"))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--seqs", nargs="*", default=[], help="only these sequences, e.g. SNGS-024 (smoke tests)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pred_conf", type=float, default=0.15, help="main-detector confidence counted as a sighting")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--n", type=int, default=6, help="preview: samples per kind")
    a = ap.parse_args()
    if a.stage == "build" and not a.src:
        ap.error("--src is required")
    {"build": stage_build, "stats": stage_stats, "preview": stage_preview}[a.stage](a)


if __name__ == "__main__":
    main()
