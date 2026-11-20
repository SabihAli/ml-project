"""
label_audit.py -- Phase 3b.2 step 4: are the teacher's near misses teacher errors or GSR label errors?
(docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md, Phase 3b.2). CPU only.

Decodes a candidate set with frozen decode_trajectory.py knobs, then:
  sheet  -- a contact sheet of N random near misses (pick 15-60 px from the label) on a split: label green,
            label at t-1 yellow, t+1 magenta, pick red cross. Judged by eye.
  lag    -- on fast balls (>= 8 px/frame), the time offset in [-2, 2] frames at which the (interpolated) label
            track best matches the pick; and recall when a pick within 15 px of the label track between t-1 and
            t+1 also counts (an upper bound: it also forgives a pick that really is one frame off).

    python scripts/pipeline_improvement_1/phase3b/label_audit.py --cands .../candidates/fuse_t2m_s1920_ft+t1_r34_a0.7 --sheet audit.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_trajectory as dt  # noqa: E402
import train_teacher_yolo as base  # noqa: E402
import trajectory_pass as tp  # noqa: E402
from gsr_io import SeqReader, discover, parse_labels  # noqa: E402

CFG = (0.08, 0.05, 8.0, 25.0, 60.0, 10, 0.0, 0.2, 1, 10)   # chosen on select, Phase 3b.2 step 1
THR = 0.06


def label_at(G: dict, t: float):
    f0 = int(np.floor(t))
    a, b = G.get(f0, []), G.get(f0 + 1, [])
    if t == f0:
        return np.array(a[0]) if a else None
    return np.array(a[0]) + (t - f0) * (np.array(b[0]) - np.array(a[0])) if a and b else None


def lag(preds, gt) -> str:
    deltas = np.round(np.arange(-2, 2.01, 0.25), 2)
    best, n, hit, hit_lag = [], 0, 0, 0
    for s, G in gt.items():
        for f, g in G.items():
            if not g:
                continue
            n += 1
            p, q = preds[s].get(f), np.array(g[0])
            if p is None:
                continue
            pp = np.array(p[:2])
            d0 = np.hypot(*(pp - q))
            near = [x for dd in (-1, -0.5, 0.5, 1) if (x := label_at(G, f + dd)) is not None]
            hit += d0 <= 15
            hit_lag += min([d0] + [np.hypot(*(pp - x)) for x in near]) <= 15
            g0, g1 = G.get(f - 1, []), G.get(f + 1, [])
            if g0 and g1 and d0 <= 60 and np.hypot(*(np.array(g1[0]) - np.array(g0[0]))) / 2 >= 8:
                ds = [np.hypot(*(pp - x)) if (x := label_at(G, f + dd)) is not None else 1e9 for dd in deltas]
                best.append(deltas[int(np.argmin(ds))])
    b = np.array(best)
    return (f"fast-ball frames {len(b)}: best label offset mean {b.mean():+.2f} frames, >0 {(b > 0).mean():.0%}, "
            f"=0 {(b == 0).mean():.0%}, <0 {(b < 0).mean():.0%}; recall strict {hit / n:.2%}, "
            f"lag-tolerant {hit_lag / n:.2%}")


def sheet(preds, gt, split: str, n: int, out: Path, seed: int = 0) -> None:
    cases = [(s, f, p, np.hypot(p[0] - g[0][0], p[1] - g[0][1])) for s in gt for f, g in gt[s].items()
             if g and (p := preds[s].get(f)) is not None and 15 < np.hypot(p[0] - g[0][0], p[1] - g[0][1]) <= 60]
    random.seed(seed)
    pick = sorted(random.sample(cases, min(n, len(cases))))
    refs = {r.name: r for r in discover(base.ROOT / "data" / "gsr") if r.split == split}
    tiles, R = [], 60
    for s, f, p, d in pick:
        rd = SeqReader(refs[s])
        try:
            im_dir, files, *_ = parse_labels(rd.labels())
        finally:
            rd.close()
        im = cv2.imread(str(Path(refs[s].container) / im_dir / files[f]))
        gx, gy = gt[s][f][0]
        for k, col in ((-1, (0, 255, 255)), (1, (255, 0, 255))):
            if g2 := gt[s].get(f + k, []):
                cv2.circle(im, (int(g2[0][0]), int(g2[0][1])), 3, col, 1)
        cv2.circle(im, (int(gx), int(gy)), 4, (0, 255, 0), 1)
        cv2.drawMarker(im, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
        H, W = im.shape[:2]
        cx, cy = int((gx + p[0]) / 2), int((gy + p[1]) / 2)
        x0, y0 = max(0, min(W - 2 * R, cx - R)), max(0, min(H - 2 * R, cy - R))
        t = cv2.resize(im[y0:y0 + 2 * R, x0:x0 + 2 * R], (240, 240), interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, f"{s[5:]} {f} d{d:.0f}", (3, 14), 0, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 8)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
    print(f"{len(cases)} near misses on {split}; sheet of {len(pick)} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Label audit of the teacher's near misses")
    ap.add_argument("--cands", type=Path, required=True)
    ap.add_argument("--sheet", type=Path, default=None, help="write a contact sheet of select near misses here")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    for split in ("select", "test"):
        cands, gt = tp.load(a.cands / f"{split}.json")
        preds = dt.run({s: dt.arrays(fr, cs) for s, (fr, cs) in cands.items()}, CFG, THR)
        print(f"{split}: {lag(preds, gt)}")
        if split == "select" and a.sheet:
            sheet(preds, gt, split, a.n, a.sheet)


if __name__ == "__main__":
    main()
