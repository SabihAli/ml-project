"""
fuse_candidates.py -- Phase 3b: merge two teachers' per-frame candidates (dump format) into one candidate set
for trajectory_pass.py (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md, Phase 3b).

Per frame, candidates of A and B within --radius px are the same object: fused score = a * conf_A + (1 - a) *
conf_B (a missing partner counts 0), position = the score-weighted mean. Unmatched candidates keep their own
weighted score, so an object only one model sees can still win if it is confident. Top --topk kept.
`a` is a knob like the trajectory pass's: one folder per value, chosen on select (trajectory_pass.py on each).

    python scripts/pipeline_improvement_1/phase3b/fuse_candidates.py --a_dir .../candidates/t2m_s1920_ft --b_dir .../candidates/t1_r34 --alpha 0.3 0.5 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fuse_frame(ca, cb, a: float, radius: float, topk: int):
    used, out = set(), []
    for xa, ya, pa in ca:
        j = min((k for k in range(len(cb)) if k not in used),
                key=lambda k: (cb[k][0] - xa) ** 2 + (cb[k][1] - ya) ** 2, default=None)
        if j is not None and (cb[j][0] - xa) ** 2 + (cb[j][1] - ya) ** 2 <= radius ** 2:
            used.add(j)
            xb, yb, pb = cb[j]
            wa, wb = a * pa, (1 - a) * pb
            out.append([(wa * xa + wb * xb) / (wa + wb), (wa * ya + wb * yb) / (wa + wb), wa + wb])
        else:
            out.append([xa, ya, a * pa])
    out += [[x, y, (1 - a) * p] for k, (x, y, p) in enumerate(cb) if k not in used]
    return sorted(out, key=lambda c: -c[2])[:topk]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fuse two teachers' candidate dumps")
    ap.add_argument("--a_dir", type=Path, required=True)
    ap.add_argument("--b_dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.5], help="weight of A")
    ap.add_argument("--radius", type=float, default=15.0)
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()
    for alpha in a.alpha:
        out = a.a_dir.parent / f"fuse_{a.a_dir.name}+{a.b_dir.name}_a{alpha:g}"
        out.mkdir(exist_ok=True)
        for split in ("select", "test"):
            da, db = (json.loads((d / f"{split}.json").read_text()) for d in (a.a_dir, a.b_dir))
            seqs = {}
            for s, sa in da["seqs"].items():
                sb = db["seqs"][s]
                frames = set(sa["cands"]) | set(sb["cands"])
                seqs[s] = {"gt": sa["gt"], "cands": {f: fuse_frame(sa["cands"].get(f, []), sb["cands"].get(f, []),
                                                                   alpha, a.radius, a.topk) for f in frames}}
            doc = {"weights": [da["weights"], db["weights"]], "alpha": alpha, "topk": a.topk, "seqs": seqs}
            (out / f"{split}.json").write_text(json.dumps(doc))
        print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
