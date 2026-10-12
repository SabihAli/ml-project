"""
cache_main_detections.py -- step 1 of docs/path_b/plans/BALL_TEMPORAL_PLAN.md §9: run the main
athlete/ball detector over every frame and cache its ball detections per sequence.

The cache feeds step 2 (the temporal dataset builder): it marks hard frames (GSR has a ball,
the main detector missed it) and centres crops on the last prediction, as the runtime does.

Standalone apart from gsr_io.py (same folder). Needs ultralytics>=8.4.0 (YOLO26).

  list     print the sequences per split (train / pool / select / test); no GPU
  run      detect and cache; --shard i --nshards n splits the sequences across GPUs
  summary  main-detector recall per split and hard-frame counts, from the cache; no GPU

Cache: <out>/<split>/<SNGS-xxx>.json =
  {"meta": {weights, imgsz, conf, top}, "frames": {"<frame>": {"balls": [[cx, cy, w, h, conf], ...],
                                                               "gt": [[cx, cy, size], ...]}}}
balls: every ball detection with conf >= --conf, highest first, at most --top, in native pixels.
Finished sequences are skipped, and each file is written atomically, so a stopped run resumes.

    python cache_main_detections.py run --src /kaggle/input/datasets/sabihalii/soccernetgsr \
        --out /kaggle/working/main_cache --weights /kaggle/working/kit/best.pt --device 0 --half
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsr_io import SeqReader, decode, discover, parse_labels  # noqa: E402

SPLITS = ("train", "pool", "select", "test")
MATCH_PX = 15.0      # ball match radius, as the gates (plan §8)
HARD_CONF = 0.15     # a frame is hard when no ball at this confidence is within MATCH_PX of GT


def _balls(result, cls: int, top: int) -> list[list[float]]:
    b = result.boxes
    if b is None or len(b) == 0:
        return []
    keep = b.cls.cpu().numpy().astype(int) == cls
    xywh, conf = b.xywh.cpu().numpy()[keep], b.conf.cpu().numpy()[keep]
    order = np.argsort(-conf)[:top]
    return [[round(float(v), 2) for v in (*xywh[i], conf[i])] for i in order]


def _refs(a):
    refs = [r for r in discover(a.src) if r.split in a.splits]
    if a.seqs:
        refs = [r for r in refs if r.name in a.seqs]
    return refs


def stage_list(a):
    refs = discover(a.src)
    for s in SPLITS:
        names = [r.name for r in refs if r.split == s]
        print(f"{s:6s} {len(names):3d}  {' '.join(n.replace('SNGS-', '') for n in names)}")


def stage_run(a):
    from ultralytics import YOLO
    model = YOLO(a.weights)
    names = {v: k for k, v in model.names.items()}
    if "ball" not in names:
        raise SystemExit(f"model has no 'ball' class: {model.names}")
    cls = names["ball"]
    kw = dict(imgsz=a.imgsz, conf=a.conf, iou=0.7, max_det=300, device=a.device, verbose=False)
    if a.half:
        kw["half"] = True                         # only when asked: Ultralytics 8.4 warns on any `half`
    meta = {"weights": str(a.weights), "imgsz": a.imgsz, "conf": a.conf, "top": a.top}

    refs = [r for i, r in enumerate(_refs(a)) if i % a.nshards == a.shard]
    tag = f"[gpu{a.device}]"
    print(f"{tag} {len(refs)} sequences", flush=True)
    pool = ThreadPoolExecutor(a.decode_threads)
    t_all = time.time()
    for k, ref in enumerate(refs, 1):
        out = a.out / ref.split / f"{ref.name}.json"
        if out.exists():
            print(f"{tag} {k}/{len(refs)} {ref.name}: cached", flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rd = SeqReader(ref)
        im_dir, files, gt, *_ = parse_labels(rd.labels())
        frames = sorted(files)[:a.max_frames] if a.max_frames else sorted(files)
        rec = {}
        for i in range(0, len(frames), a.batch):
            chunk = frames[i:i + a.batch]
            bufs = [rd.read(f"{im_dir}/{files[f]}") for f in chunk]   # zip reads stay on one thread
            imgs = list(pool.map(decode, bufs))
            for f, r in zip(chunk, model.predict(imgs, **kw)):
                rec[str(f)] = {"balls": _balls(r, cls, a.top),
                               "gt": [[round(b[0], 2), round(b[1], 2), round(max(b[2], b[3]), 2)]
                                      for b in gt.get(f, [])]}
        rd.close()
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps({"meta": meta, "frames": rec}))
        tmp.replace(out)
        dt = time.time() - t0
        print(f"{tag} {k}/{len(refs)} {ref.split}/{ref.name}: {len(rec)} frames in {dt:.0f}s "
              f"({1000 * dt / max(len(rec), 1):.0f} ms/frame), elapsed {(time.time() - t_all) / 60:.1f} min",
              flush=True)
    print(f"{tag} done in {(time.time() - t_all) / 60:.1f} min", flush=True)


def _hit(balls, gts, conf) -> bool:
    return any(b[4] >= conf and np.hypot(b[0] - g[0], b[1] - g[1]) <= MATCH_PX for b in balls for g in gts)


def stage_summary(a):
    lines = ["# Main-detector cache summary", "",
             f"Hard frame: GSR has a ball and no main-detector ball at conf >= {HARD_CONF} is within "
             f"{MATCH_PX:.0f} px of it.", "",
             "| split | sequences | frames | frames with GT ball | main recall (top ball) | hard frames | hard share |",
             "|---|---|---|---|---|---|---|"]
    for s in SPLITS:
        files = sorted((a.out / s).glob("SNGS-*.json"))
        if not files:
            continue
        n_fr = n_gt = n_top = n_hard = 0
        for fp in files:
            for v in json.loads(fp.read_text())["frames"].values():
                n_fr += 1
                if not v["gt"]:
                    continue
                n_gt += 1
                n_top += _hit(v["balls"][:1], v["gt"], HARD_CONF)
                n_hard += not _hit(v["balls"], v["gt"], HARD_CONF)
        lines.append(f"| {s} | {len(files)} | {n_fr} | {n_gt} | {n_top / max(n_gt, 1):.2%} | "
                     f"{n_hard} | {n_hard / max(n_gt, 1):.2%} |")
    lines += ["", "If train recall is clearly above pool/select/test, the detector memorised train, "
                  "and hard examples must come from pool only (plan §6, decision A)."]
    text = "\n".join(lines)
    print(text)
    (a.out / "summary.md").write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["list", "run", "summary"])
    ap.add_argument("--src", type=Path, help="GSR source: sequence dirs or the split zips")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/main_cache"))
    ap.add_argument("--weights", default="/kaggle/working/kit/best.pt", help="main athlete/ball detector")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--seqs", nargs="*", default=[], help="only these sequences, e.g. SNGS-060 (smoke tests)")
    ap.add_argument("--imgsz", type=int, default=1280, help="1280 = the 736x1280 runtime input after letterbox")
    ap.add_argument("--conf", type=float, default=0.05, help="lowest ball confidence kept in the cache")
    ap.add_argument("--top", type=int, default=5, help="most ball detections kept per frame")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    ap.add_argument("--decode_threads", type=int, default=2, help="JPEG decode threads per process")
    ap.add_argument("--max_frames", type=int, default=0, help="first N frames per sequence (smoke tests)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.stage in ("list", "run") and not a.src:
        ap.error("--src is required")
    {"list": stage_list, "run": stage_run, "summary": stage_summary}[a.stage](a)


if __name__ == "__main__":
    main()
