"""
evaluate_ball_specialist.py -- gates of docs/path_b/plans/BALL_TEMPORAL_PLAN.md §6.

Standalone apart from build_ball_crop_dataset.py (same folder, for the data readers). Needs
ultralytics>=8.4.0, numpy, opencv-python.

Stages (all on the 15 held-out test sequences):

  crops   §6.1 crop-level diagnostic: specialist alone on the test crops, recall/precision by
          ball size, offset from the crop centre and edge clamping.
  main    §6.2 pass 1: main detector over EVERY frame, top-confidence ball cached per frame.
  fuse    §6.2 pass 2: replay the implementation plan §5.1 state machine (fixed 640 window,
          max_miss_frames 15, centres from predictions only), run the specialist when armed.
  report  §6.2 table: main alone vs main + specialist, plus specialist recall per gap.

`main` and `fuse` take --shard i --nshards n, so two processes can split the sequences across
both T4s. Results are cached per sequence in --cache; finished sequences are skipped.

    python scripts/evaluate_ball_specialist.py main  --src SRC --cache /kaggle/working/eval --main_weights kit/best.pt --device 0 --shard 0 --nshards 2
    python scripts/evaluate_ball_specialist.py fuse  --src SRC --cache /kaggle/working/eval --spec_weights runs/s_baseline/weights/best.pt --tag s_baseline --device 0
    python scripts/evaluate_ball_specialist.py report --cache /kaggle/working/eval --tag s_baseline
    python scripts/evaluate_ball_specialist.py crops --data /kaggle/working/ball_crops --spec_weights ... --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ball_crop_dataset import CROP, SeqReader, clamp_window, discover, parse_labels  # noqa: E402

MATCH_PX = 15.0          # ball match radius, as evaluate_detection.py
MAX_MISS = 15            # plan §2.3; --max_miss overrides
FPS = 25.0
CMC_SCALE = 4            # --cmc: phase correlation on a 1/4-size grey frame
CMC_MIN_RESPONSE = 0.1   # below this peak response the shift estimate is ignored
GAP_BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 10, "6-10"), (11, 15, "11-15"))


def _ball_class(model) -> int:
    names = {v: k for k, v in model.names.items()}
    if "ball" not in names and len(model.names) == 1:
        return next(iter(model.names))   # 1-class model (e.g. trained with single_cls -> 'item')
    if "ball" not in names:
        raise SystemExit(f"model has no 'ball' class: {model.names}")
    return names["ball"]


def _top_ball(result, cls: int, dx: int = 0, dy: int = 0):
    b = result.boxes
    if b is None or len(b) == 0:
        return None
    keep = b.cls.cpu().numpy().astype(int) == cls
    if not keep.any():
        return None
    xyxy, conf = b.xyxy.cpu().numpy()[keep], b.conf.cpu().numpy()[keep]
    i = int(conf.argmax())
    x1, y1, x2, y2 = xyxy[i]
    return [float((x1 + x2) / 2 + dx), float((y1 + y2) / 2 + dy), float(conf[i])]


def _fp16(a) -> dict:
    return {"half": True} if a.half else {}   # only pass it when asked: Ultralytics 8.4 warns on any `half`


def _test_seqs(src: Path, shard: int, nshards: int):
    seqs = [s for s in discover(src) if s.split == "test"]
    return [s for i, s in enumerate(seqs) if i % nshards == shard]


def _cmc_grey(img):
    """Downscaled float32 grey frame for cv2.phaseCorrelate (global camera translation)."""
    small = cv2.resize(img, None, fx=1 / CMC_SCALE, fy=1 / CMC_SCALE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _hit(pred, gts) -> bool:
    return pred is not None and any(np.hypot(pred[0] - g[0], pred[1] - g[1]) <= MATCH_PX for g in gts)


# --------------------------------------------------------------------------- main pass

def stage_main(a):
    from ultralytics import YOLO
    model = YOLO(a.main_weights)
    cls = _ball_class(model)
    a.cache.mkdir(parents=True, exist_ok=True)
    for ref in _test_seqs(a.src, a.shard, a.nshards):
        out = a.cache / f"{ref.name}.main.json"
        if out.exists():
            print(f"{ref.name}: cached")
            continue
        rd = SeqReader(ref)
        im_dir, files, balls, *_ = parse_labels(rd.labels())
        frames = sorted(files)[:a.max_frames] if a.max_frames else sorted(files)
        rec = {}
        for i in range(0, len(frames), a.batch):
            chunk = frames[i:i + a.batch]
            imgs = [rd.frame(im_dir, files[f]) for f in chunk]
            # imgsz 1280 + Ultralytics' auto letterbox = the 736x1280 input the runtime uses;
            # iou 0.7 / max_det 300 = the training config (implementation plan §3.5)
            res = model.predict(imgs, imgsz=1280, conf=a.main_conf, iou=0.7, max_det=300,
                                device=a.device, verbose=False, **_fp16(a))
            for f, r in zip(chunk, res):
                rec[f] = {"main": _top_ball(r, cls),
                          "gt": [[b[0], b[1], max(b[2], b[3])] for b in balls.get(f, [])]}
        rd.close()
        out.write_text(json.dumps(rec))
        print(f"{ref.name}: {len(rec)} frames, main ball in "
              f"{sum(v['main'] is not None for v in rec.values())}", flush=True)


# --------------------------------------------------------------------------- fused replay

def stage_fuse(a):
    from ultralytics import YOLO
    spec = YOLO(a.spec_weights)
    cls = _ball_class(spec)
    for ref in _test_seqs(a.src, a.shard, a.nshards):
        cache = a.cache / f"{ref.name}.main.json"
        out = a.cache / f"{ref.name}.{a.tag}.fuse.json"
        if not cache.exists():
            raise SystemExit(f"run the main stage first: {cache} missing")
        if out.exists():
            print(f"{ref.name}: cached")
            continue
        rec = {int(k): v for k, v in json.loads(cache.read_text()).items()}
        rd = SeqReader(ref)
        im_dir, files, _balls, _persons, _lines, (W, H) = parse_labels(rd.labels())
        armed, centre, miss, last_seen = False, None, 0, None
        prev = None                                        # --cmc: (frame, small grey image)
        for f in sorted(rec):
            r = rec[f]
            r["spec"], r["ran"], r["gap"] = None, False, None
            img = None
            if a.cmc and centre is not None and prev is not None and prev[0] == f - 1:
                img = rd.frame(im_dir, files[f])
                (dx, dy), resp = cv2.phaseCorrelate(prev[1], _cmc_grey(img))
                if resp >= CMC_MIN_RESPONSE:               # follow the camera pan, not the ball
                    centre = [centre[0] + dx * CMC_SCALE, centre[1] + dy * CMC_SCALE]
            if armed and centre is not None:
                if img is None:
                    img = rd.frame(im_dir, files[f])
                x0, y0, _ = clamp_window(centre[0], centre[1], W, H)
                crop = img[y0:y0 + CROP, x0:x0 + CROP]
                res = spec.predict(crop, imgsz=CROP, conf=a.spec_conf, max_det=10,
                                   device=a.device, verbose=False, **_fp16(a))[0]
                r["ran"], r["gap"] = True, f - last_seen
                r["spec"] = _top_ball(res, cls, x0, y0)
            if r["main"] is not None:                      # main wins; disarm
                r["fused"], r["source"] = r["main"], "main"
                centre, miss, armed, last_seen = r["main"][:2], 0, False, f
            elif r["spec"] is not None:                    # specialist found it; stay armed
                r["fused"], r["source"] = r["spec"], "specialist"
                centre, miss, armed, last_seen = r["spec"][:2], 0, True, f
            else:
                r["fused"], r["source"] = None, None
                if centre is not None:
                    miss += 1
                    armed = miss < a.max_miss              # the next run would be at gap miss+1 <= max_miss
                    if not armed:
                        centre = None
            if a.cmc and centre is not None:
                prev = (f, _cmc_grey(img if img is not None else rd.frame(im_dir, files[f])))
            else:
                prev = None
        rd.close()
        out.write_text(json.dumps(rec))
        print(f"{ref.name}: specialist ran {sum(v['ran'] for v in rec.values())}x, "
              f"used {sum(v['source'] == 'specialist' for v in rec.values())}x", flush=True)


# --------------------------------------------------------------------------- report

def _score(rec: dict, key: str):
    """-> GT-ball frames, hits on them, predicted frames, correct predictions, longest miss (frames)."""
    gt_frames = tp = pred = tp_pred = 0
    run = longest = 0
    for f in sorted(rec):
        p, gts = rec[f][key], rec[f]["gt"]
        hit = _hit(p, gts)
        pred += p is not None
        tp_pred += hit
        if gts:
            gt_frames += 1
            tp += hit
            run = 0 if hit else run + 1
            longest = max(longest, run)
        else:
            run = 0
    return gt_frames, tp, pred, tp_pred, longest


def stage_report(a):
    files = sorted(a.cache.glob(f"*.{a.tag}.fuse.json"))
    if not files:
        raise SystemExit(f"no {a.tag} fuse results in {a.cache}")
    tot = {k: np.zeros(4) for k in ("main", "fused")}
    longest = {k: [] for k in ("main", "fused")}
    runs = frames = 0
    by_gap: dict[str, list[int]] = {}
    for fp in files:
        rec = {int(k): v for k, v in json.loads(fp.read_text()).items()}
        frames += len(rec)
        for k in ("main", "fused"):
            g, tp, pr, tpp, lg = _score(rec, k)
            tot[k] += (g, tp, pr, tpp)
            longest[k].append(lg)
        for r in rec.values():
            if not r["ran"]:
                continue
            runs += 1
            if r["gt"]:
                b = next(lbl for lo, hi, lbl in GAP_BUCKETS if lo <= r["gap"] <= hi)
                s = by_gap.setdefault(b, [0, 0])
                s[0] += 1
                s[1] += _hit(r["spec"], r["gt"])
    lines = [f"# Ball specialist, system-level gate (tag `{a.tag}`, {len(files)} test sequences, {frames} frames)", "",
             "| | frame recall | precision | longest miss, max | longest miss, mean |", "|---|---|---|---|---|"]
    for k, label in (("main", "main detector alone"), ("fused", "main + specialist")):
        g, tp, pr, tpp = tot[k]
        lines.append(f"| {label} | {tp / max(g, 1):.2%} | {tpp / max(pr, 1):.2%} | "
                     f"{max(longest[k]) / FPS:.2f} s | {np.mean(longest[k]) / FPS:.2f} s |")
    rm, rf = tot["main"][1] / max(tot["main"][0], 1), tot["fused"][1] / max(tot["fused"][0], 1)
    pm, pf = tot["main"][3] / max(tot["main"][2], 1), tot["fused"][3] / max(tot["fused"][2], 1)
    lines += ["", f"Recall gain {100 * (rf - rm):+.2f} points (gate: >= +10). "
                  f"Precision change {100 * (pf - pm):+.2f} points (gate: >= 0). "
                  f"Specialist runs per 100 frames: {100 * runs / max(frames, 1):.1f}.",
              "", "Specialist recall per gap since the last sighting (runs on frames with a GT ball):", "",
              "| gap | runs | hits | recall |", "|---|---|---|---|"]
    for _, _, b in GAP_BUCKETS:
        if b in by_gap:
            n, h = by_gap[b]
            lines.append(f"| {b} | {n} | {h} | {h / n:.2%} |")
    lines += ["", "Camera-motion compensation only if the fuse run used --cmc (plan §6.2). The kickoff is included."]
    text = "\n".join(lines)
    print(text)
    (a.cache / f"report_{a.tag}.md").write_text(text)


# --------------------------------------------------------------------------- crop-level

def stage_crops(a):
    from ultralytics import YOLO
    spec = YOLO(a.spec_weights)
    cls = _ball_class(spec)
    with open(a.data / "manifest.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if a.max_crops:
        rows = rows[:a.max_crops]
    buckets: dict[str, list[int]] = {}           # group -> [GT balls, hits, predictions, correct predictions]

    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        res = spec.predict([str(a.data / r["file"]) for r in chunk], imgsz=CROP, conf=a.spec_conf,
                           max_det=10, device=a.device, verbose=False, **_fp16(a))
        for r, out in zip(chunk, res):
            lab = a.data / r["file"].replace("images/", "labels/").replace(".jpg", ".txt")
            gts = [[float(x) * CROP for x in ln.split()[1:3]] for ln in lab.read_text().splitlines() if ln.strip()]
            p = _top_ball(out, cls)
            hit = _hit(p, gts)
            groups = ["all", f"kind {r['kind']}", "edge-clamped" if r["clamped"] == "1" else "not clamped"]
            if gts:
                px = float(r["ball_px"])
                groups.append("size <10" if px < 10 else "size 10-20" if px < 20 else "size >=20")
                off = np.hypot(float(r["ball_cx_in_crop"]) - CROP / 2, float(r["ball_cy_in_crop"]) - CROP / 2)
                groups.append("offset <100" if off < 100 else "offset 100-200" if off < 200 else "offset >=200")
            for g in groups:
                s = buckets.setdefault(g, [0, 0, 0, 0])
                s[0] += bool(gts)
                s[1] += hit
                s[2] += p is not None
                s[3] += hit
    lines = [f"# Ball specialist, crop-level diagnostic ({len(rows)} test crops, conf {a.spec_conf})", "",
             "| group | GT balls | recall | predictions | precision |", "|---|---|---|---|---|"]
    for name, (g, tp, pr, tpp) in buckets.items():
        lines.append(f"| {name} | {g} | {f'{tp / g:.2%}' if g else 'n/a'} | {pr} | "
                     f"{f'{tpp / pr:.2%}' if pr else 'n/a'} |")
    lines += ["", "Gate: recall >= 85% on `size 10-20`, precision >= 80% on `all`."]
    text = "\n".join(lines)
    print(text)
    (a.data / "crops_report.md").write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["main", "fuse", "report", "crops"])
    ap.add_argument("--src", type=Path, help="GSR source (as build_ball_crop_dataset.py --src)")
    ap.add_argument("--cache", type=Path, default=Path("/kaggle/working/eval"))
    ap.add_argument("--data", type=Path, help="crop dataset dir (crops stage)")
    ap.add_argument("--main_weights", default="/kaggle/working/kit/best.pt")
    ap.add_argument("--spec_weights")
    ap.add_argument("--tag", default="s_baseline", help="name for this specialist's results")
    ap.add_argument("--main_conf", type=float, default=0.15)
    ap.add_argument("--spec_conf", type=float, default=0.25)
    ap.add_argument("--max_miss", type=int, default=MAX_MISS,
                    help="fuse stage: frames the specialist keeps searching after the last sighting")
    ap.add_argument("--cmc", action="store_true",
                    help="fuse stage: shift the window by the camera pan (phase correlation)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    ap.add_argument("--max_frames", type=int, default=0, help="main stage: first N frames per sequence (smoke tests)")
    ap.add_argument("--max_crops", type=int, default=0, help="crops stage: first N test crops (smoke tests)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.stage in ("main", "fuse") and not a.src:
        ap.error("--src is required")
    if a.stage in ("fuse", "crops") and not a.spec_weights:
        ap.error("--spec_weights is required")
    if a.stage == "crops" and not a.data:
        ap.error("--data is required")
    {"main": stage_main, "fuse": stage_fuse, "report": stage_report, "crops": stage_crops}[a.stage](a)


if __name__ == "__main__":
    main()
