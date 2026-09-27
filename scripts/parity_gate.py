"""
parity_gate.py -- the section 3.5 parity gate, box for box.

IMPLEMENTATION_PLAN.md section 3.5 states the gate as "within 1 point of `.pt` on ball
recall/precision and people recall". That is the right *acceptance* criterion, but it is a
blunt instrument for diagnosis: two detectors can land within a point of each other while
disagreeing on many individual boxes, and aggregate recall cannot tell a preprocessing
error from an NMS error.

So this compares the two detectors DIRECTLY, on the same frames, box by box: detection
count, per-box IoU after one-to-one Hungarian matching, confidence delta, and class
agreement. A runner
that is numerically identical to the reference must score identically on any metric, so
passing here implies the recall/precision form of the gate without re-running the full
scoring harness for every variant.

Both the rectangular (736x1280) and square (1280x1280) exports are checked, which is the
A/B plan section 3.4 asks for.

Usage:
    python scripts/parity_gate.py --frames_per_sequence 100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

VALID_ROOT = REPO_ROOT / "data" / "SoccerNetGS" / "gamestate-2024"
DEFAULT_SEQUENCES = [
    VALID_ROOT / "valid_sample" / "SNGS-021",
    VALID_ROOT / "valid" / "SNGS-022",
    VALID_ROOT / "valid" / "SNGS-033",
    VALID_ROOT / "valid" / "SNGS-047",
    VALID_ROOT / "valid" / "SNGS-090",
]

# Gate thresholds. These are tight on purpose: the runner and the reference execute the
# same graph on the same pixels, so anything beyond float noise is a real defect.
MIN_IOU = 0.99
MAX_CONF_DELTA = 0.01


def unpack(df) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(df):
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array([list(b) for b in df["bbox_ltrb"]], dtype=float).reshape(-1, 4)
    return boxes, df["bbox_conf"].to_numpy(float), df["class_id"].to_numpy(int)


def compare_sequence(seq_dir: Path, ref, run, n_frames: int, iou_fn) -> dict:
    img_dir = seq_dir / "img1" if (seq_dir / "img1").is_dir() else seq_dir / "images"
    files = sorted(img_dir.glob("*.jpg"))[:n_frames]

    count_mismatch = 0
    min_iou, max_conf, class_disagree, total_boxes = 1.0, 0.0, 0, 0

    for i, path in enumerate(files):
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"could not read frame {path} (missing or corrupt)")
        rb, rc, rk = unpack(ref.process_sequence([frame])[0])
        ob, oc, ok = unpack(run.process_sequence([frame])[0])

        if len(rb) != len(ob):
            count_mismatch += 1
            continue
        total_boxes += len(rb)
        if not len(rb):
            continue

        # Hungarian (one-to-one optimal) assignment, not per-row argmax. Two nearby
        # reference boxes -- routine on a soccer pitch -- can both have their highest IoU
        # against the same runner box under argmax, stealing a true match and leaving
        # another runner box unmatched; that produces a spuriously low min_iou / high
        # conf-delta and a false FAIL even when the runner is numerically identical to the
        # reference. len(rb) == len(ob) is already guaranteed above, so this is a genuine
        # bijection, and for a square cost matrix linear_sum_assignment returns row_ind in
        # sorted order, so `j` can be indexed exactly as the old argmax result was.
        j = linear_sum_assignment(1.0 - iou_fn(rb, ob))[1]
        pair_iou = iou_fn(rb, ob)[np.arange(len(rb)), j]
        min_iou = min(min_iou, float(pair_iou.min()))
        max_conf = max(max_conf, float(np.abs(rc - oc[j]).max()))
        class_disagree += int((rk != ok[j]).sum())

        if (i + 1) % 25 == 0:
            print(f"    {seq_dir.name}: {i + 1}/{len(files)}", flush=True)

    return {
        "sequence": seq_dir.name,
        "frames": len(files),
        "boxes_compared": total_boxes,
        "count_mismatches": count_mismatch,
        "min_iou": round(min_iou, 6),
        "max_conf_delta": round(max_conf, 8),
        "class_disagreements": class_disagree,
        "pass": (
            count_mismatch == 0
            and class_disagree == 0
            and min_iou >= MIN_IOU
            and max_conf <= MAX_CONF_DELTA
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Box-for-box parity: runner vs Ultralytics")
    parser.add_argument("--frames_per_sequence", type=int, default=100)
    parser.add_argument("--backend", default="ort_cpu",
                        choices=["tensorrt", "ort_gpu", "ort_cpu"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", default="model_testing/yolo26s_athlete_ball/best.pt")
    parser.add_argument("--output_json", default="outputs/phase1/parity.json")
    parser.add_argument("--sequences", nargs="*", default=None)
    args = parser.parse_args()

    from gsr_pipeline.detect_athlete import AthleteBallDetector
    from gsr_pipeline.evaluate_detection import _iou_matrix, UltralyticsReferenceDetector

    sequences = [Path(s) for s in args.sequences] if args.sequences else DEFAULT_SEQUENCES
    sequences = [s for s in sequences if s.is_dir()]
    if len(sequences) < 3:
        raise SystemExit(f"need >= 3 sequences for the gate; found {len(sequences)}")

    variants = [
        ("rectangular 736x1280", "model_testing/yolo26s_athlete_ball/best_736x1280.onnx", (736, 1280)),
        ("square 1280x1280", "model_testing/yolo26s_athlete_ball/best_1280x1280.onnx", (1280, 1280)),
    ]

    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backend": args.backend,
        "thresholds": {"min_iou": MIN_IOU, "max_conf_delta": MAX_CONF_DELTA},
        "variants": [],
    }

    for label, onnx, imgsz in variants:
        if not Path(onnx).exists():
            print(f"skipping {label}: {onnx} not exported")
            continue
        print(f"\n=== {label} ({args.backend}) ===", flush=True)
        ref = UltralyticsReferenceDetector(
            weights=args.weights, imgsz=imgsz, athlete_conf=0.25, ball_conf=0.15,
            device=args.device,
        )

        # Verify the reference actually runs at the size we asked for, instead of trusting
        # that it did. Ultralytics' predictor uses LetterBox(auto=True), which pads only to
        # a stride multiple: ask it for 1280x1280 on a 1920x1080 frame and it still feeds
        # the model 736x1280. Comparing a 1280-tall ONNX against a 736-tall reference
        # produces a large, entirely spurious "failure" -- so detect it and say so, rather
        # than reporting a FAIL that implicates the runner.
        probe = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ref.model.predict(probe, imgsz=imgsz, conf=0.9, device=args.device, verbose=False)
        actual = tuple(ref.model.predictor.preprocess([probe]).shape[-2:])
        if actual != imgsz:
            print(f"  SKIPPED: Ultralytics cannot serve as a reference at {imgsz} -- its "
                  f"predictor letterboxes to {actual} regardless (LetterBox auto=True).")
            print("  Square-vs-rectangular is therefore an ACCURACY question, not a parity "
                  "one: score both against GT with evaluate_detection.py (plan 3.4).")
            report["variants"].append({
                "variant": label, "onnx": onnx, "pass": None,
                "skipped": f"no valid reference: ultralytics letterboxes to {actual}, not {imgsz}",
                "sequences": [],
            })
            continue

        run = AthleteBallDetector(
            onnx_path=onnx, athlete_conf=0.25, ball_conf=0.15,
            device=args.device, backend=args.backend,
        )
        rows = [compare_sequence(s, ref, run, args.frames_per_sequence, _iou_matrix) for s in sequences]
        run.close()
        report["variants"].append({"variant": label, "onnx": onnx, "sequences": rows,
                                   "pass": all(r["pass"] for r in rows)})

        print(f"\n{'sequence':<12}{'frames':>7}{'boxes':>8}{'cnt!=':>7}{'minIoU':>10}"
              f"{'maxDconf':>12}{'cls!=':>7}  result")
        for r in rows:
            print(f"{r['sequence']:<12}{r['frames']:>7}{r['boxes_compared']:>8}"
                  f"{r['count_mismatches']:>7}{r['min_iou']:>10.6f}"
                  f"{r['max_conf_delta']:>12.8f}{r['class_disagreements']:>7}"
                  f"  {'PASS' if r['pass'] else 'FAIL'}")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    scored = [v for v in report["variants"] if v["pass"] is not None]
    overall = bool(scored) and all(v["pass"] for v in scored)
    print(f"\nwrote {out}")
    print("GATE: " + ("PASS" if overall else "FAIL"))
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
