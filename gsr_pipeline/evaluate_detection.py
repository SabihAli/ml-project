"""
evaluate_detection.py — per-class detector accuracy against SoccerNet-GSR labels.

GS-HOTA (evaluate.py) folds detection, association, localization and identity
into one number, which hides how well each detector class performs on its own.
This harness isolates the detector (yolov8n_gsr.pt, configs/default.yaml's
`detector` section) and reports:

  * per class (player / goalkeeper / referee / ball): recall, precision, F1,
    matched one-to-one per frame (Hungarian). People match on IoU; the ball
    matches on centre distance in pixels, because a ~10-20px box makes IoU
    collapse under one or two pixels of localization error.
  * a class-agnostic person confusion matrix (GT role -> predicted role), so a
    referee detected as a player shows up as a role error, not as a miss.
  * ball metrics that downstream possession/touch/pass logic actually depends
    on: fraction of ball-labelled frames where the ball was found, frames with
    more than one ball detection, and the longest consecutive run of frames
    with the ball labelled but not found.

Frames are read in batches, not all loaded into memory up front
(utils.load_frames_from_dir would hold ~4.6 GB for a 750-frame 1080p sequence).

Usage:
    python -m gsr_pipeline.evaluate_detection \\
        --sequence_dir data/SoccerNetGS/gamestate-2024/valid_sample/SNGS-021 \\
        --output_json outputs/SNGS-021_detection_eval.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from .detect import YOLOv8nDetector

# Path A: the 4-class yolov8n_gsr detector, whose class_id IS the role.
CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
PERSON_ROLES = ("player", "goalkeeper", "referee")

# Path B: the 2-class athlete/ball detector (docs/DETECTOR_AND_DATA_BLOCKERS.md 2026-09-23).
# It emits no roles at all -- player/goalkeeper/referee are recovered downstream by
# athlete_roles.py -- so scoring folds every GT person role into one "athlete" class and
# compares class-agnostically, which is how the blockers doc's section 2 table was built.
# Ball scoring is unchanged: centre distance, because a ~10-25px box makes IoU collapse.
CLASS_NAMES_B = {0: "athlete", 1: "ball"}
ATHLETE = "athlete"


def load_gt_boxes(labels_path: Path) -> Tuple[List[Tuple[str, str]], Dict[str, List[dict]]]:
    """Returns ([(image_id, file_name), ...] in frame order, {image_id: [box, ...]}).

    Each box is {"role", "ltrb"}. Annotations without an image bbox or a known
    role (e.g. the per-frame pitch/camera entries) are skipped.
    """
    with open(labels_path, "r") as f:
        data = json.load(f)

    images = sorted(
        ((img["image_id"], img["file_name"]) for img in data["images"] if img.get("is_labeled", True)),
        key=lambda pair: pair[1],
    )
    boxes: Dict[str, List[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        role = (ann.get("attributes") or {}).get("role")
        bb = ann.get("bbox_image")
        if role not in CLASS_NAMES.values() or not isinstance(bb, dict):
            continue
        boxes[ann["image_id"]].append(
            {"role": role, "ltrb": (bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"])}
        )
    return images, boxes


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) ltrb arrays."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _centre_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ca = (a[:, :2] + a[:, 2:]) / 2
    cb = (b[:, :2] + b[:, 2:]) / 2
    return np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)


def match(gt: np.ndarray, pred: np.ndarray, use_centre_distance: bool, threshold: float) -> List[Tuple[int, int]]:
    """One-to-one Hungarian matching; returns accepted (gt_idx, pred_idx) pairs."""
    if len(gt) == 0 or len(pred) == 0:
        return []
    if use_centre_distance:
        cost = _centre_distance_matrix(gt, pred)
        valid = cost <= threshold
    else:
        iou = _iou_matrix(gt, pred)
        cost = 1.0 - iou
        valid = iou >= threshold
    rows, cols = linear_sum_assignment(np.where(valid, cost, 1e6))
    return [(r, c) for r, c in zip(rows, cols) if valid[r, c]]


def _ltrb_array(boxes: List[tuple]) -> np.ndarray:
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def _longest_run(flags: List[bool]) -> int:
    best = cur = 0
    for flag in flags:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best


def evaluate(sequence_dir: Path, detector, person_iou: float, ball_match_px: float,
             max_frames: int | None, fps: float, two_class: bool = False,
             start_frame: int = 0) -> dict:
    """Score a detector against a sequence's GT.

    Args:
        two_class: Path B mode. Detector classes are {athlete, ball} and every GT person
            role is folded into "athlete", so people are scored class-agnostically. The
            role-confusion matrix is omitted, because a detector with no roles cannot
            confuse them -- that measurement moves to evaluate_athlete_roles.py.
        start_frame: index of the first frame to score. The blockers doc's ball figures are
            open-play only (frames 168-749 on SNGS-021): the kickoff has the ball static,
            white-on-white and half-occluded, and is missed by every model at every
            threshold, so including it makes two detectors look equally bad and hides the
            difference the gate is trying to measure.
    """
    images, gt_by_image = load_gt_boxes(sequence_dir / "Labels-GameState.json")
    img_dir = sequence_dir / "img1" if (sequence_dir / "img1").is_dir() else sequence_dir / "images"
    if start_frame:
        images = images[start_frame:]
    if max_frames:
        images = images[:max_frames]

    class_names = CLASS_NAMES_B if two_class else CLASS_NAMES
    counts = {role: Counter() for role in class_names.values()}  # tp / fp / fn per class
    confusion: Dict[str, Counter] = {role: Counter() for role in PERSON_ROLES}
    ball_labelled, ball_found, ball_missed_run_flags = 0, 0, []
    multi_ball_frames, frames_with_ball_detection = 0, 0
    ball_recall_by_size = {"<10px": Counter(), "10-20px": Counter(), ">=20px": Counter()}

    for start in range(0, len(images), detector.batch_size):
        chunk = images[start:start + detector.batch_size]
        frames = []
        for _, file_name in chunk:
            frame = cv2.imread(str(img_dir / file_name))
            if frame is None:
                raise FileNotFoundError(
                    f"could not read frame {img_dir / file_name} (missing or corrupt); "
                    "cv2.imread returned None"
                )
            frames.append(frame)
        for (image_id, file_name), frame, det_df in zip(chunk, frames, detector.process_sequence(frames)):
            gts = gt_by_image.get(image_id, [])
            if two_class:
                # Fold player/goalkeeper/referee into one class; the ball stays its own.
                gts = [
                    {**b, "role": ATHLETE if b["role"] in PERSON_ROLES else b["role"]}
                    for b in gts
                ]
            preds = [
                {"role": class_names[int(row["class_id"])], "ltrb": row["bbox_ltrb"]}
                for _, row in det_df.iterrows()
            ] if len(det_df) else []

            # Per-class, class-aware matching.
            for role in class_names.values():
                g = _ltrb_array([b["ltrb"] for b in gts if b["role"] == role])
                p = _ltrb_array([b["ltrb"] for b in preds if b["role"] == role])
                is_ball = role == "ball"
                pairs = match(g, p, use_centre_distance=is_ball, threshold=ball_match_px if is_ball else person_iou)
                counts[role]["tp"] += len(pairs)
                counts[role]["fn"] += len(g) - len(pairs)
                counts[role]["fp"] += len(p) - len(pairs)

                if is_ball:
                    matched_gt = {gi for gi, _ in pairs}
                    for gi, box in enumerate(g):
                        size = max(box[2] - box[0], box[3] - box[1])
                        bucket = "<10px" if size < 10 else ("10-20px" if size < 20 else ">=20px")
                        ball_recall_by_size[bucket]["gt"] += 1
                        ball_recall_by_size[bucket]["tp"] += int(gi in matched_gt)
                    if len(g):
                        ball_labelled += 1
                        ball_found += int(bool(pairs))
                        ball_missed_run_flags.append(not pairs)
                    if len(p):
                        frames_with_ball_detection += 1
                    if len(p) > 1:
                        multi_ball_frames += 1

            # Class-agnostic person matching -> role confusion. Path B's detector has no
            # roles to confuse, so this is skipped there (see evaluate_athlete_roles.py).
            if two_class:
                continue
            gp = [b for b in gts if b["role"] in PERSON_ROLES]
            pp = [b for b in preds if b["role"] in PERSON_ROLES]
            pairs = match(_ltrb_array([b["ltrb"] for b in gp]), _ltrb_array([b["ltrb"] for b in pp]),
                          use_centre_distance=False, threshold=person_iou)
            matched_gt = {gi for gi, _ in pairs}
            for gi, pi in pairs:
                confusion[gp[gi]["role"]][pp[pi]["role"]] += 1
            for gi, box in enumerate(gp):
                if gi not in matched_gt:
                    confusion[box["role"]]["missed"] += 1

        print(f"  processed {min(start + detector.batch_size, len(images))}/{len(images)} frames", flush=True)

    per_class = {}
    for role, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        recall = tp / (tp + fn) if tp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        # `is not None`, not truthiness: a total miss legitimately has recall=0.0 and
        # precision=0.0, and `0.0 and 0.0` is falsy -- the old check printed "n/a" for a
        # real, computable 0.0% F1 instead of reporting the miss. F1 is undefined only when
        # recall/precision themselves are undefined (no GT and no predictions for the
        # class); when both are 0.0 the harmonic mean is 0.0 by convention, not 0/0.
        if recall is None or precision is None:
            f1 = None
        elif recall == 0.0 and precision == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * recall * precision / (recall + precision)
        per_class[role] = {"gt": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                           "recall": recall, "precision": precision, "f1": f1}

    longest_miss = _longest_run(ball_missed_run_flags)
    # Path B's detector names its thresholds after its classes, not after roles.
    person_conf = getattr(detector, "conf_threshold", None)
    if person_conf is None:
        person_conf = getattr(detector, "athlete_conf", None)
    ball_conf = getattr(detector, "ball_conf_threshold", None)
    if ball_conf is None:
        ball_conf = getattr(detector, "ball_conf", None)

    return {
        "sequence": sequence_dir.name,
        "frames_evaluated": len(images),
        "frame_range": {"start": start_frame, "end": start_frame + len(images)},
        "mode": "2-class (athlete/ball)" if two_class else "4-class (player/gk/referee/ball)",
        "backend": getattr(detector, "backend", "ultralytics"),
        "thresholds": {"person_iou": person_iou, "ball_match_px": ball_match_px,
                       "conf_threshold": person_conf,
                       "ball_conf_threshold": ball_conf},
        "per_class": per_class,
        "person_role_confusion": (
            None if two_class else {gt: dict(c) for gt, c in confusion.items()}
        ),
        "ball": {
            "frames_with_ball_labelled": ball_labelled,
            "frame_recall": ball_found / ball_labelled if ball_labelled else None,
            "frames_with_any_ball_detection": frames_with_ball_detection,
            "frames_with_multiple_ball_detections": multi_ball_frames,
            "longest_consecutive_miss_frames": longest_miss,
            "longest_consecutive_miss_seconds": longest_miss / fps,
            "recall_by_gt_box_size": {
                bucket: {"gt": c["gt"], "recall": c["tp"] / c["gt"] if c["gt"] else None}
                for bucket, c in ball_recall_by_size.items()
            },
        },
    }


def _fmt(value) -> str:
    return "  n/a " if value is None else f"{value * 100:5.1f}%"


def print_report(report: dict) -> None:
    t = report["thresholds"]
    rng = report.get("frame_range", {})
    span = f" frames {rng.get('start')}-{rng.get('end')}" if rng.get("start") else ""
    print(f"\nDetection evaluation — {report['sequence']} ({report['frames_evaluated']} frames{span})")
    print(f"mode: {report.get('mode', 'n/a')}   backend: {report.get('backend', 'n/a')}")
    print(f"person match: IoU >= {t['person_iou']}   ball match: centre distance <= {t['ball_match_px']}px   "
          f"conf: {t['conf_threshold']} (ball {t['ball_conf_threshold']})\n")
    print(f"{'class':<11}{'GT':>7}{'TP':>7}{'FP':>7}{'FN':>7}   {'recall':>7}  {'precision':>9}  {'F1':>7}")
    for role, m in report["per_class"].items():
        print(f"{role:<11}{m['gt']:>7}{m['tp']:>7}{m['fp']:>7}{m['fn']:>7}   {_fmt(m['recall'])}  "
              f"{_fmt(m['precision']):>9}  {_fmt(m['f1'])}")

    if report.get("person_role_confusion"):
        print("\nperson role confusion (rows = GT role, class-agnostic IoU match):")
        for gt_role, row in report["person_role_confusion"].items():
            print(f"  {gt_role:<11} " + "  ".join(f"{k}={v}" for k, v in sorted(row.items())))

    b = report["ball"]
    print("\nball:")
    print(f"  frames with ball labelled          {b['frames_with_ball_labelled']}")
    print(f"  frame-level recall                 {_fmt(b['frame_recall'])}")
    print(f"  frames with >1 ball detection      {b['frames_with_multiple_ball_detections']}")
    print(f"  longest consecutive miss           {b['longest_consecutive_miss_frames']} frames "
          f"({b['longest_consecutive_miss_seconds']:.2f}s)")
    for bucket, m in b["recall_by_gt_box_size"].items():
        print(f"  recall, GT box {bucket:<8}          {_fmt(m['recall'])}  (n={m['gt']})")


class UltralyticsReferenceDetector:
    """The `.pt` through Ultralytics itself -- the reference side of the parity gate.

    Exists so both sides of the comparison go through one scoring harness. It deliberately
    uses the TRAINING config's iou=0.7 and max_det=300 (model_testing/.../args.yaml), NOT
    detect.py's Path A iou_threshold=0.45: comparing a runner tuned to one NMS setting
    against a reference tuned to another measures the settings, not the runner.
    """

    backend = "ultralytics"

    def __init__(self, weights: str, imgsz: tuple[int, int], athlete_conf: float,
                 ball_conf: float, device: str = "cpu", batch_size: int = 8,
                 iou: float = 0.7, max_det: int = 300):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.athlete_conf = athlete_conf
        self.ball_conf = ball_conf
        self.device = device
        self.batch_size = batch_size
        self.iou = iou
        self.max_det = max_det

    def process_sequence(self, frames):
        import pandas as pd

        # Run at the lower of the two floors, then apply each class's own floor, exactly as
        # AthleteBallDetector does -- otherwise the ball is cut at the athlete threshold.
        results = self.model.predict(
            frames, imgsz=self.imgsz, conf=min(self.athlete_conf, self.ball_conf),
            iou=self.iou, max_det=self.max_det, device=self.device, verbose=False,
        )
        out = []
        for r in results:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            keep = np.where(cls == 1, conf > self.ball_conf, conf > self.athlete_conf)
            xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
            if not len(xyxy):
                out.append(pd.DataFrame(columns=["bbox_ltrb", "bbox_conf", "class_id"]))
                continue
            ltrb = [tuple(float(v) for v in b) for b in xyxy]
            out.append(pd.DataFrame({"bbox_ltrb": ltrb, "bbox_conf": conf.astype(float),
                                     "class_id": cls}))
        return out


def _build_path_b_detector(args):
    """Construct either the reference (.pt via Ultralytics) or the runner under test."""
    onnx_path = Path(args.onnx)
    # The ONNX filename carries the input size the parity gate is A/B-ing (736x1280 vs
    # 1280x1280), so the reference is run at the same size rather than at a default.
    try:
        h, w = (int(v) for v in onnx_path.stem.rsplit("_", 1)[-1].split("x"))
    except ValueError:
        h, w = 736, 1280

    if args.backend == "ultralytics":
        return UltralyticsReferenceDetector(
            weights=args.weights, imgsz=(h, w), athlete_conf=args.athlete_conf,
            ball_conf=args.ball_conf, device=args.device or "cpu",
        )

    from .detect_athlete import AthleteBallDetector

    return AthleteBallDetector(
        onnx_path=str(onnx_path),
        athlete_conf=args.athlete_conf,
        ball_conf=args.ball_conf,
        device=args.device or ("cpu" if args.backend == "ort_cpu" else "cuda:0"),
        backend=args.backend,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-class detector accuracy against SoccerNet-GSR labels")
    parser.add_argument("--sequence_dir", required=True, help="Sequence folder containing img1/ and Labels-GameState.json")
    parser.add_argument("--config", default="gsr_pipeline/configs/default.yaml")
    parser.add_argument("--device", default=None, help="Override detector device (cpu | cuda)")
    parser.add_argument("--person_iou", type=float, default=0.5)
    parser.add_argument("--ball_match_px", type=float, default=15.0,
                        help="Max centre distance (px, 1080p frame) for a ball detection to count as a hit")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--start_frame", type=int, default=0,
                        help="First frame to score. Use 168 on SNGS-021 to reproduce the "
                             "open-play ball numbers in DETECTOR_AND_DATA_BLOCKERS.md 2 "
                             "(the kickoff is missed by every model and swamps the ball metrics)")
    parser.add_argument("--fps", type=float, default=25.0, help="Sequence frame rate, for miss-duration reporting")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--path", choices=["a", "b"], default="a",
                        help="a: the 4-class yolov8n_gsr detector. b: the 2-class "
                             "athlete/ball detector plus class-agnostic person scoring")
    parser.add_argument("--backend", default=None,
                        choices=["tensorrt", "ort_gpu", "ort_cpu", "ultralytics"],
                        help="Path B only. Pin one backend instead of walking the fallback "
                             "chain. 'ultralytics' runs the .pt through Ultralytics itself, "
                             "which is the reference side of the parity gate (plan 3.5)")
    parser.add_argument("--onnx", default="model_testing/yolo26s_athlete_ball/best_736x1280.onnx",
                        help="Path B: the exported ONNX to score")
    parser.add_argument("--weights", default="model_testing/yolo26s_athlete_ball/best.pt",
                        help="Path B: the .pt, used only by --backend ultralytics")
    parser.add_argument("--athlete_conf", type=float, default=0.25)
    parser.add_argument("--ball_conf", type=float, default=0.15)
    args = parser.parse_args()

    sequence_dir = Path(args.sequence_dir)
    if not (sequence_dir / "Labels-GameState.json").exists():
        print(f"No Labels-GameState.json in {sequence_dir}")
        return

    two_class = args.path == "b"
    if two_class:
        detector = _build_path_b_detector(args)
    else:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        det_cfg = dict(cfg.get("detector", {}))
        if args.device:
            det_cfg["device"] = args.device
        detector = YOLOv8nDetector(**det_cfg)

    report = evaluate(sequence_dir, detector, args.person_iou, args.ball_match_px,
                      args.max_frames, args.fps, two_class=two_class,
                      start_frame=args.start_frame)
    print_report(report)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
