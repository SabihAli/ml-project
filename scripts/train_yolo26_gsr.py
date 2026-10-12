"""
train_yolo26_gsr.py
====================
Self-contained Kaggle-ready script to fine-tune YOLO26 (nano tier, base + P2
small-object head) on SoccerNet-GSR, following the two-stage plan agreed for
cv_service's ball-detection problem:

  Stage 1: fine-tune plain `yolo26n.pt` (fully pretrained, drop-in, cheaper
           than the current `yolov8n_gsr.pt` at 6.2 GFLOPs vs 8.9 GFLOPs).
  Stage 2: fine-tune `yolo26n-p2` (adds a stride-4 detection head for small
           objects -- the ball is ~10-25px, the actual target of this
           exercise). Its backbone is loaded from Stage 1's trained weights
           (identical backbone between the two architectures, see
           cv_service/docs/path_b/DETECTOR_AND_DATA_BLOCKERS.md); only the new P2
           branch + 4-way detect head start cold. Cost: +8% GFLOPs over
           Stage 1, not the +22%/+215% you'd pay going to the `s` size.

Both stages use the SAME 4-class mapping already baked into cv_service's
detect.py (player=0, goalkeeper=1, referee=2, ball=3) so either checkpoint is
a drop-in replacement for yolov8n_gsr.pt without touching downstream code.

No cv_service import: this file is standalone on purpose so it can be pulled
onto Kaggle (which doesn't have cv_service's other GPU deps -- torchreid,
easyocr, etc.) and run immediately. It re-implements the small pieces it
needs (GSR label conversion, per-class + ball eval) itself.

--------------------------------------------------------------------------
Kaggle usage
--------------------------------------------------------------------------
1. Attach the SoccerNet-GSR dataset (gamestate-2024) as a Kaggle Dataset to
   the notebook. This script auto-searches common mount points
   (/kaggle/input/*) for folders containing `Labels-GameState.json`; pass
   --data_root explicitly if auto-detection doesn't find it.
2. Turn on a GPU accelerator (Settings -> Accelerator -> GPU T4 x2 or P100).
3. In a notebook cell:
       !pip install -q -U "ultralytics>=8.4.0"   # YOLO26 needs >=8.4.0
       !python train_yolo26_gsr.py --mode all
   (the script also self-checks/upgrades ultralytics on startup, so the
   explicit pip install line is a speed-up, not a requirement).
4. Everything is written under --output_dir (default: /kaggle/working/gsr_yolo26
   when running on Kaggle, else ./runs/gsr_yolo26). Stage 1 weights:
   <output_dir>/yolo26n_stage1/weights/best.pt. Stage 2:
   <output_dir>/yolo26n_p2_stage2/weights/best.pt.
5. Re-running with --mode all is safe/idempotent: label conversion is
   skipped for sequences already converted, and training resumes from
   existing best.pt only if you explicitly pass --resume.

--------------------------------------------------------------------------
Modes (--mode)
--------------------------------------------------------------------------
    convert      : SoccerNet-GSR JSON -> YOLO .txt labels (in-place, per sequence)
    build_yaml   : write the dataset YAML ultralytics trains against
    train_stage1 : fine-tune yolo26n.pt
    train_stage2 : fine-tune yolo26n-p2, backbone transferred from stage 1
    evaluate     : per-class + ball-specific recall/precision for both stages
                   (and --baseline_weights if given, e.g. yolov8n_gsr.pt)
    all          : convert -> build_yaml -> train_stage1 -> train_stage2 -> evaluate
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────
# Class mapping -- MUST stay identical to cv_service/gsr_pipeline/detect.py's
# _PLAYER_CLASS/_GK_CLASS/_REFEREE_CLASS/_BALL_CLASS, or a checkpoint trained
# here silently corrupts every downstream role decision if ever swapped in.
# ──────────────────────────────────────────────────────────────────────────
ROLE_TO_ID = {"player": 0, "goalkeeper": 1, "referee": 2, "ball": 3, "other": 4}
ID_TO_ROLE = {v: k for k, v in ROLE_TO_ID.items() if k != "other"}
CAT_ID_TO_ROLE = {1: "player", 2: "goalkeeper", 3: "referee", 4: "ball", 7: "other"}
KNOWN_SPLITS = ("train", "valid", "valid_sample", "test")


# ══════════════════════════════════════════════════════════════════════════
# Environment bootstrap
# ══════════════════════════════════════════════════════════════════════════

def ensure_ultralytics() -> None:
    """YOLO26 needs ultralytics>=8.4.0. Kaggle's preinstalled version is
    usually older; upgrade in-place rather than failing with an import error
    partway through a long notebook session."""
    try:
        import ultralytics
        major, minor, *_ = (int(p) for p in ultralytics.__version__.split(".")[:2] + ["0"])
        if (major, minor) >= (8, 4):
            print(f"ultralytics {ultralytics.__version__} OK")
            return
    except ImportError:
        pass
    print("Installing/upgrading ultralytics>=8.4.0 (needed for YOLO26)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics>=8.4.0"])
    import importlib
    import ultralytics
    importlib.reload(ultralytics)
    print(f"ultralytics {ultralytics.__version__} installed")


def default_output_dir() -> Path:
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / "gsr_yolo26"
    return Path("runs/gsr_yolo26")


def find_data_root(explicit: Optional[str]) -> Path:
    """Locate the SoccerNet-GSR root: the directory whose children are the
    split folders (train/valid/...), each containing SNGS-xxx sequence dirs.
    Auto-search covers Kaggle's /kaggle/input/<dataset-slug>/... mount and a
    couple of local layouts; --data_root overrides everything."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [Path.cwd()]
    if Path("/kaggle/input").is_dir():
        candidates += sorted(Path("/kaggle/input").iterdir())
    candidates += [Path("GSR_Subset/GSR_Subset_65"), Path("data/datasets/SoccerNetGS/gamestate-2024")]

    for root in candidates:
        if not root.is_dir():
            continue
        # direct hit: root/<split>/<seq>/Labels-GameState.json
        if any((root / split).is_dir() for split in KNOWN_SPLITS):
            return root
        # one level deeper, e.g. root/gamestate-2024/<split>/...
        for child in root.iterdir():
            if child.is_dir() and any((child / split).is_dir() for split in KNOWN_SPLITS):
                return child

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not auto-locate the SoccerNet-GSR root (looked for a folder "
        f"containing train/valid/valid_sample/test subfolders). Searched: "
        f"{searched}. Pass --data_root explicitly, e.g. "
        f"--data_root /kaggle/input/soccernet-gsr/gamestate-2024"
    )


def discover_splits(data_root: Path) -> Dict[str, List[Path]]:
    """Returns {split_name: [sequence_dir, ...]} for every known split
    present under data_root, e.g. {'train': [.../SNGS-060, ...], 'valid': [...]}."""
    splits: Dict[str, List[Path]] = {}
    for split in KNOWN_SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        seqs = sorted(d for d in split_dir.iterdir() if d.is_dir() and (d / "Labels-GameState.json").exists())
        if seqs:
            splits[split] = seqs
    if not splits:
        raise FileNotFoundError(f"No split folders with Labels-GameState.json found under {data_root}")
    return splits


# ══════════════════════════════════════════════════════════════════════════
# Phase 1: SoccerNet-GSR JSON -> YOLO .txt labels (per sequence, idempotent)
# ══════════════════════════════════════════════════════════════════════════

def convert_sequence(seq_dir: Path) -> Tuple[int, int, int]:
    """Returns (images_processed, boxes_kept, boxes_dropped). Skips a
    sequence whose labels/ dir already has one .txt per labelled image
    (safe to re-run across a multi-session Kaggle notebook)."""
    label_json = seq_dir / "Labels-GameState.json"
    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    img1_dir = seq_dir / "img1"
    images_dir = seq_dir / "images"
    if img1_dir.exists() and not images_dir.exists():
        img1_dir.rename(images_dir)
    if not images_dir.exists():
        print(f"  [SKIP] no image directory in {seq_dir}")
        return (0, 0, 0)

    labeled_images = [img for img in data.get("images", []) if img.get("is_labeled", True)]
    labels_dir = seq_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    if len(list(labels_dir.glob("*.txt"))) == len(labeled_images) and labeled_images:
        return (len(labeled_images), -1, -1)  # already converted, -1 = "not recomputed"

    img_info = {img["image_id"]: (img["file_name"], img.get("width", 1920), img.get("height", 1080))
                for img in data.get("images", [])}

    boxes_by_image: Dict[str, list] = defaultdict(list)
    kept, dropped = 0, 0
    for ann in data.get("annotations", []):
        if ann.get("supercategory") != "object":
            dropped += 1
            continue
        bbox = ann.get("bbox_image")
        if not bbox:
            dropped += 1
            continue
        boxes_by_image[ann["image_id"]].append(ann)
        kept += 1

    for img_id, anns in boxes_by_image.items():
        file_name, img_w, img_h = img_info.get(img_id, (f"{img_id}.jpg", 1920, 1080))
        stem = Path(file_name).stem
        lines = []
        for ann in anns:
            bbox = ann["bbox_image"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
            nw, nh = w / img_w, h / img_h
            cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))

            role = (ann.get("attributes") or {}).get("role")
            if not role:
                role = CAT_ID_TO_ROLE.get(ann.get("category_id"), "other")
            class_id = ROLE_TO_ID.get(role, 4)
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return (len(labeled_images), kept, dropped)


def convert_all(splits: Dict[str, List[Path]]) -> None:
    for split, seqs in splits.items():
        print(f"\nConverting {split} split ({len(seqs)} sequences)...")
        for seq in seqs:
            n_img, kept, dropped = convert_sequence(seq)
            status = "already converted" if kept == -1 else f"{kept} boxes kept, {dropped} dropped"
            print(f"  {seq.name}: {n_img} images, {status}")


# ══════════════════════════════════════════════════════════════════════════
# Phase 2: dataset YAML
# ══════════════════════════════════════════════════════════════════════════

LABEL_MODE_NAMES = {
    "4cls": ID_TO_ROLE,
    "athlete": {0: "athlete"},
    "athlete_ball": {0: "athlete", 1: "ball"},
}


def apply_label_mode(splits: Dict[str, List[Path]], mode: str) -> None:
    """Rewrite labels/*.txt for the chosen class scheme. The 4-class labels are
    backed up once to labels_4cls/ and every mode is derived from that backup,
    so switching modes (or re-running) is safe."""
    remap = {
        "4cls": {0: 0, 1: 1, 2: 2, 3: 3},
        "athlete": {0: 0, 1: 0, 2: 0},
        "athlete_ball": {0: 0, 1: 0, 2: 0, 3: 1},
    }[mode]
    n_files = 0
    for seqs in splits.values():
        for seq in seqs:
            labels_dir, backup = seq / "labels", seq / "labels_4cls"
            if not labels_dir.is_dir():
                continue
            if not backup.exists():
                shutil.copytree(labels_dir, backup)
            (seq / "labels.cache").unlink(missing_ok=True)
            for src in backup.glob("*.txt"):
                out = []
                for line in src.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if parts and int(parts[0]) in remap:
                        out.append(" ".join([str(remap[int(parts[0])])] + parts[1:]))
                (labels_dir / src.name).write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
                n_files += 1
    print(f"Label mode '{mode}': rewrote {n_files} label files")


def build_dataset_yaml(splits: Dict[str, List[Path]], yaml_out: Path,
                       train_stride: int = 1, val_stride: int = 1, label_mode: str = "4cls") -> Path:
    import yaml

    def image_dirs(seqs: List[Path], stride: int, list_name: str):
        dirs = []
        for seq in seqs:
            d = seq / "images" if (seq / "images").is_dir() else seq / "img1"
            if d.is_dir():
                dirs.append(d)
        if stride <= 1:
            return [str(d.resolve()) for d in dirs]
        # Ultralytics maps image->label by swapping '/images/' for '/labels/', so keep real paths
        files: List[str] = []
        for d in dirs:
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            files += [str(p.resolve()) for p in imgs[::stride]]
        list_path = yaml_out.parent / list_name
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(files) + "\n", encoding="utf-8")
        print(f"  {list_name}: {len(files)} images (every {stride}th frame)")
        return str(list_path)

    train_seqs = splits.get("train", [])
    val_seqs = splits.get("valid") or splits.get("valid_sample") or []
    if not train_seqs:
        raise ValueError("No 'train' split found -- cannot build a training YAML.")
    if not val_seqs:
        print("WARNING: no 'valid'/'valid_sample' split found; using a slice of train for validation instead.")
        val_seqs = train_seqs[: max(1, len(train_seqs) // 10)]

    dataset = {
        "train": image_dirs(train_seqs, train_stride, "train_list.txt"),
        "val": image_dirs(val_seqs, val_stride, "val_list.txt"),
        "nc": len(LABEL_MODE_NAMES[label_mode]),
        "names": LABEL_MODE_NAMES[label_mode],
    }
    apply_label_mode(splits, label_mode)
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_out, "w", encoding="utf-8") as f:
        yaml.dump(dataset, f, default_flow_style=False, sort_keys=False)
    print(f"Dataset YAML written -> {yaml_out} ({len(train_seqs)} train seqs, {len(val_seqs)} val seqs)")
    return yaml_out


# ══════════════════════════════════════════════════════════════════════════
# Phase 3/4: training
# ══════════════════════════════════════════════════════════════════════════

def train_stage1(yaml_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("STAGE 1: fine-tuning yolo26n.pt (fully pretrained, drop-in)")
    print("=" * 70)
    run_name = args.run_name or "yolo26n_stage1"
    last = output_dir / run_name / "weights" / "last.pt"
    if args.resume and last.exists():
        YOLO(str(last)).train(resume=True)
    else:
        model = YOLO(args.init_weights or "yolo26n.pt")
        model.train(
            data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            patience=args.patience, workers=args.workers, device=args.device,
            project=str(output_dir), name=run_name, exist_ok=True,
        )
    best = output_dir / run_name / "weights" / "best.pt"
    print(f"Stage 1 done. Best weights: {best}")
    return best


def train_stage2(yaml_path: Path, stage1_weights: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("STAGE 2: fine-tuning yolo26n-p2 (backbone transferred from stage 1)")
    print("=" * 70)
    if not stage1_weights.exists():
        raise FileNotFoundError(
            f"Stage 1 weights not found at {stage1_weights} -- run --mode train_stage1 first "
            f"(or pass --stage1_weights to point at an existing best.pt)."
        )
    last = output_dir / "yolo26n_p2_stage2" / "weights" / "last.pt"
    if args.resume and last.exists():
        YOLO(str(last)).train(resume=True)
    else:
        model = YOLO("yolo26n-p2.yaml").load(str(stage1_weights))
        model.train(
            data=str(yaml_path), epochs=args.epochs_stage2, imgsz=args.imgsz, batch=args.batch,
            patience=args.patience, workers=args.workers, device=args.device,
            project=str(output_dir), name="yolo26n_p2_stage2", exist_ok=True,
        )
    best = output_dir / "yolo26n_p2_stage2" / "weights" / "best.pt"
    print(f"Stage 2 done. Best weights: {best}")
    return best


# ══════════════════════════════════════════════════════════════════════════
# Phase 5: evaluation (self-contained -- mirrors
# cv_service/gsr_pipeline/evaluate_detection.py's ball/per-class metrics
# without importing cv_service, since Kaggle doesn't have its other deps)
# ══════════════════════════════════════════════════════════════════════════

def load_gt(labels_path: Path) -> Tuple[List[Tuple[str, str]], Dict[str, List[dict]]]:
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = sorted(
        ((img["image_id"], img["file_name"]) for img in data["images"] if img.get("is_labeled", True)),
        key=lambda pair: pair[1],
    )
    boxes: Dict[str, List[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        role = (ann.get("attributes") or {}).get("role")
        bb = ann.get("bbox_image")
        if role not in ID_TO_ROLE.values() or not isinstance(bb, dict):
            continue
        boxes[ann["image_id"]].append({"role": role, "ltrb": (bb["x"], bb["y"], bb["x"] + bb["w"], bb["y"] + bb["h"])})
    return images, boxes


def evaluate_weights(weights_path: Path, eval_seqs: List[Path], person_iou: float, ball_px: float,
                      conf: float, ball_conf: float, imgsz: int = 640) -> None:
    import cv2
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from ultralytics import YOLO

    print(f"\n--- Evaluating {weights_path} ---")
    model = YOLO(str(weights_path))
    per_class = {r: {"tp": 0, "fp": 0, "fn": 0} for r in ID_TO_ROLE.values()}
    ball_frames_labelled = ball_frames_found = 0

    for seq_dir in eval_seqs:
        images, gt_by_image = load_gt(seq_dir / "Labels-GameState.json")
        img_dir = seq_dir / "images" if (seq_dir / "images").is_dir() else seq_dir / "img1"

        for img_id, fname in images:
            if not (img_dir / fname).exists():
                continue
            frame = cv2.imread(str(img_dir / fname))
            if frame is None:
                continue
            result = model(frame, conf=min(conf, ball_conf), imgsz=imgsz, verbose=False)[0]

            gt_boxes = gt_by_image.get(img_id, [])
            has_ball_gt = any(b["role"] == "ball" for b in gt_boxes)
            if has_ball_gt:
                ball_frames_labelled += 1

            preds_by_role: Dict[str, List["np.ndarray"]] = defaultdict(list)
            found_ball = False
            for box in result.boxes:
                cls_id = int(box.cls[0])
                role = ID_TO_ROLE.get(cls_id)
                if role is None:
                    continue
                c = float(box.conf[0])
                if c < (ball_conf if role == "ball" else conf):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                preds_by_role[role].append(np.array([x1, y1, x2, y2]))
                if role == "ball":
                    found_ball = True
            if has_ball_gt and found_ball:
                ball_frames_found += 1

            for role in ID_TO_ROLE.values():
                gt = np.array([b["ltrb"] for b in gt_boxes if b["role"] == role])
                pred = np.array(preds_by_role.get(role, []))
                if len(gt) == 0 and len(pred) == 0:
                    continue
                if len(gt) == 0:
                    per_class[role]["fp"] += len(pred)
                    continue
                if len(pred) == 0:
                    per_class[role]["fn"] += len(gt)
                    continue
                if role == "ball":
                    ca = (gt[:, :2] + gt[:, 2:]) / 2
                    cb = (pred[:, :2] + pred[:, 2:]) / 2
                    dist = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
                    cost, threshold, better = dist, ball_px, (lambda v, t: v <= t)
                else:
                    lt = np.maximum(gt[:, None, :2], pred[None, :, :2])
                    rb = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
                    inter = np.clip(rb - lt, 0, None).prod(axis=2)
                    area_g = (gt[:, 2:] - gt[:, :2]).prod(axis=1)
                    area_p = (pred[:, 2:] - pred[:, :2]).prod(axis=1)
                    iou = inter / (area_g[:, None] + area_p[None, :] - inter + 1e-9)
                    cost, threshold, better = -iou, -person_iou, (lambda v, t: v <= t)

                row_ind, col_ind = linear_sum_assignment(cost)
                matched_gt, matched_pred = set(), set()
                for r, c in zip(row_ind, col_ind):
                    if better(cost[r, c], threshold):
                        matched_gt.add(r)
                        matched_pred.add(c)
                per_class[role]["tp"] += len(matched_gt)
                per_class[role]["fn"] += len(gt) - len(matched_gt)
                per_class[role]["fp"] += len(pred) - len(matched_pred)

    print(f"{'class':<12}{'tp':>6}{'fp':>6}{'fn':>6}{'recall':>10}{'precision':>10}")
    for role, c in per_class.items():
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan")
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else float("nan")
        print(f"{role:<12}{c['tp']:>6}{c['fp']:>6}{c['fn']:>6}{recall:>10.3f}{precision:>10.3f}")
    frame_recall = ball_frames_found / ball_frames_labelled if ball_frames_labelled else float("nan")
    print(f"\nball frame-recall: {ball_frames_found}/{ball_frames_labelled} = {frame_recall:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["convert", "build_yaml", "train_stage1", "train_stage2", "evaluate", "all"],
                   default="all")
    p.add_argument("--data_root", default=None, help="SoccerNet-GSR root; auto-detected under /kaggle/input if omitted")
    p.add_argument("--output_dir", default=None, help="default: /kaggle/working/gsr_yolo26 on Kaggle, else runs/gsr_yolo26")
    p.add_argument("--epochs", type=int, default=50, help="stage 1 epochs")
    p.add_argument("--epochs_stage2", type=int, default=80, help="stage 2 needs more: the P2 branch starts cold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=0, help="0 for first GPU, 'cpu' for CPU-only")
    p.add_argument("--resume", action="store_true", help="resume an interrupted run instead of starting fresh")
    p.add_argument("--init_weights", default=None, help="stage 1 only: start from this .pt (e.g. a 640 best.pt) instead of yolo26n.pt")
    p.add_argument("--run_name", default=None, help="stage 1 run folder name (default yolo26n_stage1); use a new name to avoid overwriting")
    p.add_argument("--label_mode", choices=list(LABEL_MODE_NAMES), default="4cls",
                   help="build_yaml: 4cls (default), athlete (player+gk+referee merged, ball dropped), athlete_ball")
    p.add_argument("--train_stride", type=int, default=1, help="use every Nth train frame (frames are near-duplicates at 25fps)")
    p.add_argument("--val_stride", type=int, default=1, help="use every Nth val frame (cheaper per-epoch validation)")
    p.add_argument("--stage1_weights", default=None, help="override path to stage 1 best.pt (for train_stage2/evaluate alone)")
    p.add_argument("--stage2_weights", default=None, help="override path to stage 2 best.pt (for evaluate alone)")
    p.add_argument("--baseline_weights", default=None, help="optional: current yolov8n_gsr.pt, for a 3-way eval comparison")
    p.add_argument("--baseline_imgsz", type=int, default=640, help="imgsz used when evaluating --baseline_weights")
    p.add_argument("--person_iou", type=float, default=0.5)
    p.add_argument("--ball_px", type=float, default=15.0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--ball_conf", type=float, default=0.15)
    p.add_argument("--eval_split", default=None, help="which split to evaluate on; default valid/valid_sample")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_ultralytics()

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "gsr_yolo26.yaml"

    data_root = None
    splits = None
    if args.mode in ("convert", "build_yaml", "all"):
        data_root = find_data_root(args.data_root)
        print(f"SoccerNet-GSR root: {data_root}")
        splits = discover_splits(data_root)
        print(f"Splits found: {[(k, len(v)) for k, v in splits.items()]}")

    if args.mode in ("convert", "all"):
        convert_all(splits)

    if args.mode in ("build_yaml", "all"):
        build_dataset_yaml(splits, yaml_path, args.train_stride, args.val_stride, args.label_mode)

    stage1_weights = Path(args.stage1_weights) if args.stage1_weights else output_dir / "yolo26n_stage1" / "weights" / "best.pt"
    stage2_weights = Path(args.stage2_weights) if args.stage2_weights else output_dir / "yolo26n_p2_stage2" / "weights" / "best.pt"

    if args.mode in ("train_stage1", "all"):
        stage1_weights = train_stage1(yaml_path, output_dir, args)

    if args.mode in ("train_stage2", "all"):
        stage2_weights = train_stage2(yaml_path, stage1_weights, output_dir, args)

    if args.mode in ("evaluate", "all"):
        if data_root is None:
            data_root = find_data_root(args.data_root)
            splits = discover_splits(data_root)
        eval_split_name = args.eval_split or ("valid" if "valid" in splits else "valid_sample" if "valid_sample" in splits else None)
        if eval_split_name is None:
            raise ValueError(f"No eval split available/specified. Splits found: {list(splits.keys())}")
        eval_seqs = splits[eval_split_name]
        print(f"\nEvaluating on '{eval_split_name}' split ({len(eval_seqs)} sequences)")

        if args.baseline_weights:
            evaluate_weights(Path(args.baseline_weights), eval_seqs, args.person_iou, args.ball_px, args.conf, args.ball_conf, args.baseline_imgsz)
        evaluate_weights(stage1_weights, eval_seqs, args.person_iou, args.ball_px, args.conf, args.ball_conf, args.imgsz)
        if stage2_weights.exists():
            evaluate_weights(stage2_weights, eval_seqs, args.person_iou, args.ball_px, args.conf, args.ball_conf, args.imgsz)


if __name__ == "__main__":
    main()
