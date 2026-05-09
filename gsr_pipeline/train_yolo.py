"""
train_yolo.py
=============
Two-phase script for YOLOv8n transfer learning on SoccerNet-GSR:

Phase 1 – In-place YOLO conversion
  • Reads each sequence's Labels-GameState.json
  • Strips non-bounding-box annotations (pitch lines, camera metadata)
  • Keeps only 'object' supercategory entries (player, goalkeeper, referee, ball)
  • Writes one YOLO .txt label file per image directly into  seq/labels/<frame>.txt
    (no image copying, no parallel directory tree)

Phase 2 – Training
  • Builds a dataset YAML that lists the original image directories
  • Fine-tunes YOLOv8n on the converted dataset

Usage
-----
# Test on a single sequence first:
python train_yolo.py --mode convert_one --seq path/to/SNGS-060

# Convert the full dataset:
python train_yolo.py --mode convert_all

# Train (assumes conversion already done):
python train_yolo.py --mode train

# All-in-one:
python train_yolo.py --mode all
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Configuration – adjust these paths to match your environment
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path("GSR_Subset/GSR_Subset_65")   # dataset root
YAML_OUT   = Path("GSR_Subset/gsr_yolo.yaml")                # dataset YAML for YOLO
SPLITS     = ["train", "valid"]                        # splits to process

# YOLOv8 training hyper-parameters
TRAIN_CFG = dict(
    model     = "yolov8n.pt",     # pretrained backbone (downloads automatically)
    epochs    = 50,
    imgsz     = 640,             # SoccerNet frames are 1920×1080; keep aspect ratio
    batch     = 64,                # lower if VRAM is tight
    patience  = 10,               # early stopping
    workers   = 4,
    project   = "runs/gsr_yolo",
    name      = "yolov8n_gsr",
    exist_ok  = True,
    device    = 0,                # 0 = first GPU; "cpu" for CPU-only
)

# Class mapping: YOLO class id → human label
# We map the GSR category names to compact integer IDs
ROLE_TO_ID = {
    "player":     0,
    "goalkeeper": 1,
    "referee":    2,
    "ball":       3,
    "other":      4,
}
ID_TO_ROLE = {v: k for k, v in ROLE_TO_ID.items()}

# GSR category_id → role name  (from the JSON 'categories' list)
CAT_ID_TO_ROLE = {
    1: "player",
    2: "goalkeeper",
    3: "referee",
    4: "ball",
    7: "other",
}
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: In-place YOLO label conversion
# ══════════════════════════════════════════════════════════════════════════════

def convert_sequence(seq_dir: Path, dry_run: bool = False) -> dict:
    """
    Convert a single sequence's Labels-GameState.json to per-frame YOLO .txt
    files written into  seq_dir/labels/<frame>.txt.

    Only bounding-box annotations (supercategory == 'object') are kept;
    pitch-line and camera annotations are discarded.

    Args:
        seq_dir : Path to a sequence directory (e.g., .../SNGS-060)
        dry_run : If True, print what would be written but don't touch the disk.

    Returns:
        stats dict with keys: images_processed, annotations_kept, annotations_dropped
    """
    label_json = seq_dir / "Labels-GameState.json"
    if not label_json.exists():
        print(f"  [SKIP] No Labels-GameState.json in {seq_dir}")
        return {}

    with open(label_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build image_id → (file_name, width, height) lookup from the 'images' list
    img_info = {}
    for img in data.get("images", []):
        img_info[img["image_id"]] = (
            img["file_name"],
            img.get("width",  1920),
            img.get("height", 1080),
        )

    # Group annotations by image_id; keep ONLY bounding-box (object) entries
    bbox_anns: dict[str, list] = {}
    kept = 0
    dropped = 0
    for ann in data.get("annotations", []):
        if ann.get("supercategory") != "object":
            dropped += 1
            continue                       # skip pitch lines, camera, etc.
        bbox = ann.get("bbox_image")
        if not bbox:
            dropped += 1
            continue                       # no bounding box → skip

        img_id = ann["image_id"]
        if img_id not in bbox_anns:
            bbox_anns[img_id] = []
        bbox_anns[img_id].append(ann)
        kept += 1

    # Create labels sub-directory
    labels_dir = seq_dir / "labels"
    if not dry_run:
        labels_dir.mkdir(exist_ok=True)

    # YOLO requirement: images must be in a directory named 'images' (not 'img1')
    # for it to automatically find the sibling 'labels' directory.
    img1_dir = seq_dir / "img1"
    images_dir = seq_dir / "images"
    if img1_dir.exists() and not images_dir.exists():
        if dry_run:
            print(f"  [DRY] Would rename {img1_dir.name} -> {images_dir.name}")
        else:
            img1_dir.rename(images_dir)
            print(f"  [MOVE] Renamed {img1_dir.name} -> {images_dir.name}")
    elif not images_dir.exists() and not img1_dir.exists():
        print(f"  [SKIP] No image directory found in {seq_dir}")
        return {}

    images_processed = 0
    for img_id, anns in bbox_anns.items():
        file_name, img_w, img_h = img_info.get(
            img_id, (f"{img_id}.jpg", 1920, 1080)
        )
        # Strip extension → use same stem for the .txt
        stem = Path(file_name).stem               # e.g. "000001"
        txt_path = labels_dir / f"{stem}.txt"

        lines = []
        for ann in anns:
            bbox = ann["bbox_image"]              # x, y, w, h (top-left origin)
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

            # Normalise to YOLO format: cx cy w h  (all 0-1)
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to [0, 1] to guard against edge artefacts
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            # Resolve class id
            cat_id = ann.get("category_id")
            role   = CAT_ID_TO_ROLE.get(cat_id, "other")
            # Also check attributes.role as fallback
            if cat_id not in CAT_ID_TO_ROLE:
                role = ann.get("attributes", {}).get("role", "other")
            class_id = ROLE_TO_ID.get(role, 4)

            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if dry_run:
            print(f"  [DRY] Would write {txt_path} ({len(lines)} boxes)")
        else:
            with open(txt_path, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(lines) + ("\n" if lines else ""))

        images_processed += 1

    return dict(
        images_processed   = images_processed,
        annotations_kept   = kept,
        annotations_dropped= dropped,
    )


def convert_one(seq_path: str, dry_run: bool = False):
    """Convert a single sequence – used for testing before full rollout."""
    seq_dir = Path(seq_path)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Converting sequence: {seq_dir.name}")
    stats = convert_sequence(seq_dir, dry_run=dry_run)
    if stats:
        print(f"  Images processed   : {stats['images_processed']}")
        print(f"  BBox annotations   : {stats['annotations_kept']}")
        print(f"  Non-bbox dropped   : {stats['annotations_dropped']}")
        print(f"  Labels written to  : {seq_dir / 'labels'}")


def convert_all(root_dir: Path = ROOT_DIR, splits: list = SPLITS):
    """Convert every sequence in the given splits in-place."""
    total_imgs = 0
    total_kept = 0
    total_drop = 0

    for split in splits:
        split_dir = root_dir / split
        if not split_dir.exists():
            print(f"[WARN] Split directory not found: {split_dir}")
            continue

        sequences = sorted(d for d in split_dir.iterdir() if d.is_dir())
        print(f"\nConverting {split} split ({len(sequences)} sequences)…")

        for seq in tqdm(sequences, desc=split):
            stats = convert_sequence(seq)
            if stats:
                total_imgs += stats["images_processed"]
                total_kept += stats["annotations_kept"]
                total_drop += stats["annotations_dropped"]

    print(f"\n{'─'*50}")
    print(f"Conversion complete.")
    print(f"  Total images    : {total_imgs}")
    print(f"  BBox labels     : {total_kept}")
    print(f"  Dropped (pitch/cam/no-bbox): {total_drop}")


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2: Dataset YAML + YOLOv8n training
# ══════════════════════════════════════════════════════════════════════════════

def collect_image_dirs(root_dir: Path, split: str) -> list[str]:
    """
    Collect all img1/ subdirectory paths for a given split.
    YOLO's directory mode will look for paired .txt files in a sibling
    'labels/' directory (i.e., seq/labels/<stem>.txt).
    """
    split_dir = root_dir / split
    dirs = []
    for seq in sorted(d for d in split_dir.iterdir() if d.is_dir()):
        # Look for 'images' (renamed) or 'img1' (original)
        img_dir = seq / "images"
        if not img_dir.exists():
            img_dir = seq / "img1"
            
        if img_dir.exists():
            dirs.append(str(img_dir.resolve()))
    return dirs


def build_yaml(root_dir: Path = ROOT_DIR, yaml_out: Path = YAML_OUT):
    """
    Write a dataset YAML that Ultralytics can consume.

    Because YOLO expects label files to live in a 'labels/' directory
    that mirrors the 'images/' tree, and our images are in seq/img1/
    while labels are in seq/labels/, we use the 'path:' + explicit
    train/val lists approach with absolute image-directory listings.
    """
    train_dirs = collect_image_dirs(root_dir, "train")
    valid_dirs = collect_image_dirs(root_dir, "valid")

    dataset = {
        # Absolute paths so the YAML is location-independent
        "path" : str(root_dir.resolve()),
        "train": train_dirs,
        "val"  : valid_dirs,
        "nc"   : len(ROLE_TO_ID),
        "names": ID_TO_ROLE,
    }

    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_out, "w", encoding="utf-8") as f:
        yaml.dump(dataset, f, default_flow_style=False, sort_keys=False)

    print(f"\nDataset YAML written → {yaml_out}")
    print(f"  Train sequences : {len(train_dirs)}")
    print(f"  Valid sequences : {len(valid_dirs)}")
    print(f"  Classes         : {ID_TO_ROLE}")


def train(yaml_path: Path = YAML_OUT, cfg: dict = None):
    """
    Fine-tune YOLOv8n on the SoccerNet-GSR dataset.

    Requires: pip install ultralytics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is not installed. Run:  pip install ultralytics"
        )

    if cfg is None:
        cfg = TRAIN_CFG.copy()

    model_weights = cfg.pop("model", "yolov8n.pt")
    print(f"\nLoading model: {model_weights}")
    model = YOLO(model_weights)

    print(f"Starting training…  (YAML: {yaml_path})")
    results = model.train(data=str(yaml_path), **cfg)
    print(f"\nTraining complete. Results saved to: {results.save_dir}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SoccerNet-GSR annotations to YOLO format and train YOLOv8n."
    )
    parser.add_argument(
        "--mode",
        choices=["convert_one", "convert_all", "build_yaml", "train", "all"],
        default="convert_one",
        help=(
            "convert_one : test conversion on a single sequence (safe, no dataset-wide changes)\n"
            "convert_all : convert every sequence in all splits in-place\n"
            "build_yaml  : write the dataset YAML (after conversion)\n"
            "train       : run YOLOv8n fine-tuning (needs conversion + yaml done)\n"
            "all         : convert_all → build_yaml → train"
        ),
    )
    parser.add_argument(
        "--seq",
        default="data/SoccerNetGS/gamestate-2024/train/SNGS-060",
        help="Path to a single sequence directory (used with --mode convert_one)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files (convert_one only)",
    )
    parser.add_argument(
        "--root",
        default=str(ROOT_DIR),
        help="Root dataset directory (overrides ROOT_DIR constant)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root)

    if args.mode == "convert_one":
        convert_one(args.seq, dry_run=args.dry_run)

    elif args.mode == "convert_all":
        convert_all(root_dir=root)

    elif args.mode == "build_yaml":
        build_yaml(root_dir=root)

    elif args.mode == "train":
        build_yaml(root_dir=root)   # always regenerate YAML before training
        train()

    elif args.mode == "all":
        convert_all(root_dir=root)
        build_yaml(root_dir=root)
        train()
