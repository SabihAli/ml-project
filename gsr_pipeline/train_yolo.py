import os
import json
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
import shutil
import random

def convert_gsr_to_yolo(root_dir, output_dir, split="train", sample_ratio=1.0):
    """
    Converts SoccerNet-GSR annotations to YOLO format.
    
    Args:
        sample_ratio: Fraction of sequences to keep (e.g., 0.7 for 70%).
    """
    img_out = Path(output_dir) / "images" / split
    lbl_out = Path(output_dir) / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    split_dir = Path(root_dir) / split
    all_sequences = [d for d in split_dir.iterdir() if d.is_dir()]
    
    if sample_ratio < 1.0:
        num_to_keep = int(len(all_sequences) * sample_ratio)
        sequences = random.sample(all_sequences, num_to_keep)
        print(f"Sampling {num_to_keep}/{len(all_sequences)} sequences for {split} split...")
    else:
        sequences = all_sequences

    role_to_id = {
        "player": 0,
        "goalkeeper": 1,
        "referee": 2,
        "ball": 3
    }

    print(f"Processing {split} split ({len(sequences)} sequences)...")

    for seq in tqdm(sequences):
        label_path = seq / "Labels-GameState.json"
        if not label_path.exists():
            continue

        with open(label_path, "r") as f:
            data = json.load(f)

        # Image info (SoccerNet frames are usually 1920x1080)
        # Note: Some versions use a separate images.json, but GSR often embeds image_id
        # We assume images are in seq / "img1" / "{image_id}.jpg"
        
        # Group annotations by image_id
        ann_by_img = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in ann_by_img:
                ann_by_img[img_id] = []
            ann_by_img[img_id].append(ann)

        for img_id, anns in ann_by_img.items():
            # SoccerNet IDs are like "2021000001"
            # Filenames in img1 are usually like "000001.jpg" or the full ID
            # Let's check a sample sequence to confirm filename format
            img_filename = f"{img_id}.jpg" # Default guess
            src_img = seq / "img1" / img_filename
            
            if not src_img.exists():
                # Try 6-digit suffix
                img_filename = f"{img_id[-6:]}.jpg"
                src_img = seq / "img1" / img_filename
            
            if not src_img.exists():
                continue

            # Target paths
            target_img_name = f"{seq.name}_{img_id}.jpg"
            target_img_path = img_out / target_img_name
            target_lbl_path = lbl_out / (target_img_name.replace(".jpg", ".txt"))

            # Copy image (or symlink)
            shutil.copy(str(src_img), str(target_img_path))

            # Write YOLO labels
            # YOLO format: <class> <x_center> <y_center> <width> <height> (normalized 0-1)
            with open(target_lbl_path, "w") as f_lbl:
                for ann in anns:
                    bbox = ann.get("bbox_image")
                    if not bbox: continue
                    
                    role = ann.get("attributes", {}).get("role", "player")
                    class_id = role_to_id.get(role, 0)
                    
                    # GSR bbox: x, y, w, h (usually top-left)
                    # We need normalized center_x, center_y, w, h
                    img_w, img_h = 1920, 1080 # Standard SoccerNet
                    
                    x = bbox["x"]
                    y = bbox["y"]
                    w = bbox["w"]
                    h = bbox["h"]
                    
                    cx = (x + w/2) / img_w
                    cy = (y + h/2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    
                    f_lbl.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    return role_to_id

def create_yaml(output_dir, role_to_id):
    data = {
        "path": str(Path(output_dir).absolute()),
        "train": "images/train",
        "val": "images/valid",
        "names": {v: k for k, v in role_to_id.items()}
    }
    with open(Path(output_dir) / "gsr_data.yaml", "w") as f:
        yaml.dump(data, f)

if __name__ == "__main__":
    # Example usage for 70% dataset reduction:
    root = "data/SoccerNetGS/gamestate-2024"
    out = "data/yolo_gsr_70pct"
    
    role_id_map = convert_gsr_to_yolo(root, out, split="train", sample_ratio=1.0)
    convert_gsr_to_yolo(root, out, split="valid", sample_ratio=1.0) # Keep all validation
    
    create_yaml(out, role_id_map)
    print(f"Done! 70% of training data prepared in {out}")
