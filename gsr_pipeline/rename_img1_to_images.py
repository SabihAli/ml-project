import os
from pathlib import Path
from tqdm import tqdm

def rename_img1_to_images(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        print(f"Error: {root} does not exist.")
        return

    # Process all subdirectories (splits like train, valid, test, etc.)
    splits = [d for d in root.iterdir() if d.is_dir()]
    
    total_renamed = 0
    
    for split in splits:
        sequences = [d for d in split.iterdir() if d.is_dir()]
        print(f"\nProcessing split: {split.name} ({len(sequences)} sequences)")
        
        for seq in tqdm(sequences, desc=split.name):
            img1 = seq / "img1"
            images = seq / "images"
            
            if img1.exists() and not images.exists():
                try:
                    img1.rename(images)
                    total_renamed += 1
                except Exception as e:
                    print(f"Error renaming {img1}: {e}")
            elif images.exists():
                # Already renamed or exists
                pass
                
    print(f"\nFinished! Total directories renamed: {total_renamed}")

if __name__ == "__main__":
    # You can change this to your subset path if needed
    DEFAULT_ROOT = "GSR_Subset_65"
    
    import argparse
    parser = argparse.ArgumentParser(description="Rename all 'img1' directories to 'images' for YOLO compatibility.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help=f"Path to dataset root (default: {DEFAULT_ROOT})")
    
    args = parser.parse_args()
    rename_img1_to_images(args.root)
