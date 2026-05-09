import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def subset_gsr_dataset(src_root, dst_root, ratio=0.65):
    """
    Moves/Copies a percentage of sequences from train/valid to a new folder.
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    
    for split in ["train", "valid"]:
        src_split = src_root / split
        dst_split = dst_root / split
        
        if not src_split.exists():
            print(f"Skipping {split}, source not found.")
            continue
            
        dst_split.mkdir(parents=True, exist_ok=True)
        
        # Get all sequence folders
        all_seqs = [d for d in src_split.iterdir() if d.is_dir()]
        num_to_move = int(len(all_seqs) * ratio)
        
        subset_seqs = random.sample(all_seqs, num_to_move)
        
        print(f"Copying {num_to_move}/{len(all_seqs)} sequences from {split} to {dst_split}...")
        
        for seq in tqdm(subset_seqs):
            target_path = dst_split / seq.name
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(str(seq), str(target_path))
            
        # Also copy sequences_info.json if it exists
        info_file = src_split / "sequences_info.json"
        if info_file.exists():
            shutil.copy2(str(info_file), str(dst_split / "sequences_info.json"))

if __name__ == "__main__":
    src = "data/SoccerNetGS/gamestate-2024"
    dst = "data/GSR_Subset_65"
    
    # We use copy instead of move to avoid destroying the local dataset
    subset_gsr_dataset(src, dst, ratio=0.65)
    print(f"\nDone! 65% subset created at {dst}")
    print("You can now zip this folder and upload it to Drive.")
