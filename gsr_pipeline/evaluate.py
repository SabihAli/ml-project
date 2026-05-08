"""
evaluate.py — Evaluation script for SoccerNet Game State Reconstruction.

Calculates accuracy benchmarks by comparing predictions.json against 
the ground truth Labels-GameState.json.

Metrics:
- Team Accuracy (matched players)
- Role Accuracy (matched persons)
- Average Localization Error (meters on pitch)
- Detection Precision/Recall (at 5m threshold)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)

def load_gt(gt_path: Path):
    with open(gt_path, "r") as f:
        data = json.load(f)
    
    # Flatten annotations
    rows = []
    for ann in data.get("annotations", []):
        if ann.get("supercategory") != "object": continue
        
        row = {
            "image_id": ann["image_id"],
            "track_id": ann.get("track_id"),
            "role": ann["attributes"].get("role"),
            "team": ann["attributes"].get("team"),
            "jersey": ann["attributes"].get("jersey"),
        }
        
        bp = ann.get("bbox_pitch")
        if isinstance(bp, dict):
            row["x"] = bp.get("x_bottom_middle")
            row["y"] = bp.get("y_bottom_middle")
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def load_preds(pred_path: Path):
    with open(pred_path, "r") as f:
        data = json.load(f)
    
    rows = []
    for frame in data.get("predictions", []):
        img_id = frame["image_id"]
        for det in frame.get("detections", []):
            row = {
                "image_id": img_id,
                "track_id": det.get("track_id"),
                "role": det.get("role"),
                "team": det.get("team"),
                "jersey": det.get("jersey_number"),
            }
            bp = det.get("bbox_pitch")
            if isinstance(bp, dict):
                row["x"] = bp.get("x_bottom_middle")
                row["y"] = bp.get("y_bottom_middle")
            rows.append(row)
            
    return pd.DataFrame(rows)

def evaluate_gs_hota(gt_df: pd.DataFrame, pred_df: pd.DataFrame, tau: float = 5.0):
    """
    Official GS-HOTA metric implementation.
    LocSim(P, G) = exp(ln(0.05) * dist^2 / tau^2)
    IdSim(P, G) = 1 if all attributes match, 0 otherwise.
    """
    # 0. Normalize image_ids
    def normalize_id(img_id):
        s = str(img_id)
        if len(s) > 6: return int(s[-6:])
        return int(s)

    gt_df["norm_id"] = gt_df["image_id"].apply(normalize_id)
    pred_df["norm_id"] = pred_df["image_id"].apply(normalize_id)
    
    gt_ids = set(gt_df["norm_id"].unique())
    pred_ids = set(pred_df["norm_id"].unique())
    img_ids = sorted(list(gt_ids.intersection(pred_ids)))
    
    if not img_ids:
        print("Error: No matching image IDs found between ground truth and predictions.")
        return 0.0
    
    print(f"Evaluating on {len(img_ids)} matching frames.")

    # Filter DFs to only evaluated frames
    gt_df = gt_df[gt_df["norm_id"].isin(img_ids)]
    pred_df = pred_df[pred_df["norm_id"].isin(img_ids)]
    
    # Store track information for AssA
    gt_track_counts = gt_df["track_id"].value_counts().to_dict()
    pred_track_counts = pred_df["track_id"].value_counts().to_dict()
    
    ln_005 = np.log(0.05)
    frame_pairs = [] # (img_id, gt_tids, pred_tids, sim_matrix)

    for img_id in img_ids:
        gt_frame = gt_df[gt_df["norm_id"] == img_id].dropna(subset=["x", "y"])
        pred_frame = pred_df[pred_df["norm_id"] == img_id].dropna(subset=["x", "y"])
        
        if gt_frame.empty or pred_frame.empty:
            frame_pairs.append((img_id, gt_frame, pred_frame, None))
            continue

        gt_coords = gt_frame[["x", "y"]].values
        pred_coords = pred_frame[["x", "y"]].values
        dists_sq = np.sum((gt_coords[:, None, :] - pred_coords[None, :, :])**2, axis=2)
        
        # LocSim = exp(ln(0.05) * dist^2 / tau^2)
        loc_sim = np.exp(ln_005 * dists_sq / (tau**2))
        
        # IdSim
        id_sim = np.zeros((len(gt_frame), len(pred_frame)))
        for i, (_, gt_row) in enumerate(gt_frame.iterrows()):
            for j, (_, pred_row) in enumerate(pred_frame.iterrows()):
                # Role match
                role_match = (gt_row["role"] == pred_row["role"])
                if not role_match: continue
                
                # Team match
                team_match = True
                if gt_row["role"] in ["player", "goalkeeper"] and gt_row["team"]:
                    team_match = (gt_row["team"] == pred_row["team"])
                if not team_match: continue
                
                # Jersey match
                jersey_match = True
                gt_j = gt_row["jersey"]
                if pd.notna(gt_j) and gt_j is not None:
                    pred_j = pred_row["jersey"]
                    jersey_match = (str(gt_j) == str(pred_j))
                
                if jersey_match:
                    id_sim[i, j] = 1.0
        
        sim_matrix = loc_sim * id_sim
        frame_pairs.append((img_id, gt_frame, pred_frame, sim_matrix))

    # 2. Evaluate over alpha thresholds [0.05, 0.10, ..., 0.95]
    alphas = np.linspace(0.05, 0.95, 19)
    hota_results = []

    for alpha in alphas:
        tp_count = 0
        fp_count = 0
        fn_count = 0
        matches = {} # (gt_tid, pred_tid) -> count

        for img_id, gt_frame, pred_frame, sim_matrix in frame_pairs:
            if sim_matrix is None:
                fn_count += len(gt_frame)
                fp_count += len(pred_frame)
                continue
            
            # Filter sim_matrix by alpha
            alpha_sim = sim_matrix.copy()
            alpha_sim[alpha_sim <= alpha] = 0
            
            if np.all(alpha_sim == 0):
                fn_count += len(gt_frame)
                fp_count += len(pred_frame)
                continue
                
            # Bipartite matching to maximize similarity
            gt_indices, pred_indices = linear_sum_assignment(-alpha_sim)
            
            matched_gt = set()
            matched_pred = set()
            for g_idx, p_idx in zip(gt_indices, pred_indices):
                if alpha_sim[g_idx, p_idx] > 0:
                    tp_count += 1
                    matched_gt.add(g_idx)
                    matched_pred.add(p_idx)
                    
                    gt_tid = gt_frame.iloc[g_idx]["track_id"]
                    pred_tid = pred_frame.iloc[p_idx]["track_id"]
                    matches[(gt_tid, pred_tid)] = matches.get((gt_tid, pred_tid), 0) + 1
            
            fn_count += len(gt_frame) - len(matched_gt)
            fp_count += len(pred_frame) - len(matched_pred)

        detA = tp_count / (tp_count + fp_count + fn_count + 1e-6)
        
        assa_sum = 0
        for (gt_tid, pred_tid), tpa in matches.items():
            fpa = pred_track_counts.get(pred_tid, 0) - tpa
            fna = gt_track_counts.get(gt_tid, 0) - tpa
            assa_sum += tpa * (tpa / (tpa + fpa + fna + 1e-6))
        assA = assa_sum / (tp_count + 1e-6)
        
        hota_results.append({
            "alpha": alpha,
            "detA": detA,
            "assA": assA,
            "gs_hota": np.sqrt(detA * assA)
        })

    # Summary
    mean_gs_hota = np.mean([r["gs_hota"] for r in hota_results])
    mean_detA = np.mean([r["detA"] for r in hota_results])
    mean_assA = np.mean([r["assA"] for r in hota_results])

    print("\n" + "="*60)
    print(f"{'OFFICIAL GS-HOTA EVALUATION (tau=5m)':^60}")
    print("="*60)
    print(f"{'Alpha':<10} | {'DetA':<10} | {'AssA':<10} | {'GS-HOTA':<10}")
    print("-" * 60)
    # Print only a few thresholds to keep output clean
    for r in hota_results[::2]: 
        print(f"{r['alpha']:<10.2f} | {r['detA']*100:<10.2f}% | {r['assA']*100:<10.2f}% | {r['gs_hota']*100:<10.2f}%")
    print("-" * 60)
    print(f"{'MEAN':<10} | {mean_detA*100:<10.2f}% | {mean_assA*100:<10.2f}% | {mean_gs_hota*100:<10.2f}%")
    print("="*60)
    
    return mean_gs_hota

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    args = parser.parse_args()
    
    if not Path(args.gt_path).exists():
        print(f"Error: Ground truth file not found at {args.gt_path}")
        return
    if not Path(args.pred_path).exists():
        print(f"Error: Prediction file not found at {args.pred_path}")
        return

    print(f"Loading Ground Truth: {args.gt_path}")
    gt_df = load_gt(Path(args.gt_path))
    print(f"Loading Predictions:  {args.pred_path}")
    pred_df = load_preds(Path(args.pred_path))
    
    evaluate_gs_hota(gt_df, pred_df)

if __name__ == "__main__":
    main()
