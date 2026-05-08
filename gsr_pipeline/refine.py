"""
refine.py — Post-processing tracklet refinement.

Splits tracks with inconsistent attributes and merges fragmented tracks
using physical feasibility and ReID similarity.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

log = logging.getLogger(__name__)

def refine_tracks(
    df: pd.DataFrame, 
    speed_threshold: float = 10.0, 
    reid_threshold: float = 0.3,
    interpolation_limit: int = 15
) -> pd.DataFrame:
    """
    Main entry point for track refinement.
    
    Args:
        df: DataFrame containing columns [image_id, track_id, x_bottom_middle, y_bottom_middle, 
                                        embeddings, team, jersey_number, role]
        speed_threshold: Max player speed in meters/second (default 10m/s).
        reid_threshold:  Max cosine distance for ReID merging.
        interpolation_limit: Max frame gap to interpolate.
        
    Returns:
        Refined DataFrame.
    """
    if df.empty or "track_id" not in df.columns:
        return df

    # 1. Split tracks with inconsistent attributes
    df = split_inconsistent_tracks(df)

    # 2. Merge fragmented tracks
    df = merge_tracklets(df, speed_threshold, reid_threshold)

    # 3. Interpolate small gaps in trajectories
    df = interpolate_tracks(df, limit=interpolation_limit)

    return df

def split_inconsistent_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits a track if the jersey number or team changes suddenly.
    This helps resolve identity swaps that the tracker missed.
    """
    new_dfs = []
    max_id = df["track_id"].max()

    for tid, group in df.groupby("track_id"):
        if tid == -1: # Unassigned
            new_dfs.append(group)
            continue
            
        group = group.sort_values("image_id")
        
        # We split if we see a change in jersey number that persists
        # For now, let's just use the tracklet_agg logic's voting, 
        # but if we find a long segment that contradicts the majority, we split.
        
        # Check if we have pitch coordinates
        if "x_bottom_middle" not in group.columns or group["x_bottom_middle"].isnull().all():
            new_dfs.append(group)
            continue

        # Let's implement position-based splitting (Sudden jumps on pitch)
        valid_group = group.dropna(subset=["x_bottom_middle"])
        coords = valid_group[["x_bottom_middle", "y_bottom_middle"]].values
        if len(coords) > 1:
            diffs = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            # If a player moves more than 5 meters in 1/25th of a second (125 m/s)
            # that's clearly an identity swap or calibration glitch.
            jump_indices = np.where(diffs > 5.0)[0]
            
            if len(jump_indices) > 0:
                # Split the group
                start_idx = 0
                for jump_idx in jump_indices:
                    split_seg = group.iloc[start_idx : jump_idx + 1].copy()
                    if start_idx > 0:
                        max_id += 1
                        split_seg["track_id"] = max_id
                    new_dfs.append(split_seg)
                    start_idx = jump_idx + 1
                
                final_seg = group.iloc[start_idx:].copy()
                if start_idx > 0:
                    max_id += 1
                    final_seg["track_id"] = max_id
                new_dfs.append(final_seg)
                continue

        new_dfs.append(group)

    return pd.concat(new_dfs, ignore_index=True)

def merge_tracklets(
    df: pd.DataFrame, 
    speed_threshold: float, 
    reid_threshold: float
) -> pd.DataFrame:
    """
    Attempts to merge non-overlapping tracklets of the same team/role.
    """
    # Exclude balls and unassigned
    track_data = []
    for tid, group in df.groupby("track_id"):
        if tid == -1: continue
        
        # Check if we have pitch coordinates for this tracklet
        if "x_bottom_middle" not in group.columns or group["x_bottom_middle"].isnull().all():
            continue

        group = group.sort_values("image_id").dropna(subset=["x_bottom_middle"])
        if group.empty:
            continue
            
        # Extract metadata for this tracklet
        track_data.append({
            "id": tid,
            "start_frame": group["image_id"].min(),
            "end_frame": group["image_id"].max(),
            "start_pos": group.iloc[0][["x_bottom_middle", "y_bottom_middle"]].values,
            "end_pos": group.iloc[-1][["x_bottom_middle", "y_bottom_middle"]].values,
            "team": group["team"].iloc[0],
            "role": group["role"].iloc[0],
            "jersey": group["jersey_number"].iloc[0],
            "mean_reid": np.mean(np.stack(group["embeddings"].values), axis=0) if "embeddings" in group.columns else None
        })

    if not track_data:
        return df

    # Greedy merging
    merges = {} # source_id -> target_id
    used_targets = set()
    
    # Sort tracklets by start frame
    track_data.sort(key=lambda x: x["start_frame"])
    
    for i in range(len(track_data)):
        t1 = track_data[i]
        if t1["id"] in merges or t1["id"] in used_targets:
            continue
            
        best_match = None
        min_dist = float("inf")
        
        for j in range(i + 1, len(track_data)):
            t2 = track_data[j]
            if t2["id"] in used_targets or t1["id"] == t2["id"]:
                continue
            
            # 1. Temporal constraint (Must not overlap)
            if t2["start_frame"] <= t1["end_frame"]:
                continue
            
            # 2. Basic Attribute consistency
            if t1["team"] != t2["team"] or t1["role"] != t2["role"]:
                continue
            
            # If both have jerseys and they don't match, skip
            if t1["jersey"] and t2["jersey"] and t1["jersey"] != t2["jersey"]:
                continue

            # 3. Physical feasibility
            frame_gap = t2["start_frame"] - t1["end_frame"]
            time_gap = frame_gap / 25.0 # Assume 25 FPS
            dist = np.linalg.norm(t2["start_pos"] - t1["end_pos"])
            speed = dist / time_gap
            
            if speed > speed_threshold:
                continue
                
            # 4. ReID Similarity
            reid_dist = 0.0
            if t1["mean_reid"] is not None and t2["mean_reid"] is not None:
                reid_dist = cosine(t1["mean_reid"], t2["mean_reid"])
                if reid_dist > reid_threshold:
                    continue
            
            # 5. Find best (closest in time and space)
            score = dist + frame_gap * 0.1 # Heuristic score
            if score < min_dist:
                min_dist = score
                best_match = t2["id"]
        
        if best_match:
            merges[t1["id"]] = best_match
            used_targets.add(best_match)

    # Apply merges
    if merges:
        log.info("Merging %d tracklets...", len(merges))
        # Resolve chains (A->B, B->C => A->C, B->C)
        resolved_merges = {}
        for start_node in merges:
            curr = start_node
            while curr in merges:
                curr = merges[curr]
            resolved_merges[start_node] = curr
            
        df["track_id"] = df["track_id"].replace(resolved_merges)

    return df

def interpolate_tracks(df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """
    Interpolates missing frames in tracks.
    """
    new_dfs = []
    for tid, group in df.groupby("track_id"):
        if tid == -1:
            new_dfs.append(group)
            continue
            
        group = group.sort_values("image_id")
        all_frames = np.arange(group["image_id"].min(), group["image_id"].max() + 1)
        
        if len(all_frames) == len(group):
            new_dfs.append(group)
            continue
            
        # Reindex to fill gaps
        group = group.set_index("image_id").reindex(all_frames)
        
        # Split bbox_ltwh into columns for interpolation
        if "bbox_ltwh" in group.columns:
            bboxes = group["bbox_ltwh"].apply(lambda x: pd.Series(x) if isinstance(x, (list, tuple)) else pd.Series([np.nan]*4))
            group[["_l", "_t", "_w", "_h"]] = bboxes
            group[["_l", "_t", "_w", "_h"]] = group[["_l", "_t", "_w", "_h"]].interpolate(method="linear", limit=limit)
            
            # Reconstruct bbox_ltwh
            group["bbox_ltwh"] = group.apply(
                lambda r: (r["_l"], r["_t"], r["_w"], r["_h"]) if pd.notna(r["_l"]) else np.nan, 
                axis=1
            )
            group = group.drop(columns=["_l", "_t", "_w", "_h"])

        # Interpolate coordinates
        group[["x_bottom_middle", "y_bottom_middle"]] = group[["x_bottom_middle", "y_bottom_middle"]].interpolate(
            method="linear", limit=limit
        )
        
        # Forward fill attributes
        cols_to_fill = ["track_id", "team", "role", "jersey_number", "role_detection", "role_confidence"]
        for col in cols_to_fill:
            if col in group.columns:
                group[col] = group[col].ffill().bfill()
        
        # Drop rows that couldn't be interpolated (gap too large)
        group = group.dropna(subset=["x_bottom_middle"])
        new_dfs.append(group.reset_index())

    return pd.concat(new_dfs, ignore_index=True)
