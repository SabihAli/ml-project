"""
team.py — Team clustering and side labeling.

Implements KMeans clustering on ReID embeddings to separate players into two teams,
and uses pitch positions to label them as 'left' or 'right' side teams.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)


class TeamClassifier:
    """Clusters player tracklets into two teams and assigns side labels."""

    def __init__(self, n_clusters: int = 2, random_state: int = 0) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state

    def process_sequence(self, detections_df: pd.DataFrame) -> pd.DataFrame:
        """Process an entire sequence of detections to assign teams.

        Args:
            detections_df:  DataFrame with 'track_id', 'embeddings', 'color_features', 'role_detection'.

        Returns:
            detections_df with updated 'team', 'role', and 'team_cluster' columns.
        """
        detections_df = detections_df.copy()
        if "team_cluster" not in detections_df.columns:
            detections_df["team_cluster"] = np.nan
        if "team" not in detections_df.columns:
            detections_df["team"] = None

        if detections_df.empty:
            return detections_df

        # 1. Prepare tracklet-level features
        player_mask = (detections_df["role_detection"] == "player") & (detections_df["track_id"].notna())
        if not player_mask.any():
            return detections_df

        tracklet_data = []
        for track_id, group in detections_df[player_mask].groupby("track_id"):
            # Average ReID embeddings
            embs = np.vstack(group["embeddings"].values)
            mean_emb = np.mean(embs, axis=0)
            
            # Average color features (histograms)
            colors = np.vstack(group["color_features"].values)
            mean_color = np.mean(colors, axis=0)
            
            tracklet_data.append({
                "track_id": track_id, 
                "mean_embedding": mean_emb,
                "mean_color": mean_color,
                "count": len(group)
            })

        if not tracklet_data:
            return detections_df

        tracklet_df = pd.DataFrame(tracklet_data)
        
        # 2. Clustering
        # We use color as the primary feature, ReID as secondary
        X_color = np.vstack(tracklet_df["mean_color"].values)
        X_reid = np.vstack(tracklet_df["mean_embedding"].values)
        
        # Normalize ReID to have similar scale/influence as color histograms
        from sklearn.preprocessing import normalize
        X_reid_norm = normalize(X_reid) * 0.3 # Give ReID 30% weight relative to color
        
        X_combined = np.hstack([X_color, X_reid_norm])
        
        n_clusters = min(3, len(tracklet_df))
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
            tracklet_df["team_cluster"] = kmeans.fit_predict(X_combined)
        else:
            tracklet_df["team_cluster"] = 0

        # 3. Identify Referee Cluster
        # Referees are usually:
        # A) Smallest cluster
        # B) Different color signature (often low saturation/value or distinct hue)
        cluster_info = []
        for cid in tracklet_df["team_cluster"].unique():
            c_group = tracklet_df[tracklet_df["team_cluster"] == cid]
            avg_color = np.mean(np.vstack(c_group["mean_color"].values), axis=0)
            cluster_info.append({
                "cluster": cid,
                "size": len(c_group),
                "avg_color": avg_color
            })
            
        referee_cluster = -1
        if len(cluster_info) == 3:
            # Sort by size (referees are usually fewer)
            cluster_info.sort(key=lambda x: x["size"])
            
            # Additional check: Is the smallest cluster actually a referee?
            # Referees often have a very distinct color profile.
            # For now, we keep the smallest cluster logic but make it safer.
            referee_cluster = cluster_info[0]["cluster"]
        
        # 4. Side Labeling (Left vs Right)
        team_clusters = [c["cluster"] for c in cluster_info if c["cluster"] != referee_cluster]
        cluster_map = tracklet_df.set_index("track_id")["team_cluster"].to_dict()
        detections_df["team_cluster"] = detections_df["track_id"].map(cluster_map)

        side_stats = []
        for cid in team_clusters:
            mask = (detections_df["team_cluster"] == cid)
            x_coords = [bp["x_bottom_middle"] for bp in detections_df.loc[mask, "bbox_pitch"] if bp and "x_bottom_middle" in bp]
            avg_x = np.nanmean(x_coords) if x_coords else 0.0
            side_stats.append({"cluster": cid, "avg_x": avg_x})

        labels = {}
        if referee_cluster != -1:
            labels[referee_cluster] = "referee"
            ref_mask = (detections_df["team_cluster"] == referee_cluster)
            detections_df.loc[ref_mask, "role"] = "referee"
            detections_df.loc[ref_mask, "role_detection"] = "referee"

        if len(side_stats) >= 2:
            side_stats.sort(key=lambda x: x["avg_x"])
            labels[side_stats[0]["cluster"]] = "left"
            labels[side_stats[1]["cluster"]] = "right"
            
        detections_df["team"] = detections_df["team_cluster"].map(labels)

        # 5. Final Goalkeeper Pass
        for tid, group in detections_df.groupby("track_id"):
            if pd.isna(tid) or tid == -1: continue
            # If the role is already referee, don't change it
            if group["role"].iloc[0] == "referee": continue
            
            x_vals = [bp["x_bottom_middle"] for bp in group["bbox_pitch"] if bp and "x_bottom_middle" in bp]
            if x_vals:
                avg_x = np.mean(x_vals)
                if abs(avg_x) > 40: # Near goals
                    detections_df.loc[group.index, "role"] = "goalkeeper"

        return detections_df

        return detections_df
