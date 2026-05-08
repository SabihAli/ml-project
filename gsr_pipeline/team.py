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
            detections_df:  DataFrame with 'track_id', 'embeddings', 'role_detection', 'bbox_pitch'.

        Returns:
            detections_df with 'team_cluster' and 'team' columns.
        """
        detections_df = detections_df.copy()
        detections_df["team_cluster"] = np.nan
        detections_df["team"] = None

        if detections_df.empty:
            return detections_df

        # 1. Cluster player tracklets
        player_mask = (detections_df["role_detection"] == "player") & (detections_df["track_id"].notna())
        if not player_mask.any():
            return detections_df

        # Compute mean embedding per track_id
        tracklet_data = []
        for track_id, group in detections_df[player_mask].groupby("track_id"):
            # Stack all embeddings for this tracklet and average
            embs = np.vstack(group["embeddings"].values)
            mean_emb = np.mean(embs, axis=0)
            tracklet_data.append({"track_id": track_id, "mean_embedding": mean_emb})

        if not tracklet_data:
            return detections_df

        tracklet_df = pd.DataFrame(tracklet_data)
        
        if len(tracklet_df) >= 3:
            X = np.vstack(tracklet_df["mean_embedding"].values)
            # Use 3 clusters to separate Team A, Team B, and Referees
            kmeans = KMeans(n_clusters=3, random_state=self.random_state, n_init="auto")
            tracklet_df["team_cluster"] = kmeans.fit_predict(X)
        else:
            tracklet_df["team_cluster"] = 0

        # Map clusters back to main dataframe
        cluster_map = tracklet_df.set_index("track_id")["team_cluster"].to_dict()
        detections_df["team_cluster"] = detections_df["track_id"].map(cluster_map)

        # 2. Assign roles (identify referee)
        # Usually, the smallest cluster is the referee
        cluster_counts = tracklet_df["team_cluster"].value_counts().sort_values()
        referee_cluster = cluster_counts.index[0] if len(cluster_counts) == 3 else -1
        
        # 3. Side labeling (left vs right)
        # Look at the average X position of the team clusters
        side_stats = []
        team_clusters = [c for c in range(3) if c != referee_cluster]
        
        for cluster_id in team_clusters:
            cluster_mask = (detections_df["team_cluster"] == cluster_id)
            x_coords = []
            for bp in detections_df.loc[cluster_mask, "bbox_pitch"]:
                if bp and "x_bottom_middle" in bp:
                    x_coords.append(bp["x_bottom_middle"])
            
            avg_x = np.nanmean(x_coords) if x_coords else 0.0
            side_stats.append({"cluster": cluster_id, "avg_x": avg_x})

        labels = {}
        if referee_cluster != -1:
            labels[referee_cluster] = "referee"
            # Also update role_detection for referees
            ref_mask = (detections_df["team_cluster"] == referee_cluster)
            detections_df.loc[ref_mask, "role"] = "referee"
            detections_df.loc[ref_mask, "role_detection"] = "referee"

        if len(side_stats) >= 2:
            # Sort by avg_x to determine left/right
            side_stats.sort(key=lambda x: x["avg_x"])
            labels[side_stats[0]["cluster"]] = "left"
            labels[side_stats[1]["cluster"]] = "right"
            
        detections_df["team"] = detections_df["team_cluster"].map(labels)

        # 4. Goalkeeper labeling (based on proximity to goals)
        # Goalkeepers are often detected as players but stay near the goals.
        # We can also detect them if they are in the 'player' clusters but at extreme X.
        for tid, group in detections_df.groupby("track_id"):
            if tid == -1: continue
            avg_x = np.nanmean([bp["x_bottom_middle"] for bp in group["bbox_pitch"] if bp and "x_bottom_middle" in bp])
            if abs(avg_x) > 45: # Penalty area is roughly > 36m from center
                # This player is likely a goalkeeper
                idx_list = group.index
                detections_df.loc[idx_list, "role"] = "goalkeeper"
                detections_df.loc[idx_list, "role_detection"] = "goalkeeper"

        return detections_df
