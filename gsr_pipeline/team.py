"""
team.py — Team clustering and side labeling.

Implements KMeans clustering on ReID embeddings + color histograms to separate
players into two teams, and uses pitch positions to label them as 'left' or
'right' side teams.

Referees are NOT clustered here: the fine-tuned detector (detect.py) already
emits a dedicated 'referee' class, which reid.py's `_assign_roles` maps to
`role_detection` before this stage runs -- a referee never belongs to either
team, so it is excluded outright.

Goalkeepers ARE NOT part of the KMeans fit (fitting on player-only tracklets
keeps the two clusters clean team-appearance groups, since a goalkeeper's kit
usually looks nothing like either outfield team's), but each goalkeeper
tracklet IS assigned a side afterward: nearest-centroid assignment against
the two fitted clusters, using the same feature representation (color
histogram + weighted ReID embedding) as the player tracklets that were
clustered. This was a known gap (goalkeepers previously kept `team = None`
indefinitely) -- see `_assign_nearest_cluster`.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)


def combine_color_reid(color: np.ndarray, embedding: np.ndarray, reid_weight: float = 0.3) -> np.ndarray:
    """Shared feature-combination primitive: L2-normalize the ReID embedding,
    scale by `reid_weight`, concatenate with the (already unit-norm, per
    utils.get_color_histogram) color histogram.

    Works on either a single row (`color`: (D_c,), `embedding`: (D_r,)) or a
    batch (`color`: (N, D_c), `embedding`: (N, D_r)) -- shapes are normalized
    to 2D internally and squeezed back if a single row was passed in.

    This is the one place the color+ReID combination math lives; both the
    batch `TeamClassifier` (via `_combined_feature_matrix`) and the live
    incremental assigner (`team_live.py`) call through here so the two can
    never silently drift out of sync.
    """
    from sklearn.preprocessing import normalize

    color_arr = np.asarray(color)
    was_single_row = color_arr.ndim == 1

    color_2d = np.atleast_2d(color_arr)
    embedding_2d = np.atleast_2d(np.asarray(embedding))
    reid_norm = normalize(embedding_2d) * reid_weight
    combined = np.hstack([color_2d, reid_norm])

    return combined[0] if was_single_row else combined


class TeamClassifier:
    """Clusters player tracklets into two teams and assigns side labels."""

    def __init__(self, n_clusters: int = 2, random_state: int = 0) -> None:
        """
        Args:
            n_clusters:    Number of team clusters. Should be 2 (Team A / Team B) —
                           referees and goalkeepers are already excluded from the
                           clustering input via `role_detection`, so a 3rd
                           "referee" cluster is neither needed nor correct.
            random_state:  KMeans random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

    @staticmethod
    def _tracklet_features(detections_df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Build one row per track_id under `mask`: mean ReID embedding + mean
        color histogram across all of that tracklet's detections.
        """
        rows = []
        for track_id, group in detections_df[mask].groupby("track_id"):
            embs = np.vstack(group["embeddings"].values)
            mean_emb = np.mean(embs, axis=0)

            colors = np.vstack(group["color_features"].values)
            mean_color = np.mean(colors, axis=0)

            rows.append({
                "track_id": track_id,
                "mean_embedding": mean_emb,
                "mean_color": mean_color,
                "count": len(group),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _combined_feature_matrix(tracklet_df: pd.DataFrame, reid_weight: float = 0.3) -> np.ndarray:
        """Same color+ReID feature construction used for both the KMeans fit
        (player tracklets) and nearest-centroid assignment (goalkeeper
        tracklets) -- must stay identical between the two call sites, or
        distances to the fitted centroids are meaningless.

        Note on `reid_weight`: because KMeans/Euclidean distance is driven by
        *squared* magnitude, a weight of 0.3 on an L2-normalized vector does
        NOT translate to "30% influence" -- concatenated with a unit-norm
        color vector, its actual contribution to squared distance is
        0.3^2 / (1^2 + 0.3^2) ~= 8%, not 30%. Named here so the discrepancy is
        documented rather than hidden behind a magic number (see
        docs/architecture/STREAMING_ARCHITECTURE.md §1.7.1 for the full analysis).
        """
        X_color = np.vstack(tracklet_df["mean_color"].values)
        X_reid = np.vstack(tracklet_df["mean_embedding"].values)
        return combine_color_reid(X_color, X_reid, reid_weight)

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

        # 1. Prepare tracklet-level features (players only -- referees are
        # excluded outright, goalkeepers are handled separately in step 2b)
        player_mask = (detections_df["role_detection"] == "player") & (detections_df["track_id"].notna())
        if not player_mask.any():
            return detections_df

        tracklet_df = self._tracklet_features(detections_df, player_mask)
        if tracklet_df.empty:
            return detections_df

        # 2. Clustering
        X_combined = self._combined_feature_matrix(tracklet_df)

        # Only "player" tracklets are fit here (referees are excluded via
        # role_detection, goalkeepers are assigned in step 2b below), so
        # clustering only ever needs to split the two teams -- no separate
        # referee/goalkeeper cluster.
        n_clusters = min(self.n_clusters, len(tracklet_df))
        kmeans = None
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
            tracklet_df["team_cluster"] = kmeans.fit_predict(X_combined)
        else:
            tracklet_df["team_cluster"] = 0

        cluster_map = tracklet_df.set_index("track_id")["team_cluster"].to_dict()

        # 2b. Goalkeepers: not part of the KMeans fit (their kit usually
        # doesn't resemble either outfield team's appearance, which would
        # pollute the two team clusters), but each goalkeeper tracklet is
        # still assigned a side by nearest-centroid distance against the
        # already-fitted clusters, using the identical feature construction.
        # Only possible once real clustering happened (kmeans fit, n>=2) --
        # with fewer than 2 clusters there are no meaningful centroids to
        # compare against, so goalkeepers are left unassigned in that case.
        gk_mask = (detections_df["role_detection"] == "goalkeeper") & (detections_df["track_id"].notna())
        if kmeans is not None and gk_mask.any():
            gk_tracklet_df = self._tracklet_features(detections_df, gk_mask)
            if not gk_tracklet_df.empty:
                X_gk = self._combined_feature_matrix(gk_tracklet_df)
                gk_tracklet_df["team_cluster"] = kmeans.predict(X_gk)
                cluster_map.update(gk_tracklet_df.set_index("track_id")["team_cluster"].to_dict())

        detections_df["team_cluster"] = detections_df["track_id"].map(cluster_map)

        # Side labeling uses player + goalkeeper detections alike now that
        # both are mapped into team_cluster -- a goalkeeper's extreme
        # near-own-goal x position only reinforces its side's average, it
        # doesn't skew it toward the wrong label.
        side_stats = []
        for cid in sorted(set(cluster_map.values())):
            mask = (detections_df["team_cluster"] == cid)
            x_coords = [bp["x_bottom_middle"] for bp in detections_df.loc[mask, "bbox_pitch"] if bp and "x_bottom_middle" in bp]
            avg_x = np.nanmean(x_coords) if x_coords else 0.0
            side_stats.append({"cluster": cid, "avg_x": avg_x})

        labels = {}
        if len(side_stats) >= 2:
            side_stats.sort(key=lambda x: x["avg_x"])
            labels[side_stats[0]["cluster"]] = "left"
            labels[side_stats[1]["cluster"]] = "right"

        detections_df["team"] = detections_df["team_cluster"].map(labels)

        return detections_df
