"""
tracklet_agg.py — Tracklet aggregation and voting.

Aggregates frame-level predictions (role, jersey number) into tracklet-level 
decisions using voting.
"""

from __future__ import annotations

import logging
from collections import Counter
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class TrackletAggregator:
    """Aggregates per-detection attributes into tracklet-level consensus."""

    def __init__(self) -> None:
        pass

    def process_sequence(self, detections_df: pd.DataFrame) -> pd.DataFrame:
        """Apply voting over tracklets for roles and jersey numbers.

        Args:
            detections_df: DataFrame with 'track_id', 'role_detection', 
                           'jersey_number_detection', 'jersey_number_confidence'.

        Returns:
            detections_df with updated 'role' and 'jersey_number' columns.
        """
        detections_df = detections_df.copy()
        detections_df["role"] = detections_df["role_detection"]
        detections_df["jersey_number"] = detections_df["jersey_number_detection"]

        if "track_id" not in detections_df.columns:
            return detections_df

        # Group by track_id
        for track_id, group in detections_df.groupby("track_id"):
            if pd.isna(track_id):
                continue

            # 1. Role voting
            roles = group["role_detection"].dropna().tolist()
            if roles:
                common_role = Counter(roles).most_common(1)[0][0]
                detections_df.loc[group.index, "role"] = common_role

            # 2. Jersey number weighted voting
            # We use confidence to weigh the votes
            jns = group["jersey_number_detection"].tolist()
            confs = group["jersey_number_confidence"].tolist()
            
            votes = {}
            for jn, conf in zip(jns, confs):
                if jn is not None:
                    votes[jn] = votes.get(jn, 0.0) + conf
            
            if votes:
                best_jn = max(votes.items(), key=lambda x: x[1])[0]
                detections_df.loc[group.index, "jersey_number"] = best_jn

        return detections_df
