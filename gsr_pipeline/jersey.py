"""
jersey.py — Jersey number detection using EasyOCR.

Replaces the MMOCR stack from the original baseline with a lighter EasyOCR-based module.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from gsr_pipeline.utils import ltwh_to_ltrb, crop_bbox

log = logging.getLogger(__name__)


class EasyOCRJerseyDetector:
    """Detects jersey numbers on player crops using EasyOCR."""

    def __init__(
        self,
        languages: List[str] = ["en"],
        gpu: bool = True,
        batch_size: int = 8,
    ) -> None:
        """
        Args:
            languages:  List of languages for EasyOCR.
            gpu:        Whether to use GPU.
            batch_size: Processing batch size.
        """
        try:
            import easyocr
        except ImportError as e:
            raise ImportError(
                "easyocr is required for jersey detection. Install with:\n"
                "  pip install easyocr"
            ) from e

        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.batch_size = batch_size
        log.info("EasyOCRJerseyDetector loaded (gpu=%s)", gpu)

    def process_frame(
        self,
        image_bgr: np.ndarray,
        detections_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run OCR on all player detections in a single frame.

        Args:
            image_bgr:      Full BGR frame.
            detections_df:  DataFrame with ['bbox_ltwh', 'role_detection'].

        Returns:
            detections_df with 'jersey_number_detection' and 'jersey_number_confidence'.
        """
        detections_df = detections_df.copy()
        detections_df["jersey_number_detection"] = None
        detections_df["jersey_number_confidence"] = 0.0

        if detections_df.empty:
            return detections_df

        # Filter for players/goalkeepers
        mask = detections_df["role_detection"].isin(["player", "goalkeeper"])
        if not mask.any():
            return detections_df

        indices = detections_df.index[mask]
        crops = []
        for idx in indices:
            row = detections_df.loc[idx]
            l, t, w, h = row["bbox_ltwh"]
            l, t, r, b = ltwh_to_ltrb(l, t, w, h)
            crop = crop_bbox(image_bgr, int(l), int(t), int(r), int(b))
            crops.append(crop)

        # Process in batches
        results = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i : i + self.batch_size]
            # EasyOCR readtext_batched is more efficient
            # but simple readtext on individual crops is often more robust for jersey numbers
            # We'll use a loop for better reliability on small numbers
            for crop in batch:
                if crop.size == 0:
                    results.append((None, 0.0))
                    continue
                # EasyOCR likes RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                res = self.reader.readtext(crop_rgb)
                
                best_jn = None
                best_conf = 0.0
                for (_, text, conf) in res:
                    # Filter for numbers only
                    clean_text = "".join(filter(str.isdigit, text))
                    if clean_text and conf > best_conf:
                        best_jn = clean_text
                        best_conf = conf
                results.append((best_jn, best_conf))

        # Map back to dataframe
        for idx, (jn, conf) in zip(indices, results):
            detections_df.at[idx, "jersey_number_detection"] = jn
            detections_df.at[idx, "jersey_number_confidence"] = conf

        return detections_df
