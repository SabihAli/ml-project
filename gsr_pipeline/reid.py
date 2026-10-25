"""
reid.py — Lightweight Re-Identification using torchreid OSNet.

Replaces the heavy PRTReId / BPBreID stack from the original baseline.

Design:
- Uses torchreid's FeatureExtractor with osnet_x0_25 (market1501 pretrained).
- Extracts a 512-d embedding per detection crop.
- Used only by the batch refiner (refine.py merges tracklets on embedding distance).
  Roles and teams come from kit colour (live_roles.py); the old pass-through of the
  4-class detector's class_id as the role went with Path A (archive/path_a/).

Output columns added to the per-frame DataFrame:
    embeddings          np.ndarray of shape (512,)
    color_features      colour histogram per crop
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from gsr_pipeline.utils import ltwh_to_ltrb, crop_bbox, bgr_to_rgb, get_color_histogram

log = logging.getLogger(__name__)

# Image size expected by OSNet
_REID_HEIGHT = 256
_REID_WIDTH = 128


class OSNetReID:
    """
    Lightweight ReID feature extractor using torchreid OSNet-x0_25.
    """

    def __init__(
        self,
        model_name: str = "osnet_x0_25",
        weights_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        image_height: int = _REID_HEIGHT,
        image_width: int = _REID_WIDTH,
    ) -> None:
        """
        Args:
            model_name:    torchreid model key (e.g. 'osnet_x0_25', 'resnet50').
            weights_path:  Path to a custom .pth checkpoint.  None → use the
                           torchreid pretrained weights from market1501.
            device:        Torch device string.
            batch_size:    Crops to process per forward pass.
            image_height:  Crop resize height (default 256).
            image_width:   Crop resize width (default 128).
        """
        try:
            import torchreid
        except ImportError as e:
            raise ImportError(
                f"torchreid or one of its dependencies is missing: {e}\n"
                "Please ensure all requirements are installed: pip install -r requirements_pipeline.txt"
            ) from e

        self.device = device
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width

        extractor_kwargs = dict(
            model_name=model_name,
            model_path=weights_path or "",
            device=device,
            image_size=(image_height, image_width),
            pixel_mean=[0.485, 0.456, 0.406],
            pixel_std=[0.229, 0.224, 0.225],
        )
        # torchreid FeatureExtractor accepts an empty model_path for pretrained
        self.extractor = torchreid.utils.FeatureExtractor(**extractor_kwargs)
        log.info(
            "OSNetReID loaded: model=%s  device=%s", model_name, device
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, image_bgr: np.ndarray, detections_df: pd.DataFrame) -> pd.DataFrame:
        """Embeddings + colour histograms for one frame (same as extract_appearance_features)."""
        return self.extract_appearance_features(image_bgr, detections_df)

    def extract_appearance_features(self, image_bgr: np.ndarray, detections_df: pd.DataFrame) -> pd.DataFrame:
        """ReID embeddings + color histograms for every row's crop. Adds the
        `embeddings` and `color_features` columns; roles are set elsewhere
        (live_roles.py).
        """
        if detections_df.empty:
            out = detections_df.copy()
            out["embeddings"] = pd.Series(dtype=object)
            out["color_features"] = pd.Series(dtype=object)
            return out

        image_rgb = bgr_to_rgb(image_bgr)
        crops_rgb, crops_bgr = [], []
        for _, row in detections_df.iterrows():
            l, t, w, h = row["bbox_ltwh"]
            l, t, r, b = ltwh_to_ltrb(l, t, w, h)
            crops_rgb.append(crop_bbox(image_rgb, int(l), int(t), int(r), int(b)))
            crops_bgr.append(crop_bbox(image_bgr, int(l), int(t), int(r), int(b)))

        embeddings = self._extract_embeddings(crops_rgb)
        color_feats = [get_color_histogram(c) for c in crops_bgr]

        out = detections_df.copy()
        out["embeddings"] = list(embeddings)
        out["color_features"] = color_feats
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """Run torchreid extractor on a list of RGB crop arrays."""
        import cv2
        import torch

        if not crops:
            return np.zeros((0, 512), dtype=np.float32)

        resized = []
        for crop in crops:
            if crop.size == 0:
                resized.append(
                    np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                )
            else:
                resized.append(
                    cv2.resize(crop, (self.image_width, self.image_height))
                )

        all_embeddings = []
        for start in range(0, len(resized), self.batch_size):
            batch = resized[start : start + self.batch_size]
            with torch.no_grad():
                feats = self.extractor(batch)  # tensor (B, D)
            all_embeddings.append(feats.cpu().numpy())

        return np.vstack(all_embeddings)
