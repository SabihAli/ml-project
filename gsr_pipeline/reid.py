"""
reid.py — Lightweight Re-Identification using torchreid OSNet.

Replaces the heavy PRTReId / BPBreID stack from the original baseline.

Design:
- Uses torchreid's FeatureExtractor with osnet_x0_25 (market1501 pretrained).
- Extracts a 512-d embedding per detection crop.
- Role classification is done heuristically (bbox size / aspect ratio).

Output columns added to the per-frame DataFrame:
    embeddings          np.ndarray of shape (512,)
    role_detection      one of {'ball', 'player', 'goalkeeper', 'referee', 'other'}
    role_confidence     float in [0, 1]
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from gsr_pipeline.utils import ltwh_to_ltrb, crop_bbox, bgr_to_rgb

log = logging.getLogger(__name__)

# COCO class IDs
_PERSON_CLASS = 0
_BALL_CLASS = 32

# Image size expected by OSNet
_REID_HEIGHT = 256
_REID_WIDTH = 128


class OSNetReID:
    """
    Lightweight ReID feature extractor using torchreid OSNet-x0_25.

    Also performs heuristic role classification (ball vs. player vs. referee
    vs. goalkeeper) because we no longer have the PRTReId multi-task head.
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

    def process_frame(
        self,
        image_bgr: np.ndarray,
        detections_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract embeddings for all detections in a single frame.

        Args:
            image_bgr:      Full BGR frame as numpy array.
            detections_df:  DataFrame with columns ['bbox_ltwh', 'class_id'].

        Returns:
            detections_df with new columns: embeddings, role_detection,
            role_confidence.
        """
        if detections_df.empty:
            detections_df["embeddings"] = pd.Series(dtype=object)
            detections_df["role_detection"] = pd.Series(dtype=str)
            detections_df["role_confidence"] = pd.Series(dtype=float)
            return detections_df

        img_h, img_w = image_bgr.shape[:2]
        image_rgb = bgr_to_rgb(image_bgr)

        # Assign heuristic roles first (no forward pass needed)
        roles, role_confs = self._assign_roles(detections_df, img_h, img_w)

        # Crop person detections for ReID
        crops = []
        for _, row in detections_df.iterrows():
            l, t, w, h = row["bbox_ltwh"]
            l, t, r, b = ltwh_to_ltrb(l, t, w, h)
            crop = crop_bbox(image_rgb, int(l), int(t), int(r), int(b))
            crops.append(crop)

        # Extract embeddings in batches
        embeddings = self._extract_embeddings(crops)

        detections_df = detections_df.copy()
        detections_df["embeddings"] = list(embeddings)
        detections_df["role_detection"] = roles
        detections_df["role_confidence"] = role_confs
        return detections_df

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

    @staticmethod
    def _assign_roles(
        detections_df: pd.DataFrame, img_h: int, img_w: int
    ):
        """Heuristic role assignment without a classification head.

        Rules (applied in order):
            1. class_id == _BALL_CLASS → 'ball' (conf 0.95)
            2. bbox area < 1% of image → 'ball' (small distant ball, conf 0.7)
            3. aspect ratio (h/w) < 1.2  → 'referee' (wider stance, conf 0.5)
               Actually referees vs players are hard without a head, so we
               default everyone to 'player' with low confidence and let the
               downstream tracklet aggregation/voting decide.
            4. Everything else → 'player' (conf 0.6)

        Goalkeeper assignment is deferred to `team.py` which can use the
        pitch coordinate to identify the player closest to each goal.
        """
        roles = []
        confs = []
        img_area = img_h * img_w

        for _, row in detections_df.iterrows():
            cls = int(row.get("class_id", _PERSON_CLASS))
            l, t, w, h = row["bbox_ltwh"]
            bbox_area = w * h

            if cls == _BALL_CLASS:
                roles.append("ball")
                confs.append(0.95)
            elif bbox_area < 0.0005 * img_area:
                # Very small detection — likely the ball
                roles.append("ball")
                confs.append(0.70)
            else:
                # Default to player; goalkeeper upgraded in team.py
                roles.append("player")
                confs.append(0.60)

        return roles, confs
