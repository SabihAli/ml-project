"""Shared utility helpers for the GSR pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image / video helpers
# ---------------------------------------------------------------------------

def load_frames_from_dir(img_dir: Path) -> List[Tuple[int, np.ndarray]]:
    """Load all .jpg/.png frames from a directory, sorted by filename.

    Returns a list of (frame_idx, bgr_image) tuples.
    The frame index is the 1-based integer parsed from the filename
    (standard SoccerNet img1/ naming: 000001.jpg, 000002.jpg, …).
    """
    extensions = {".jpg", ".jpeg", ".png"}
    paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(f"No image files found in {img_dir}")

    frames = []
    for p in paths:
        # Parse frame index from filename stem (drop leading zeros)
        try:
            idx = int(p.stem)
        except ValueError:
            idx = len(frames) + 1
        img = cv2.imread(str(p))
        if img is None:
            log.warning("Could not read %s — skipping.", p)
            continue
        frames.append((idx, img))
    return frames


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV image to RGB."""
    return img[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

def ltrb_to_ltwh(l: float, t: float, r: float, b: float) -> Tuple[float, float, float, float]:
    """Convert left-top-right-bottom to left-top-width-height."""
    return l, t, r - l, b - t


def ltwh_to_ltrb(l: float, t: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Convert left-top-width-height to left-top-right-bottom."""
    return l, t, l + w, t + h


def clip_bbox_to_image(
    l: int, t: int, r: int, b: int, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """Clip a bounding box to the image boundaries."""
    l = max(0, min(l, img_w - 1))
    t = max(0, min(t, img_h - 1))
    r = max(0, min(r, img_w))
    b = max(0, min(b, img_h))
    return l, t, r, b


def crop_bbox(img: np.ndarray, l: int, t: int, r: int, b: int) -> np.ndarray:
    """Return the image crop for a bounding box (ltrb, integer)."""
    l, t, r, b = clip_bbox_to_image(l, t, r, b, img.shape[1], img.shape[0])
    crop = img[t:b, l:r]
    return crop


def get_color_histogram(crop_bgr: np.ndarray) -> np.ndarray:
    """Extract a small HSV histogram from the center part of a crop (torso area).
    
    Returns a normalized 1D feature vector.
    """
    if crop_bgr.size == 0:
        return np.zeros(32, dtype=np.float32)

    # Focus on the torso (middle 50% height, middle 80% width)
    h, w = crop_bgr.shape[:2]
    t, b = int(h * 0.2), int(h * 0.7)
    l, r = int(w * 0.1), int(w * 0.9)
    torso = crop_bgr[t:b, l:r]

    if torso.size == 0:
        return np.zeros(32, dtype=np.float32)

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    
    # 8 bins for H, 4 for S, 4 for V = 128 dimensions? 
    # Let's keep it smaller: 8 for H, 4 for S = 32 dimensions (ignore V to be light-robust)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# ---------------------------------------------------------------------------
# Pandas helpers
# ---------------------------------------------------------------------------

def require_columns(df, columns: List[str], step: str) -> None:
    """Assert that a DataFrame contains the given columns."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{step}] DataFrame is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
