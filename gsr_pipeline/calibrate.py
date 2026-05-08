"""
calibrate.py — Pitch calibration using the NBJW model.

Wraps the nbjw_calib package without tracklab types.
Checkpoints are downloaded from Zenodo on first use.

Output columns added to per-frame DataFrame:
    bbox_pitch  dict with pitch-coordinate keys, or None when calib fails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_KP_URL = "https://zenodo.org/records/12626395/files/SV_kp?download=1"
_LINES_URL = "https://zenodo.org/records/12626395/files/SV_lines?download=1"

_KP_TO_LINE: Dict[str, list] = {
    "Big rect. left bottom": [24, 68, 25],
    "Big rect. left main": [5, 64, 31, 46, 34, 66, 25],
    "Big rect. left top": [4, 62, 5],
    "Big rect. right bottom": [26, 69, 27],
    "Big rect. right main": [6, 65, 33, 56, 36, 67, 26],
    "Big rect. right top": [6, 63, 7],
    "Circle central": [32, 48, 38, 50, 42, 53, 35, 54, 43, 52, 39, 49],
    "Circle left": [31, 37, 47, 41, 34],
    "Circle right": [33, 40, 55, 44, 36],
    "Goal left crossbar": [16, 12],
    "Goal left post left": [16, 17],
    "Goal left post right": [12, 13],
    "Goal right crossbar": [15, 19],
    "Goal right post left": [15, 14],
    "Goal right post right": [19, 18],
    "Middle line": [2, 32, 51, 35, 29],
    "Side line bottom": [28, 70, 71, 29, 72, 73, 30],
    "Side line left": [1, 4, 8, 13, 17, 20, 24, 28],
    "Side line right": [3, 7, 11, 14, 18, 23, 27, 30],
    "Side line top": [1, 58, 59, 2, 60, 61, 3],
    "Small rect. left bottom": [20, 21],
    "Small rect. left main": [9, 21],
    "Small rect. left top": [8, 9],
    "Small rect. right bottom": [22, 23],
    "Small rect. right main": [10, 22],
    "Small rect. right top": [10, 11],
}


def _ensure_download(url: str, path: Path) -> None:
    if path.is_file():
        return
    log.info("Downloading calibration checkpoint → %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from tracklab.utils.download import download_file
        download_file(url, str(path))
    except ImportError:
        import urllib.request
        urllib.request.urlretrieve(url, str(path))
    log.info("Download complete: %s", path)


def _kp_to_lines(keypoints: Dict) -> Dict:
    lines = {}
    for name, indices in _KP_TO_LINE.items():
        pts = [{"x": keypoints[i]["x"], "y": keypoints[i]["y"]}
               for i in indices if i in keypoints]
        if pts:
            lines[name] = pts
    return lines


def _make_kp_cfg():
    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.NAME = "cls_hrnet"
    cfg.MODEL.IMAGE_SIZE = [960, 540]
    cfg.MODEL.HEATMAP_SIZE = [240, 135]
    cfg.MODEL.NUM_JOINTS = 58
    cfg.MODEL.EXTRA = CN()
    for stage_name, nb, nb_b, ch in [
        ("STAGE2", 1, [4, 4], [48, 96]),
        ("STAGE3", 4, [4, 4, 4], [48, 96, 192]),
        ("STAGE4", 3, [4, 4, 4, 4], [48, 96, 192, 384]),
    ]:
        s = CN()
        s.NUM_MODULES = nb
        s.NUM_BRANCHES = len(ch)
        s.NUM_BLOCKS = nb_b
        s.NUM_CHANNELS = ch
        s.BLOCK = "BASIC"
        s.FUSE_METHOD = "SUM"
        setattr(cfg.MODEL.EXTRA, stage_name, s)
    cfg.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
    cfg.MODEL.EXTRA.PRETRAINED_LAYERS = ["*"]
    cfg.defrost()
    return cfg


def _make_line_cfg():
    from yacs.config import CfgNode as CN
    cfg = _make_kp_cfg()
    cfg.MODEL.NAME = "cls_hrnet_l"
    cfg.MODEL.NUM_JOINTS = 24
    cfg.defrost()
    return cfg


class NBJWKeypointDetector:
    """Detects pitch keypoints and line endpoints using two HRNet models."""

    def __init__(
        self,
        checkpoint_kp: str,
        checkpoint_l: str,
        image_width: int = 1920,
        image_height: int = 1080,
        device: str = "cpu",
    ) -> None:
        import torch
        import torchvision.transforms as T
        from gsr_pipeline.nbjw_calib.model.cls_hrnet import get_cls_net
        from gsr_pipeline.nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l

        self.device = device
        _ensure_download(_KP_URL, Path(checkpoint_kp))
        _ensure_download(_LINES_URL, Path(checkpoint_l))

        state = torch.load(checkpoint_kp, map_location=device)
        self.model_kp = get_cls_net(_make_kp_cfg())
        self.model_kp.load_state_dict(state)
        self.model_kp.to(device).eval()

        state_l = torch.load(checkpoint_l, map_location=device)
        self.model_l = get_cls_net_l(_make_line_cfg())
        self.model_l.load_state_dict(state_l)
        self.model_l.to(device).eval()

        self.tfm = T.Compose([T.Resize((540, 960)), T.ToTensor()])
        log.info("NBJWKeypointDetector ready on %s", device)

    def detect(self, image_bgr: np.ndarray) -> Tuple[Dict, Dict]:
        import torch
        from PIL import Image
        from gsr_pipeline.nbjw_calib.utils.utils_heatmap import (
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
            complete_keypoints,
            coords_to_dict,
        )

        img = Image.fromarray(image_bgr[:, :, ::-1]).convert("RGB")
        t = self.tfm(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            hm_kp = self.model_kp(t)
            hm_l = self.model_l(t)

        kp_raw = coords_to_dict(
            get_keypoints_from_heatmap_batch_maxpool(hm_kp[:, :-1, :, :]),
            threshold=0.1449,
        )
        l_raw = coords_to_dict(
            get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1, :, :]),
            threshold=0.2983,
        )
        final = complete_keypoints(kp_raw, l_raw, w=t.size(-1), h=t.size(-2), normalize=True)
        keypoints = final[0] if final else {}
        return keypoints, _kp_to_lines(keypoints)


class NBJWHomography:
    """Computes a pitch homography from NBJW keypoints."""

    def __init__(
        self,
        image_width: int = 1920,
        image_height: int = 1080,
        use_prev_homography: bool = True,
    ) -> None:
        from gsr_pipeline.nbjw_calib.utils.utils_calib import FramebyFrameCalib
        self.cam = FramebyFrameCalib(image_width, image_height, denormalize=True)
        self.use_prev = use_prev_homography
        self._last_h: Optional[np.ndarray] = None
        self._last_params: Dict = {}

    def compute(self, keypoints: Dict) -> Tuple[Optional[np.ndarray], Dict]:
        self.cam.update(keypoints)
        h = self.cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
        if h is not None:
            params = self.cam.heuristic_voting()["cam_params"]
            if self.use_prev:
                self._last_h, self._last_params = h, params
            return h, params
        if self.use_prev and self._last_h is not None:
            return self._last_h, self._last_params
        return None, {}


def _unproject(h: np.ndarray, pt: list) -> np.ndarray:
    p = h @ np.array([pt[0], pt[1], 1.0])
    p /= p[2]
    return p


def project_detections(
    detections_df: pd.DataFrame,
    h: Optional[np.ndarray],
) -> pd.DataFrame:
    """Add bbox_pitch column to detections_df using homography H."""
    detections_df = detections_df.copy()
    if h is None:
        detections_df["bbox_pitch"] = None
        return detections_df

    pitches = []
    for _, row in detections_df.iterrows():
        l, t, w, ht = row["bbox_ltwh"]
        r, b = l + w, t + ht
        try:
            pbl = _unproject(h, [l, b])
            pbr = _unproject(h, [r, b])
            pbm = _unproject(h, [l + w / 2, b])
            if np.any(np.isnan([*pbl[:2], *pbr[:2], *pbm[:2]])):
                pitches.append(None)
            else:
                pitches.append({
                    "x_bottom_left": float(pbl[0]), "y_bottom_left": float(pbl[1]),
                    "x_bottom_right": float(pbr[0]), "y_bottom_right": float(pbr[1]),
                    "x_bottom_middle": float(pbm[0]), "y_bottom_middle": float(pbm[1]),
                })
        except Exception:
            pitches.append(None)

    detections_df["bbox_pitch"] = pitches
    return detections_df
