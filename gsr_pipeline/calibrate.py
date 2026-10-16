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


NBJW_INPUT_HW = (540, 960)
KP_THRESHOLD, LINE_THRESHOLD = 0.1449, 0.2983     # NBJW's heatmap score floors
NBJW_BACKENDS = ("torch", "auto", "tensorrt", "ort_gpu", "ort_cpu")


def load_nbjw_models(checkpoint_kp: str, checkpoint_l: str, device: str = "cpu"):
    """The two NBJW torch models (keypoints, lines), eval mode, on `device`."""
    import torch
    from gsr_pipeline.nbjw_calib.model.cls_hrnet import get_cls_net
    from gsr_pipeline.nbjw_calib.model.cls_hrnet_l import get_cls_net as get_cls_net_l

    _ensure_download(_KP_URL, Path(checkpoint_kp))
    _ensure_download(_LINES_URL, Path(checkpoint_l))

    model_kp = get_cls_net(_make_kp_cfg())
    model_kp.load_state_dict(torch.load(checkpoint_kp, map_location=device))
    model_l = get_cls_net_l(_make_line_cfg())
    model_l.load_state_dict(torch.load(checkpoint_l, map_location=device))
    return model_kp.to(device).eval(), model_l.to(device).eval()


class _TorchNet:
    """Eager torch forward with the same `.infer(x)` shape as OrtRunner / TrtRunner."""

    def __init__(self, model, device: str, fp16: bool = False) -> None:
        import torch
        self.model = model
        self.device = torch.device(device)
        self.fp16 = fp16
        self.backend_name = "torch_fp16" if fp16 else "torch"

    def infer(self, x):
        import torch
        with torch.inference_mode():
            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self.model(x).float()
            return self.model(x)


def _nbjw_runner(onnx_path: Optional[str], backend: str, device: str, torch_model, fp16: bool,
                 auto_build_engine: bool = True):
    """Pick the forward-pass backend for one NBJW network.

    `auto` tries TensorRT (an engine next to the ONNX, checked against its sidecar; built here
    on first launch if it is missing and auto_build_engine is on, or ahead of time by
    scripts/export_nbjw.py --build_engine), then ONNX Runtime GPU, then torch.
    An explicitly named backend that cannot load raises instead of falling back.
    """
    on_cuda = str(device).startswith("cuda")
    if backend == "torch" or (backend == "auto" and not onnx_path):
        return _TorchNet(torch_model, device, fp16=fp16 and on_cuda)
    if not onnx_path:
        raise ValueError(f"calibration backend {backend!r} needs onnx_kp / onnx_l paths")

    from gsr_pipeline.runtime.backend import sha256_file

    onnx = Path(onnx_path)
    engine = onnx.with_name(f"{onnx.stem}_fp16.engine")
    tiers = ["tensorrt", "ort_gpu", "torch"] if backend == "auto" else [backend]
    errors = []
    for tier in tiers:
        try:
            if tier == "tensorrt":
                from gsr_pipeline.runtime.trt_runner import TrtRunner, check_sidecar, ensure_engine
                digest = sha256_file(onnx)
                ensure_engine(onnx, engine, digest, auto_build=auto_build_engine)
                check_sidecar(engine, digest)
                return TrtRunner(engine, device=device if on_cuda else "cuda:0")
            if tier in ("ort_gpu", "ort_cpu"):
                from gsr_pipeline.runtime.ort_runner import OrtRunner
                return OrtRunner(onnx, device="cuda:0" if tier == "ort_gpu" else "cpu")
            return _TorchNet(torch_model, device, fp16=fp16 and on_cuda)
        except Exception as exc:  # noqa: BLE001 -- each tier fails for its own reasons
            if backend != "auto":
                raise
            errors.append(f"{tier}: {exc}")
            log.info("NBJW %s: tier %s unavailable (%s)", onnx.name, tier, exc)
    raise RuntimeError("no NBJW backend could load: " + "; ".join(errors))


class NBJWKeypointDetector:
    """Detects pitch keypoints and line endpoints using two HRNet models.

    The networks run through `backend`: "torch" (eager, the accuracy reference), the
    exported ONNX via "tensorrt" / "ort_gpu" / "ort_cpu", or "auto" (TensorRT, then ORT GPU,
    then torch). Preprocessing (BGR->RGB, resize to 960x540, scale to [0, 1]) runs on the
    network's device, so only the uint8 frame crosses the bus.
    """

    def __init__(
        self,
        checkpoint_kp: str,
        checkpoint_l: str,
        image_width: int = 1920,
        image_height: int = 1080,
        device: str = "cpu",
        backend: str = "torch",
        onnx_kp: Optional[str] = None,
        onnx_l: Optional[str] = None,
        fp16: bool = False,
        fast_decode: bool = True,
        auto_build_engine: bool = True,
    ) -> None:
        import torch

        self.fast_decode = fast_decode
        if backend not in NBJW_BACKENDS:
            raise ValueError(f"calibration backend must be one of {NBJW_BACKENDS}, got {backend!r}")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            log.warning("calibration device %s requested but CUDA is unavailable; using cpu", device)
            device = "cpu"

        self.device = device
        self.model_kp, self.model_l = load_nbjw_models(checkpoint_kp, checkpoint_l, device)
        self.net_kp = _nbjw_runner(onnx_kp, backend, device, self.model_kp, fp16, auto_build_engine)
        self.net_l = _nbjw_runner(onnx_l, backend, device, self.model_l, fp16, auto_build_engine)
        self.input_device = self.net_kp.device
        log.info("NBJWKeypointDetector ready: %s, input on %s", self.backend_name, self.input_device)

    @property
    def backend_name(self) -> str:
        return f"{self.net_kp.backend_name}/{self.net_l.backend_name}"

    def preprocess(self, image_bgr: np.ndarray):
        """uint8 HxWx3 BGR -> float32 1x3x540x960 RGB in [0, 1], on the network's device.

        Antialiased bilinear matches the PIL resize the NBJW reference used (torchvision
        Resize on a PIL image); the one difference is that PIL rounds the resized pixels to
        uint8 before scaling, a <=1/255 step.
        """
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.ascontiguousarray(image_bgr)).to(self.input_device, non_blocking=True)
        x = x.permute(2, 0, 1).flip(0).unsqueeze(0).float().div_(255.0)
        if tuple(x.shape[-2:]) != NBJW_INPUT_HW:
            x = F.interpolate(x, size=NBJW_INPUT_HW, mode="bilinear", align_corners=False, antialias=True)
        return x.contiguous()

    def forward(self, x):
        """Both networks on one preprocessed batch; returns (hm_kp, hm_l)."""
        return self.net_kp.infer(x), self.net_l.infer(x)

    def decode(self, hm_kp, hm_l, fast: Optional[bool] = None) -> Dict:
        """Heatmaps -> completed, normalised keypoint dict. `fast` (default: the detector's
        `fast_decode`) picks nbjw_fast_decode over the vendored reference; both give the same
        keypoints, which scripts/parity_pipeline.py checks on real frames."""
        fast = self.fast_decode if fast is None else fast
        h, w = NBJW_INPUT_HW
        if fast:
            from gsr_pipeline import nbjw_fast_decode as dec
            kp_raw = dec.coords_to_dict(dec.peaks_kp(hm_kp[:, :-1]), threshold=KP_THRESHOLD)
            l_raw = dec.coords_to_dict(dec.peaks_lines(hm_l[:, :-1]), threshold=LINE_THRESHOLD)
            final = dec.complete_keypoints(kp_raw, l_raw, w=w, h=h, normalize=True)
        else:
            from gsr_pipeline.nbjw_calib.utils.utils_heatmap import (
                get_keypoints_from_heatmap_batch_maxpool,
                get_keypoints_from_heatmap_batch_maxpool_l,
                complete_keypoints,
                coords_to_dict,
            )
            kp_raw = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool(hm_kp[:, :-1, :, :]),
                                    threshold=KP_THRESHOLD)
            l_raw = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1, :, :]),
                                   threshold=LINE_THRESHOLD)
            final = complete_keypoints(kp_raw, l_raw, w=w, h=h, normalize=True)
        return final[0] if final else {}

    def detect(self, image_bgr: np.ndarray) -> Tuple[Dict, Dict]:
        hm_kp, hm_l = self.forward(self.preprocess(image_bgr))
        keypoints = self.decode(hm_kp, hm_l)
        return keypoints, _kp_to_lines(keypoints)


class NBJWHomography:
    """Computes a pitch homography (image pixels -> pitch metres) from NBJW keypoints.

    `compute_cam_params` additionally runs NBJW's `heuristic_voting`: 18 full
    `cv2.calibrateCamera` fits (3 modes x 6 RANSAC thresholds) to recover camera
    intrinsics/pose. Projection never uses them -- only the homography -- and they only
    feed the `camera_parameters` passthrough field of predictions.json, so they are off by
    default; they were most of calibration's per-frame cost.
    """

    def __init__(
        self,
        image_width: int = 1920,
        image_height: int = 1080,
        use_prev_homography: bool = True,
        compute_cam_params: bool = False,
    ) -> None:
        from gsr_pipeline.nbjw_calib.utils.utils_calib import FramebyFrameCalib
        self.cam = FramebyFrameCalib(image_width, image_height, denormalize=True)
        self.use_prev = use_prev_homography
        self.compute_cam_params = compute_cam_params
        self._last_h: Optional[np.ndarray] = None
        self._last_params: Dict = {}

    def compute(self, keypoints: Dict) -> Tuple[Optional[np.ndarray], Dict]:
        self.cam.update(keypoints)
        h = self.cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
        if h is not None:
            params = {}
            if self.compute_cam_params:
                voted = self.cam.heuristic_voting()
                params = voted["cam_params"] if voted else {}
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

    if detections_df.empty:
        detections_df["bbox_pitch"] = None
        return detections_df

    # One batched transform of the three bottom points of every box (same maths as
    # _unproject, which stays for single points).
    ltwh = np.asarray(detections_df["bbox_ltwh"].tolist(), dtype=np.float64).reshape(-1, 4)
    l, t, w, ht = ltwh.T
    b = t + ht
    pts = np.stack([
        np.stack([l, b], 1), np.stack([l + w, b], 1), np.stack([l + w / 2, b], 1),
    ], 1)                                                      # (N, 3, 2)
    homog = np.concatenate([pts, np.ones(pts.shape[:2] + (1,))], -1) @ np.asarray(h, np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        proj = homog[..., :2] / homog[..., 2:3]                # (N, 3, 2)
    ok = np.isfinite(proj).all(axis=(1, 2))

    pitches = [
        {
            "x_bottom_left": float(p[0, 0]), "y_bottom_left": float(p[0, 1]),
            "x_bottom_right": float(p[1, 0]), "y_bottom_right": float(p[1, 1]),
            "x_bottom_middle": float(p[2, 0]), "y_bottom_middle": float(p[2, 1]),
        } if good else None
        for p, good in zip(proj, ok)
    ]
    detections_df["bbox_pitch"] = pitches
    return detections_df
