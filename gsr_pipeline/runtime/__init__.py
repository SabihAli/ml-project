"""
gsr_pipeline.runtime -- backend-agnostic inference runners.

Path B replaces the per-call Ultralytics `YOLO(...)` hot path with thin runners that own
their own buffers (docs/path_b_implementation/IMPLEMENTATION_PLAN.md section 3.3). Three
tiers, tried in order by `backend.select_backend`: TensorRT FP16, ONNX Runtime GPU, ONNX
Runtime CPU.

Every tier shares one post-processing implementation (`postprocess.py`, including NMS), so
the tiers differ only in how the forward pass is executed. That is what makes the parity
gate (plan section 3.5) able to attribute a discrepancy to the backend rather than to
post-processing.

The ball specialist (plan section 5.3) and NBJW calibration (section 7) reuse these runners.
"""

from .backend import Backend, BackendSelection, ModelArtifact, select_backend
from .postprocess import letterbox_params, postprocess, preprocess

__all__ = [
    "Backend",
    "BackendSelection",
    "ModelArtifact",
    "select_backend",
    "letterbox_params",
    "postprocess",
    "preprocess",
]
