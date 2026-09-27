"""
backend.py -- artifact loading and the three-tier backend fallback.

IMPLEMENTATION_PLAN.md section 3.1. Tiers are tried in priority order and every rejection
is logged with its reason, so a host that silently ends up in degraded mode can be
diagnosed from the log alone rather than from a stopwatch:

    1. TensorRT FP16    -- tensorrt importable, CUDA present, sidecar matches, warm-up OK
    2. ONNX Runtime GPU -- CUDAExecutionProvider available AND actually active
    3. ONNX Runtime CPU -- always; degraded mode, not real-time

Two deliberate choices:

  * **Warm-up is part of selection.** A tier that loads but throws on its first real
    inference is not a working tier. Each candidate runs one inference on a zero tensor
    before it is accepted, so a broken tier is rejected at startup rather than on the
    first frame of a live match.
  * **The sidecar is a gate, not a hint.** A mismatched engine is refused outright
    (plan 3.2), because running an engine built for another GPU is undefined behaviour
    and a wrong answer is worse than a slower one.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch

log = logging.getLogger(__name__)

EXPECTED_NAMES = {0: "athlete", 1: "ball"}


class Backend(str, Enum):
    TENSORRT = "tensorrt"
    ORT_GPU = "ort_gpu"
    ORT_CPU = "ort_cpu"


# Priority order, best first.
TIER_ORDER = (Backend.TENSORRT, Backend.ORT_GPU, Backend.ORT_CPU)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelArtifact:
    """An exported ONNX plus the sidecar contract written beside it by
    scripts/export_athlete_detector.py."""

    onnx_path: Path
    meta: dict

    @property
    def height(self) -> int:
        return int(self.meta["imgsz"]["height"])

    @property
    def width(self) -> int:
        return int(self.meta["imgsz"]["width"])

    @property
    def names(self) -> dict[int, str]:
        return {int(k): v for k, v in self.meta["names"].items()}

    @property
    def nms_iou(self) -> float:
        return float(self.meta["nms"]["iou"])

    @property
    def nms_max_det(self) -> int:
        return int(self.meta["nms"]["max_det"])

    @property
    def onnx_sha256(self) -> str:
        return self.meta["onnx_sha256"]

    @property
    def engine_path(self) -> Path:
        return self.onnx_path.with_name(f"{self.onnx_path.stem}_fp16.engine")

    @classmethod
    def load(cls, onnx_path: str | Path) -> "ModelArtifact":
        onnx_path = Path(onnx_path)
        sidecar = Path(str(onnx_path) + ".json")
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} not found. Run: python scripts/export_athlete_detector.py"
            )
        if not sidecar.exists():
            raise FileNotFoundError(
                f"{sidecar} not found. The ONNX must ship with the sidecar written by "
                "scripts/export_athlete_detector.py; the runtime reads its class names, "
                "input size and NMS parameters from it."
            )
        meta = json.loads(sidecar.read_text())

        names = {int(k): v for k, v in meta["names"].items()}
        if names != EXPECTED_NAMES:
            raise ValueError(
                f"artifact declares classes {names}; Path B requires exactly "
                f"{EXPECTED_NAMES} (plan section 3.6)"
            )
        if meta.get("nms", {}).get("embedded", False):
            raise ValueError(
                "artifact has NMS embedded in the graph, but the runtime applies NMS "
                "host-side; this would double-suppress. Re-export with nms=False."
            )
        layout = meta.get("output", {}).get("layout")
        if layout != "xywh+cls":
            raise ValueError(
                f"unsupported output layout {layout!r}; postprocess expects "
                "(1, 4 + nc, anchors) in xywh centre form"
            )

        # The ONNX may have been replaced without re-exporting the sidecar.
        actual = sha256_file(onnx_path)
        if actual != meta.get("onnx_sha256"):
            raise ValueError(
                f"{onnx_path.name} does not match its sidecar checksum "
                f"({actual[:12]}... != {str(meta.get('onnx_sha256'))[:12]}...). Re-export."
            )
        return cls(onnx_path=onnx_path, meta=meta)


@dataclass
class BackendSelection:
    """The chosen tier, plus why every better tier was rejected."""

    backend: Backend
    runner: object
    device: torch.device
    rejected: list[tuple[Backend, str]] = field(default_factory=list)

    def describe(self) -> str:
        parts = [f"active={self.backend.value} on {self.device}"]
        parts += [f"{tier.value} rejected: {why}" for tier, why in self.rejected]
        return "; ".join(parts)


def _warm_up(runner, artifact: ModelArtifact, device: torch.device) -> None:
    """One inference on a zero tensor. A tier that cannot do this is not a working tier."""
    x = torch.zeros((1, 3, artifact.height, artifact.width), dtype=torch.float32, device=device)
    out = runner.infer(x)
    if out is None:
        raise RuntimeError("warm-up returned no output")
    expected = int(artifact.meta["output"]["channels"])
    if out.ndim != 3 or out.shape[1] != expected:
        raise RuntimeError(
            f"warm-up output {tuple(out.shape)} disagrees with the sidecar "
            f"(expected channel dim {expected})"
        )


def select_backend(
    artifact: ModelArtifact,
    prefer: Backend | None = None,
    device: str = "cuda:0",
) -> BackendSelection:
    """Try each tier in priority order and return the first that loads and warms up.

    `prefer` pins a single tier instead of walking the chain -- used by the parity gate to
    force one backend, and by tests to force the degraded path. A pinned tier that fails
    raises rather than silently falling back, because "I asked for TensorRT and got CPU"
    is exactly the confusion this module exists to prevent.
    """
    from .ort_runner import OrtRunner
    from .trt_runner import SidecarMismatch, TrtRunner, check_sidecar

    cuda_available = torch.cuda.is_available()
    tiers = (prefer,) if prefer is not None else TIER_ORDER
    rejected: list[tuple[Backend, str]] = []

    for tier in tiers:
        try:
            if tier is Backend.TENSORRT:
                if not cuda_available:
                    raise RuntimeError("no CUDA device")
                if not artifact.engine_path.exists():
                    raise RuntimeError(
                        f"no engine at {artifact.engine_path.name}; "
                        "run scripts/build_trt_engine.py on this host"
                    )
                check_sidecar(artifact.engine_path, artifact.onnx_sha256)
                dev = torch.device(device)
                runner = TrtRunner(artifact.engine_path, device=device)

            elif tier is Backend.ORT_GPU:
                if not cuda_available:
                    raise RuntimeError("no CUDA device")
                dev = torch.device(device)
                runner = OrtRunner(artifact.onnx_path, device=device)

            elif tier is Backend.ORT_CPU:
                dev = torch.device("cpu")
                runner = OrtRunner(artifact.onnx_path, device="cpu")

            else:
                raise ValueError(f"unknown tier {tier!r}")

            _warm_up(runner, artifact, dev)

        except SidecarMismatch as exc:
            # Called out separately: the operator has a specific action to take.
            log.error(
                "TensorRT engine refused: %s. Re-run scripts/build_trt_engine.py on this "
                "host; continuing on the next tier.", exc,
            )
            rejected.append((tier, f"sidecar mismatch: {exc}"))
            if prefer is not None:
                raise RuntimeError(f"pinned backend {tier.value} unavailable: {exc}") from exc
            continue
        except Exception as exc:
            log.info("backend %s unavailable: %s", tier.value, exc)
            rejected.append((tier, str(exc)))
            if prefer is not None:
                raise RuntimeError(f"pinned backend {tier.value} unavailable: {exc}") from exc
            continue

        selection = BackendSelection(backend=tier, runner=runner, device=dev, rejected=rejected)
        if tier is Backend.ORT_CPU:
            log.warning(
                "running on the CPU tier -- degraded mode, not real-time "
                "(~0.73 s/frame at 1280). %s", selection.describe(),
            )
        else:
            log.info("backend selected: %s", selection.describe())
        return selection

    raise RuntimeError(
        "no usable backend. Rejections: "
        + "; ".join(f"{t.value}: {why}" for t, why in rejected)
    )
