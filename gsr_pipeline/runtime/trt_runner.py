"""
trt_runner.py -- TensorRT backend (tier 1) and the engine builder behind it.

TensorRT engines are specific to the GPU model, driver and TRT version, so they cannot be
shipped. The ONNX is the shipped artifact; each production host builds its own engine on
first install (IMPLEMENTATION_PLAN.md section 3.2). `build_engine` here is what
scripts/build_trt_engine.py drives.

Every engine gets a sidecar recording the environment it was built for. At startup a
mismatch is a REFUSAL, not a warning: running an engine built for a different GPU or TRT
version is undefined behaviour, and falling back to ORT GPU is a far better outcome than a
plausible-looking wrong answer. The plan (3.2) is explicit that this logs at ERROR telling
the operator to re-run the builder.

NOTE (2026-09-26): none of this has executed. The dev machine has no NVIDIA GPU (plan
section 16 item 2), so tier 1 is code-complete but UNVERIFIED -- it is Phase 1b, to be
closed on a CUDA host. Treat the TensorRT API calls here as reviewed-but-untested.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

log = logging.getLogger(__name__)

SIDECAR_SCHEMA = 1


class SidecarMismatch(RuntimeError):
    """The engine on disk was built for a different environment than this one."""


@dataclass(frozen=True)
class EngineEnvironment:
    """The facts an engine is only valid for."""

    gpu_name: str
    compute_capability: str
    driver: str
    cuda: str
    tensorrt: str

    @staticmethod
    def current(trt_module=None) -> "EngineEnvironment":
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device; cannot describe a TensorRT environment")
        major, minor = torch.cuda.get_device_capability(0)
        if trt_module is None:
            import tensorrt as trt_module
        return EngineEnvironment(
            gpu_name=torch.cuda.get_device_name(0),
            compute_capability=f"{major}.{minor}",
            driver=getattr(torch.version, "cuda", "") or "",
            cuda=torch.version.cuda or "",
            tensorrt=trt_module.__version__,
        )

    def as_dict(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "compute_capability": self.compute_capability,
            "driver": self.driver,
            "cuda": self.cuda,
            "tensorrt": self.tensorrt,
        }


def sidecar_path_for(engine_path: str | Path) -> Path:
    return Path(str(engine_path) + ".json")


def check_sidecar(engine_path: str | Path, onnx_sha256: str) -> dict:
    """Load an engine's sidecar and verify it matches this host. Raises SidecarMismatch.

    Checked: the source ONNX checksum (so a re-exported model cannot be served by a stale
    engine), GPU name, compute capability, and the CUDA and TensorRT versions.
    """
    path = sidecar_path_for(engine_path)
    if not path.exists():
        raise SidecarMismatch(f"no sidecar beside {engine_path}; refusing the engine")
    meta = json.loads(path.read_text())

    if meta.get("schema") != SIDECAR_SCHEMA:
        raise SidecarMismatch(f"sidecar schema {meta.get('schema')} != {SIDECAR_SCHEMA}")
    if meta.get("onnx_sha256") != onnx_sha256:
        raise SidecarMismatch(
            "engine was built from a different ONNX "
            f"({str(meta.get('onnx_sha256'))[:12]}... != {onnx_sha256[:12]}...)"
        )

    current = EngineEnvironment.current().as_dict()
    built = meta.get("environment", {})
    for key in ("gpu_name", "compute_capability", "cuda", "tensorrt"):
        if built.get(key) != current.get(key):
            raise SidecarMismatch(
                f"engine built for {key}={built.get(key)!r}, this host has {current.get(key)!r}"
            )
    return meta


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    onnx_sha256: str,
    fp16: bool = True,
    workspace_gb: float = 4.0,
) -> Path:
    """Build a static-batch-1 engine from the ONNX and write its sidecar.

    Builder optimisation level is set to its maximum: this runs once per host at install
    time, so a slower build for a faster engine is the right trade (plan 3.2).
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "tensorrt is required to build an engine. Install the NVIDIA TensorRT Python "
            "package matching this host's CUDA version."
        ) from exc

    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse {onnx_path}: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        if not builder.platform_has_fast_fp16:
            log.warning("platform reports no fast FP16; building anyway")
        config.set_flag(trt.BuilderFlag.FP16)
    try:
        config.builder_optimization_level = 5  # maximum; TensorRT >= 8.6
    except AttributeError:
        log.info("builder_optimization_level unavailable on TensorRT %s", trt.__version__)

    log.info("building TensorRT engine from %s (fp16=%s); this takes minutes", onnx_path.name, fp16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no engine; see the builder log above")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)

    sidecar_path_for(engine_path).write_text(
        json.dumps(
            {
                "schema": SIDECAR_SCHEMA,
                "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "onnx": onnx_path.name,
                "onnx_sha256": onnx_sha256,
                "fp16": fp16,
                "environment": EngineEnvironment.current(trt).as_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    log.info("wrote %s (%.1f MB) and its sidecar", engine_path, engine_path.stat().st_size / 1e6)
    return engine_path


class TrtRunner:
    """Deserialise once, preallocate once, `execute_async_v3` per frame."""

    backend_name = "tensorrt"

    def __init__(self, engine_path: str | Path, device: str = "cuda:0"):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError("tensorrt is required for the TensorRT backend.") from exc

        self._trt = trt
        self.engine_path = Path(engine_path)
        self.device = torch.device(device)

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {self.engine_path}")
        self.context = self.engine.create_execution_context()

        # Preallocate every binding as a torch CUDA tensor, then hand TRT the pointers.
        # No per-frame allocation in steady state (plan principle 4).
        self.buffers: dict[str, torch.Tensor] = {}
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            dtype = torch.from_numpy(np_dtype(0).reshape(1)).dtype
            self.buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]
        self.input_shape = tuple(self.buffers[self.input_name].shape)
        log.info(
            "TrtRunner loaded %s  input=%s%s  outputs=%s",
            self.engine_path.name, self.input_name, self.input_shape, self.output_names,
        )

    def infer(self, x: torch.Tensor, stream: "torch.cuda.Stream | None" = None) -> torch.Tensor:
        """Run one forward pass on `stream` (default: the current stream).

        The caller owns synchronisation. On the live path S1 records CUDA events around
        this rather than synchronising, so GPU time is measured without stalling the
        pipeline (plan 10.2).
        """
        dst = self.buffers[self.input_name]
        dst.copy_(x.to(dtype=dst.dtype, non_blocking=True))
        cuda_stream = (stream or torch.cuda.current_stream(self.device)).cuda_stream
        if not self.context.execute_async_v3(stream_handle=cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return self.buffers[self.output_name]

    def close(self) -> None:
        self.context = None
        self.engine = None
        self.buffers.clear()
