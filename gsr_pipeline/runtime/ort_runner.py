"""
ort_runner.py -- ONNX Runtime backend (tiers 2 and 3).

Tier 2 is CUDAExecutionProvider with IOBinding onto torch GPU buffers, so no full frame
crosses the PCIe bus. Tier 3 is the CPU provider -- degraded mode, ~0.73 s/frame at 1280
(blockers doc section 2), kept because it is the only tier that runs everywhere and so is
the one the parity gate can always be closed against.

The critical behaviour here is that the active provider is READ BACK, never inferred: ORT
silently falls back to CPU when the CUDA provider fails to load, which would otherwise
present as a mysterious 100x slowdown rather than as an error
(IMPLEMENTATION_PLAN.md section 3.1).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


class OrtRunner:
    """Thin ONNX Runtime wrapper with one input and one output tensor."""

    def __init__(self, onnx_path: str | Path, device: str = "cpu", intra_op_threads: int = 0):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # repo convention: lazy import, chained, with a hint
            raise ImportError(
                "onnxruntime is required for the ORT backend. Install 'onnxruntime-gpu' "
                "(GPU hosts) or 'onnxruntime' (CPU-only hosts)."
            ) from exc

        self._ort = ort
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        want_cuda = str(device).startswith("cuda")
        available = ort.get_available_providers()
        if want_cuda and CUDA_PROVIDER not in available:
            raise RuntimeError(
                f"{CUDA_PROVIDER} not available in this onnxruntime build "
                f"(have: {available}). Install onnxruntime-gpu."
            )
        requested = [CUDA_PROVIDER, CPU_PROVIDER] if want_cuda else [CPU_PROVIDER]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            opts.intra_op_num_threads = intra_op_threads

        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=requested)

        # Read the truth back rather than trusting `requested`.
        self.providers = self.session.get_providers()
        self.uses_cuda = CUDA_PROVIDER in self.providers
        if want_cuda and not self.uses_cuda:
            raise RuntimeError(
                f"requested {CUDA_PROVIDER} but the session is running on {self.providers}; "
                "refusing to run a 'GPU' tier silently on the CPU"
            )

        self.device = torch.device(device if self.uses_cuda else "cpu")
        self._input = self.session.get_inputs()[0]
        self._output = self.session.get_outputs()[0]
        self.input_name, self.output_name = self._input.name, self._output.name
        self.input_shape = tuple(self._input.shape)

        # Preallocated once, reused every call on the CUDA path (plan principle 4,
        # "allocate once") -- _infer_bound used to torch.empty() a fresh output tensor per
        # frame, a CUDA allocation on every inference that TrtRunner never does.
        self._output_buffer = None
        if self.uses_cuda:
            out_shape = tuple(d if isinstance(d, int) else 1 for d in self._output.shape)
            self._output_buffer = torch.empty(out_shape, dtype=torch.float32, device=self.device)

        log.info(
            "OrtRunner loaded %s  providers=%s  input=%s%s",
            self.onnx_path.name, self.providers, self.input_name, self.input_shape,
        )

    @property
    def backend_name(self) -> str:
        return "ort_gpu" if self.uses_cuda else "ort_cpu"

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass. Returns the raw head output as a torch tensor.

        On CUDA the input stays on the device via IOBinding; on CPU it round-trips through
        numpy, which is the cheaper of the two options at this tier's speed anyway.
        """
        if self.uses_cuda:
            return self._infer_bound(x)
        out = self.session.run(
            [self.output_name],
            {self.input_name: np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)},
        )[0]
        return torch.from_numpy(out)

    def _infer_bound(self, x: torch.Tensor) -> torch.Tensor:
        """CUDA path: bind torch device pointers directly, so ORT reads and writes our
        buffers instead of copying through host memory."""
        x = x.to(self.device, dtype=torch.float32).contiguous()
        binding = self.session.io_binding()
        device_id = self.device.index or 0
        binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr(),
        )
        # Bind the preallocated output buffer -- no per-call CUDA allocation.
        out = self._output_buffer
        binding.bind_output(
            name=self.output_name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(out.shape),
            buffer_ptr=out.data_ptr(),
        )
        self.session.run_with_iobinding(binding)
        return out

    def close(self) -> None:
        self.session = None
