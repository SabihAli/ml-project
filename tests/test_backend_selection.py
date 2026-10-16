"""
Backend selection and artifact-contract tests (IMPLEMENTATION_PLAN.md sections 3.1-3.2, 12).

These are the "fake backends" the plan's Phase 1 exit criterion asks for. They exist because
the real fallback paths cannot be exercised on a machine without a GPU (plan section 16
item 2): the *decisions* -- which tier wins, which rejections are fatal, which artifacts are
refused -- are pure logic and can be tested anywhere, so they are tested here rather than
deferred to a CUDA host along with the kernels.

No weights, no GPU, no data. Run: python -m pytest tests/ -q
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from gsr_pipeline.runtime.backend import Backend, ModelArtifact, select_backend
from gsr_pipeline.runtime.trt_runner import SidecarMismatch, check_sidecar, sidecar_path_for

CHANNELS = 6  # 4 box + 2 classes
ANCHORS = 19320


def _write_artifact(tmp_path: Path, **overrides) -> Path:
    """A syntactically valid ONNX + sidecar pair. The 'ONNX' is only bytes -- nothing in
    these tests loads it, because selection is decided before any graph is parsed."""
    onnx_path = tmp_path / "best_736x1280.onnx"
    onnx_path.write_bytes(b"not-a-real-onnx-graph")
    meta = {
        "schema": 1,
        "names": {"0": "athlete", "1": "ball"},
        "imgsz": {"height": 736, "width": 1280},
        "stride": [8, 16, 32],
        "nms": {"embedded": False, "iou": 0.7, "max_det": 300},
        "output": {"name": "output0", "shape": [1, CHANNELS, ANCHORS],
                   "layout": "xywh+cls", "channels": CHANNELS},
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
    }
    meta.update(overrides)
    Path(str(onnx_path) + ".json").write_text(json.dumps(meta))
    return onnx_path


class FakeRunner:
    """Stands in for a real runner: returns a correctly-shaped output, or misbehaves on
    demand so the warm-up gate can be tested."""

    def __init__(self, name="fake", channels=CHANNELS, raise_on_infer=False):
        self.backend_name = name
        self.channels = channels
        self.raise_on_infer = raise_on_infer

    def infer(self, x):
        if self.raise_on_infer:
            raise RuntimeError("simulated CUDA failure on first inference")
        return torch.zeros((1, self.channels, ANCHORS), dtype=torch.float32)

    def close(self):
        pass


# --------------------------------------------------------------------------- artifacts


def test_artifact_loads_and_exposes_the_sidecar_contract(tmp_path):
    art = ModelArtifact.load(_write_artifact(tmp_path))
    assert art.names == {0: "athlete", 1: "ball"}
    assert (art.height, art.width) == (736, 1280)
    # The NMS parameters must come from the sidecar, not from a default, or the parity
    # gate silently scores a different configuration than the one that was trained.
    assert art.nms_iou == 0.7
    assert art.nms_max_det == 300
    assert art.engine_path.name == "best_736x1280_fp16.engine"


def test_artifact_rejects_wrong_class_list(tmp_path):
    onnx = _write_artifact(tmp_path, names={"0": "player", "1": "ball"})
    with pytest.raises(ValueError, match="requires exactly"):
        ModelArtifact.load(onnx)


def test_artifact_rejects_embedded_nms(tmp_path):
    """Embedded NMS plus the runtime's own NMS would double-suppress -- and silently, since
    the output would still look like plausible detections (plan 3.2, decision 7)."""
    onnx = _write_artifact(tmp_path, nms={"embedded": True, "iou": 0.7, "max_det": 300})
    with pytest.raises(ValueError, match="double-suppress"):
        ModelArtifact.load(onnx)


def test_artifact_rejects_checksum_drift(tmp_path):
    """A re-exported ONNX beside a stale sidecar is the realistic way the contract breaks."""
    onnx = _write_artifact(tmp_path)
    onnx.write_bytes(b"a-different-graph-entirely")
    with pytest.raises(ValueError, match="does not match its sidecar checksum"):
        ModelArtifact.load(onnx)


def test_artifact_requires_a_sidecar(tmp_path):
    onnx = _write_artifact(tmp_path)
    Path(str(onnx) + ".json").unlink()
    with pytest.raises(FileNotFoundError, match="export_athlete_detector"):
        ModelArtifact.load(onnx)


def test_artifact_rejects_unknown_output_layout(tmp_path):
    onnx = _write_artifact(
        tmp_path,
        output={"name": "o", "shape": [1, 300, 6], "layout": "unknown", "channels": CHANNELS},
    )
    with pytest.raises(ValueError, match="unsupported output layout"):
        ModelArtifact.load(onnx)


# ------------------------------------------------------------------- tier selection


def test_falls_back_to_cpu_without_cuda(tmp_path, monkeypatch):
    art = ModelArtifact.load(_write_artifact(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "gsr_pipeline.runtime.ort_runner.OrtRunner",
        lambda path, device="cpu", **kw: FakeRunner("ort_cpu"),
    )
    sel = select_backend(art)
    assert sel.backend is Backend.ORT_CPU
    # Both better tiers must be rejected *with a reason*: a silent demotion to CPU is the
    # failure mode this whole module exists to prevent.
    assert [t for t, _ in sel.rejected] == [Backend.TENSORRT, Backend.ORT_GPU]
    assert all("no CUDA device" in why for _, why in sel.rejected)


def test_prefers_tensorrt_when_everything_is_present(tmp_path, monkeypatch):
    onnx = _write_artifact(tmp_path)
    art = ModelArtifact.load(onnx)
    art.engine_path.write_bytes(b"engine")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("gsr_pipeline.runtime.trt_runner.check_sidecar", lambda *a, **k: {})
    monkeypatch.setattr(
        "gsr_pipeline.runtime.trt_runner.TrtRunner",
        lambda path, device="cuda:0": FakeRunner("tensorrt"),
    )
    # device="cpu" because this torch build has no CUDA: the warm-up allocates a real
    # tensor on the selected device, and we are testing tier *ordering*, not allocation.
    sel = select_backend(art, device="cpu")
    assert sel.backend is Backend.TENSORRT
    assert sel.rejected == []


def test_sidecar_mismatch_demotes_to_ort_gpu(tmp_path, monkeypatch):
    """A mismatched engine must not be used, but must also not take the service down --
    the next tier is a correct, merely slower, answer (plan 3.2)."""
    onnx = _write_artifact(tmp_path)
    art = ModelArtifact.load(onnx)
    art.engine_path.write_bytes(b"engine")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(*a, **k):
        raise SidecarMismatch("engine built for gpu_name='RTX 3090', this host has 'RTX 3060'")

    monkeypatch.setattr("gsr_pipeline.runtime.trt_runner.check_sidecar", boom)
    monkeypatch.setattr(
        "gsr_pipeline.runtime.ort_runner.OrtRunner",
        lambda path, device="cpu", **kw: FakeRunner("ort_gpu"),
    )
    sel = select_backend(art, device="cpu")
    assert sel.backend is Backend.ORT_GPU
    assert "sidecar mismatch" in sel.rejected[0][1]


def test_missing_engine_is_reported_with_the_fix(tmp_path, monkeypatch):
    art = ModelArtifact.load(_write_artifact(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "gsr_pipeline.runtime.ort_runner.OrtRunner",
        lambda path, device="cpu", **kw: FakeRunner("ort_gpu"),
    )
    sel = select_backend(art, device="cpu", auto_build=False)
    assert sel.backend is Backend.ORT_GPU
    assert "build_trt_engine.py" in sel.rejected[0][1]


def test_missing_engine_without_tensorrt_says_what_to_install(tmp_path, monkeypatch):
    import sys
    art = ModelArtifact.load(_write_artifact(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "tensorrt", None)      # import tensorrt -> ImportError
    monkeypatch.setattr(
        "gsr_pipeline.runtime.ort_runner.OrtRunner",
        lambda path, device="cpu", **kw: FakeRunner("ort_gpu"),
    )
    sel = select_backend(art, device="cpu")
    assert sel.backend is Backend.ORT_GPU
    assert "tensorrt is not installed" in sel.rejected[0][1]
    assert not art.engine_path.exists()


def test_warm_up_failure_rejects_the_tier(tmp_path, monkeypatch):
    """A tier that loads but throws on first inference is not a working tier -- better to
    find out at startup than on the first frame of a live match.

    The first OrtRunner built is the GPU tier and it fails its warm-up; the second is the
    CPU tier and succeeds, so selection must skip past the broken tier rather than abort.
    """
    art = ModelArtifact.load(_write_artifact(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    built: list[str] = []

    def fake_ort(path, device="cpu", **kw):
        built.append(str(device))
        return FakeRunner("ort", raise_on_infer=len(built) == 1)

    monkeypatch.setattr("gsr_pipeline.runtime.ort_runner.OrtRunner", fake_ort)
    sel = select_backend(art, device="cpu")
    assert sel.backend is Backend.ORT_CPU
    assert len(built) == 2
    assert any("simulated CUDA failure" in why for _, why in sel.rejected)


def test_warm_up_rejects_output_disagreeing_with_the_sidecar(tmp_path, monkeypatch):
    """Catches an engine built from a different model than the sidecar claims."""
    art = ModelArtifact.load(_write_artifact(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "gsr_pipeline.runtime.ort_runner.OrtRunner",
        lambda path, device="cpu", **kw: FakeRunner("ort_cpu", channels=8),
    )
    with pytest.raises(RuntimeError, match="no usable backend"):
        select_backend(art)


def test_pinned_backend_raises_rather_than_falling_back(tmp_path, monkeypatch):
    """`--backend ort_gpu` returning a CPU session would make the parity gate compare the
    wrong thing, so a pinned tier fails loudly."""
    art = ModelArtifact.load(_write_artifact(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="pinned backend ort_gpu unavailable"):
        select_backend(art, prefer=Backend.ORT_GPU)


# ------------------------------------------------------------------ engine sidecar


def test_check_sidecar_requires_one(tmp_path):
    engine = tmp_path / "e.engine"
    engine.write_bytes(b"x")
    with pytest.raises(SidecarMismatch, match="no sidecar"):
        check_sidecar(engine, "deadbeef")


def test_check_sidecar_catches_a_re_exported_onnx(tmp_path):
    engine = tmp_path / "e.engine"
    engine.write_bytes(b"x")
    sidecar_path_for(engine).write_text(json.dumps({
        "schema": 1, "onnx_sha256": "a" * 64, "environment": {},
    }))
    with pytest.raises(SidecarMismatch, match="different ONNX"):
        check_sidecar(engine, "b" * 64)


# ------------------------------------------------------------------ engine auto-build


@pytest.fixture
def fake_trt_host(monkeypatch):
    """A host that looks like it has CUDA and TensorRT, with a builder that records calls
    instead of compiling anything."""
    import sys
    import types

    from gsr_pipeline.runtime import trt_runner

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "tensorrt", types.ModuleType("tensorrt"))
    calls = []

    def fake_build(onnx_path, engine_path, onnx_sha256, fp16=True, workspace_gb=4.0):
        calls.append(Path(engine_path))
        Path(engine_path).write_bytes(b"engine")
        sidecar_path_for(engine_path).write_text(json.dumps({"schema": 1, "onnx_sha256": onnx_sha256}))
        return Path(engine_path)

    monkeypatch.setattr(trt_runner, "build_engine", fake_build)
    return calls


def test_ensure_engine_leaves_an_existing_engine_alone(tmp_path, fake_trt_host):
    from gsr_pipeline.runtime.trt_runner import ensure_engine
    engine = tmp_path / "m_fp16.engine"
    engine.write_bytes(b"old")
    assert ensure_engine(tmp_path / "m.onnx", engine, "a" * 64) == engine
    assert fake_trt_host == [] and engine.read_bytes() == b"old"


def test_ensure_engine_builds_a_missing_engine_once(tmp_path, fake_trt_host):
    from gsr_pipeline.runtime.trt_runner import ensure_engine
    engine = tmp_path / "m_fp16.engine"
    ensure_engine(tmp_path / "m.onnx", engine, "a" * 64)
    ensure_engine(tmp_path / "m.onnx", engine, "a" * 64)
    assert len(fake_trt_host) == 1                       # second launch loads, no rebuild
    assert engine.read_bytes() == b"engine"
    assert json.loads(sidecar_path_for(engine).read_text())["onnx_sha256"] == "a" * 64
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m_fp16.engine", "m_fp16.engine.json"]


def test_ensure_engine_off_means_no_build(tmp_path, fake_trt_host):
    from gsr_pipeline.runtime.trt_runner import ensure_engine
    with pytest.raises(FileNotFoundError, match="auto-build is off"):
        ensure_engine(tmp_path / "m.onnx", tmp_path / "m_fp16.engine", "a" * 64, auto_build=False)
    assert fake_trt_host == []


def test_ensure_engine_refuses_while_another_build_holds_the_lock(tmp_path, fake_trt_host):
    from gsr_pipeline.runtime.trt_runner import ensure_engine
    (tmp_path / "m_fp16.engine.building").write_text("")
    with pytest.raises(RuntimeError, match="another process is building"):
        ensure_engine(tmp_path / "m.onnx", tmp_path / "m_fp16.engine", "a" * 64)
    assert fake_trt_host == []


def test_ensure_engine_failed_build_leaves_nothing_behind(tmp_path, monkeypatch, fake_trt_host):
    from gsr_pipeline.runtime import trt_runner

    def broken_build(onnx_path, engine_path, onnx_sha256, fp16=True, workspace_gb=4.0):
        Path(engine_path).write_bytes(b"half-written")
        raise RuntimeError("builder crashed")

    monkeypatch.setattr(trt_runner, "build_engine", broken_build)
    with pytest.raises(RuntimeError, match="builder crashed"):
        trt_runner.ensure_engine(tmp_path / "m.onnx", tmp_path / "m_fp16.engine", "a" * 64)
    assert list(tmp_path.iterdir()) == []                # no engine, no temp file, no lock


def test_select_backend_builds_a_missing_engine_then_loads_it(tmp_path, monkeypatch, fake_trt_host):
    art = ModelArtifact.load(_write_artifact(tmp_path))
    monkeypatch.setattr("gsr_pipeline.runtime.trt_runner.check_sidecar", lambda *a, **k: {})
    monkeypatch.setattr("gsr_pipeline.runtime.trt_runner.TrtRunner",
                        lambda *a, **k: FakeRunner("tensorrt"))
    sel = select_backend(art, device="cpu")
    assert sel.backend is Backend.TENSORRT
    assert fake_trt_host == [art.engine_path.with_name(art.engine_path.name + ".tmp")]
    assert art.engine_path.exists()


def test_select_backend_without_auto_build_falls_back(tmp_path, monkeypatch, fake_trt_host):
    art = ModelArtifact.load(_write_artifact(tmp_path))
    monkeypatch.setattr("gsr_pipeline.runtime.ort_runner.OrtRunner",
                        lambda *a, **k: FakeRunner("ort"))
    sel = select_backend(art, device="cpu", auto_build=False)
    assert sel.backend is Backend.ORT_GPU
    assert "auto-build is off" in sel.rejected[0][1]
    assert fake_trt_host == []
