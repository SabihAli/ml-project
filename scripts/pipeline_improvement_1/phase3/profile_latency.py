"""
profile_latency.py -- one-shot latency profile of the live path on this machine, for Phase 3 of
docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md: how much per-frame GPU/CPU headroom is left for the
planned additions (temporal detector input, batched c5c windows, the event model).

Runs the real pipeline objects from configs/default.yaml (GSRPipeline: detector, c5c BallFuser,
ByteTrack, LiveCalibrator, LiveAthleteRoles) over frames already in memory, and times:
  1. per frame, end to end: detect -> ball fusion -> track -> calibrate -> project -> roles
     (process_stream's fast path; jersey OCR is off the fast path and not included)
  2. each stage inside it (wrapped methods: detector, c5c call, tracker, calibrator, NBJW, roles)
  3. c5c alone on a fixed window (its cost per call, whether or not the fuser arms it)
  4. frame decode (cv2.imread of one 1080p JPEG; NVDEC is the plan, not built)
  5. optional --candidate ONNX models (future students, e.g. a 5-channel detector), random input,
     on onnxruntime TensorRT EP (FP16) if present, else CUDA EP
and compares e2e p95 with the plan's budget (IMPLEMENTATION_PLAN.md §1.4: p95 <= 25 ms target,
40 ms hard ceiling = 25 fps). Stages run back to back on one thread, as process_stream does today.

    python scripts/pipeline_improvement_1/phase3/profile_latency.py \
        --sequence_dir data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021 \
        --out data/outputs/pipeline_improvement_1/phase3/latency --device cuda:0 --calib_backend auto [--frames 600] \
        [--candidate runs/x/detector_5ch.onnx:1,5,736,1280]

Outputs (--out): latency_report.md, latency.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

TARGET_P95_MS, CEILING_MS = 25.0, 40.0
DEVICE_SYNC = True       # off in async calibration: a device-wide sync would wait for the background NBJW too


def _sync():
    if not DEVICE_SYNC:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class Timers:
    def __init__(self):
        self.t = defaultdict(list)

    def wrap(self, obj, attr, name):
        fn = getattr(obj, attr)

        def timed(*a, **k):
            _sync()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            _sync()
            self.t[name].append((time.perf_counter() - t0) * 1000)
            return r
        setattr(obj, attr, timed)

    def reset(self):
        self.t.clear()


def stats(xs, n_frames=None):
    a = np.asarray(xs, float)
    if not len(a):
        return None
    s = {"calls": int(len(a)), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
         "p95": float(np.percentile(a, 95)), "max": float(a.max())}
    if n_frames:
        s["per_frame"] = float(a.sum() / n_frames)          # amortised cost per frame
    return s


def env_info(pipe):
    info = {"host": platform.node(), "python": platform.python_version(), "time": datetime.now().isoformat(timespec="seconds")}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    except ImportError:
        info["torch"] = "missing"
    for mod in ("onnxruntime", "tensorrt"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", None) or "unknown (no __version__)"
        except ImportError:
            info[mod] = "missing"
    info["detector_backend"] = str(pipe.detector.backend)
    info["specialist_backend"] = pipe.ball_specialist.backend.value if pipe.ball_specialist else "off"
    c = pipe.cfg.get("calibration", {})
    info["calibration"] = {k: c.get(k) for k in ("backend", "calib_every_k", "mode", "fp16")}
    return info


def time_candidate(spec, iters, warmup):
    import onnxruntime as ort
    path, shape = spec.rsplit(":", 1)
    shape = [int(v) for v in shape.split(",")]
    provs = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider") if p in ort.get_available_providers()]
    opts = {"TensorrtExecutionProvider": {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                          "trt_engine_cache_path": str(Path(path).parent)}}
    sess = ort.InferenceSession(path, providers=[(p, opts.get(p, {})) for p in provs] or ["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(*shape).astype(np.float16 if "16" in sess.get_inputs()[0].type else np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000)
    return {"model": path, "shape": shape, "provider": sess.get_providers()[0], **stats(ts)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence_dir", type=Path, required=True, help="a GSR sequence folder (with img1/)")
    ap.add_argument("--config", type=Path, default=ROOT / "gsr_pipeline" / "configs" / "default.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "outputs" / "pipeline_improvement_1" / "phase3" / "latency")
    ap.add_argument("--device", default="", help="override every model's device, e.g. cuda:0 (default.yaml has "
                    "calibration on cpu)")
    ap.add_argument("--calib_backend", default="", help="override calibration.backend: torch | auto | tensorrt | "
                    "ort_gpu (non-torch needs the NBJW ONNX from scripts/path_b/calibration/export_nbjw.py)")
    ap.add_argument("--calib_mode", default="", choices=["", "inline", "async"],
                    help="override calibration.mode (async: NBJW on a background thread)")
    ap.add_argument("--calib_k", type=int, default=0, help="override calibration.calib_every_k")
    ap.add_argument("--frames", type=int, default=600, help="frames timed (after warm-up)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--spec_iters", type=int, default=300)
    ap.add_argument("--candidate", action="append", default=[], help="path.onnx:N,C,H,W (repeatable)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import cv2
    import yaml
    from gsr_pipeline.pipeline import GSRPipeline

    cfg = yaml.safe_load(a.config.read_text())
    if a.device:                                       # as run.py --device: every model on this device
        for sec in ("detector", "ball_specialist", "reid", "calibration"):
            cfg.setdefault(sec, {})["device"] = a.device
        cfg.setdefault("jersey", {})["gpu"] = "cuda" in a.device     # EasyOCR takes gpu: bool, not device
    if a.calib_backend:
        cfg.setdefault("calibration", {})["backend"] = a.calib_backend
    if a.calib_mode:
        cfg.setdefault("calibration", {})["mode"] = a.calib_mode
    if a.calib_k:
        cfg.setdefault("calibration", {})["calib_every_k"] = a.calib_k
    global DEVICE_SYNC
    # Every stage returns host arrays, so timings hold without a device sync. In async mode a
    # device-wide sync would also wait for the background NBJW and hide what async buys.
    DEVICE_SYNC = cfg.get("calibration", {}).get("mode", "inline") != "async"
    t0 = time.time()
    pipe = GSRPipeline(cfg)
    load_s = time.time() - t0
    env = env_info(pipe)
    print(json.dumps(env, indent=1), f"\nmodels loaded in {load_s:.0f}s", flush=True)

    files = sorted((a.sequence_dir / "img1").glob("*.jpg"))
    need = a.warmup + a.frames
    dec = []
    frames = []
    for f in files[:need]:
        t1 = time.perf_counter()
        frames.append(cv2.imread(str(f)))
        dec.append((time.perf_counter() - t1) * 1000)
    while len(frames) < need:                                # short sequence: loop it
        frames += frames[: need - len(frames)]
    print(f"{len(frames)} frames in memory ({a.sequence_dir.name})", flush=True)

    T = Timers()
    T.wrap(pipe.detector, "detect", "detector")
    if pipe.ball_specialist:
        T.wrap(pipe.ball_specialist, "detect", "c5c (when armed)")
    T.wrap(pipe.tracker, "update", "tracker")
    T.wrap(pipe.kp_detector, "detect", "NBJW keypoints (every K / trigger)")

    fuser, roles = pipe._new_ball_fuser(), pipe._new_live_roles()
    pipe.calibrator.close()
    pipe.calibrator = pipe._new_calibrator()
    T.wrap(pipe.calibrator, "update", "calibrator (incl. NBJW)")
    if pipe.calibrator.motion is not None:
        T.wrap(pipe.calibrator.motion, "update", "camera motion")
        T.wrap(pipe.calibrator.motion, "_prepare", "camera motion: resize/grey/mask")
    if pipe.calibrator.mode == "async":       # gated path skips kp_detector.detect: time the whole run
        T.wrap(pipe.calibrator, "_run_nbjw_here", "NBJW run, background wall time incl. frame-tick waits")
    T.wrap(fuser, "update", "ball fusion (incl. c5c)")
    T.wrap(roles, "update", "roles")

    e2e = []
    for i, img in enumerate(frames):
        if i == a.warmup:
            T.reset()
            e2e.clear()
        _sync()
        t1 = time.perf_counter()
        df, *_ = pipe._detect_track_calibrate(i, img, fuser)
        roles.update(i, img, df)
        _sync()
        e2e.append((time.perf_counter() - t1) * 1000)
        if (i + 1) % 100 == 0:
            print(f"  frame {i + 1}/{len(frames)}: e2e p50 {np.percentile(e2e, 50):.1f} ms", flush=True)
    n = len(e2e)
    pipe.calibrator.close()
    stages = {k: stats(v, n) for k, v in T.t.items()}

    spec_alone = None
    if pipe.ball_specialist:
        crop = pipe.ball_specialist.crop
        k = min(100, len(frames) - 3)
        win = frames[k:k + 3]
        x0, y0 = (win[0].shape[1] - crop) // 2, (win[0].shape[0] - crop) // 2
        for _ in range(20):
            pipe.ball_specialist.detect(win, x0, y0)
        T.reset()
        for _ in range(a.spec_iters):
            pipe.ball_specialist.detect(win, x0, y0)
        spec_alone = stats(T.t["c5c (when armed)"])

    cands = []
    for c in a.candidate:
        try:
            cands.append(time_candidate(c, 300, 30))
        except Exception as e:                                  # report, don't abort the profile
            cands.append({"model": c, "error": repr(e)[:300]})

    e = stats(e2e)
    res = {"env": env, "sequence": a.sequence_dir.name, "frames": n, "e2e": e, "fps_sequential": 1000 / e["mean"],
           "stages": stages, "c5c_alone": spec_alone, "decode_imread": stats(dec), "candidates": cands,
           "budget": {"target_p95_ms": TARGET_P95_MS, "ceiling_ms": CEILING_MS,
                      "headroom_vs_target_ms": TARGET_P95_MS - e["p95"], "headroom_vs_ceiling_ms": CEILING_MS - e["p95"]}}
    (a.out / "latency.json").write_text(json.dumps(res, indent=1))

    f = lambda s: "-" if s is None else f"{s['mean']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} | {s['max']:.1f}"  # noqa: E731
    L = ["# Live-path latency profile", "",
         f"{env.get('gpu')} | torch {env.get('torch')} (CUDA {env.get('cuda')}) | onnxruntime {env.get('onnxruntime')} | "
         f"TensorRT {env.get('tensorrt')} | {env['time']}", "",
         f"Backends: detector `{env['detector_backend']}`, c5c `{env['specialist_backend']}`, calibration "
         f"`{env['calibration']}`. Sequence {a.sequence_dir.name}, {n} frames after {a.warmup} warm-up, frames in "
         "memory, stages back to back on one thread (as process_stream runs today)"
         + ("; async calibration: NBJW on a background thread, no device-wide sync in the timers, "
            "so the NBJW row is its time on that thread, off the frame loop" if not DEVICE_SYNC else "") + ".", "",
         "## End to end (detect -> ball fusion -> track -> calibrate -> project -> roles)", "",
         "| | mean | p50 | p95 | max |", "|---|---|---|---|---|",
         f"| ms per frame | {f(e)} |", "",
         f"Sequential throughput **{res['fps_sequential']:.1f} fps**. Budget (§1.4): p95 <= {TARGET_P95_MS:.0f} ms target, "
         f"{CEILING_MS:.0f} ms ceiling. **Headroom at p95: {res['budget']['headroom_vs_target_ms']:+.1f} ms vs target, "
         f"{res['budget']['headroom_vs_ceiling_ms']:+.1f} ms vs ceiling.**", "",
         "## Stages", "", "Nested: ball fusion includes c5c; calibrator includes NBJW. `per frame` = total / frames.", "",
         "| stage | calls | mean ms | p50 | p95 | max | per frame ms |", "|---|---|---|---|---|---|---|"]
    for k, s in stages.items():
        L.append(f"| {k} | {s['calls']} | {s['mean']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} | {s['max']:.1f} | {s['per_frame']:.2f} |")
    L += ["", "## Isolated", "", "| | mean | p50 | p95 | max |", "|---|---|---|---|---|",
          f"| c5c, one call ({a.spec_iters} x) | {f(spec_alone)} |",
          f"| JPEG decode, cv2.imread 1080p | {f(stats(dec))} |"]
    if cands:
        L += ["", "## Candidate models (random input)", "", "| model | input | provider | mean | p50 | p95 | max |",
              "|---|---|---|---|---|---|---|"]
        for c in cands:
            L.append(f"| {c['model']} | error: {c['error']} | | | | | |" if "error" in c else
                     f"| {c['model']} | {c['shape']} | {c['provider']} | {f(c)} |")
    (a.out / "latency_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
