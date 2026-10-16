"""
path_b_eval.py -- one-shot Path B evaluation runner (Kaggle / any CUDA host).

Runs every Path B measurement we can make today over a set of SoccerNet-GSR sequences and
writes one summary:

  parity      GPU-tier parity gate (scripts/parity_gate.py) per backend: ort_gpu, tensorrt.
  detection   per-class detector accuracy (evaluate_detection, 2-class mode), pooled over all
              sequences; optionally against the Ultralytics .pt reference (plan 3.5 gate).
  roles       role/team classifier accuracy on GT boxes + GT pitch positions
              (evaluate_athlete_roles protocol: fit on the first N frames, score the rest).
  gshota      END-TO-END Path B -> predictions.json -> GS-HOTA (evaluate.py, unchanged).

The end-to-end flow is the Phase 9 batch path of docs/path_b/plans/IMPLEMENTATION_PLAN.md,
built from existing parts and nothing new in gsr_pipeline/:

    AthleteBallDetector -> ByteTracker (athletes) -> LiveCalibrator (NBJW every K frames, camera-motion
      propagation between) -> pitch projection
      -> AthleteRoleTeamClassifier (fit on the first --fit_frames frames, predict all)
      -> per-tracklet vote on role / team -> save_results_json -> evaluate_gs_hota

What this deliberately does NOT do, so read GS-HOTA numbers with it in mind:
  * No tracklet refinement (refine.py needs ReID embeddings; Path B's live design drops ReID).
  * Jersey OCR is OFF unless --jersey. GS-HOTA's IdSim requires a jersey match whenever the GT
    has a jersey, so with OCR off the GT jersey is blanked and the score is labelled
    "jersey ignored". It is NOT comparable to a Path A number that includes jersey.
  * The ball is the single highest-confidence ball detection per frame under one constant
    track id. The real ball tracker is plan Phase 6 and does not exist yet.
  * Detections the tracker leaves unassociated become one-frame singleton tracklets instead of
    being dropped.

Per-sequence results are cached in <output_dir>/<sequence>/result.json, so an interrupted run
(Kaggle's 12h limit) resumes where it stopped. --force recomputes.

Kaggle quick start (cells):
    !cp -r /kaggle/input/bk-cvservice/* /kaggle/working/cv/ && cd /kaggle/working/cv
    !pip uninstall -y -q onnxruntime && pip install -q "ultralytics>=8.4.0" onnx onnxruntime-gpu yacs scikit-learn
    !pip install -q -e gsr_pipeline/nbjw_calib
    !python notebooks/path_b_eval.py --data_root data/datasets/SoccerNetGS/gamestate-2024/valid \\
        --output_dir /kaggle/working/out --limit 3 --max_frames 100        # smoke test first
    !python notebooks/path_b_eval.py --data_root data/datasets/SoccerNetGS/gamestate-2024/valid \\
        --output_dir /kaggle/working/out --reference                       # full run
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = ("parity", "detection", "roles", "gshota")
ATHLETE_CLASS, BALL_CLASS = 0, 1
BALL_TRACK_ID = 90_000          # constant id for the single fused ball (stand-in for plan 5.4)
SINGLETON_ID_START = 100_000    # ids for detections the tracker did not associate
PARITY_PREFERRED = ("SNGS-021", "SNGS-022", "SNGS-033", "SNGS-047", "SNGS-090")
TEAM_SIDE = {"a": "left", "b": "right"}   # team_a = lower mean pitch x, as team.py labels "left"
HOTA_VARIANTS = (("persons", True), ("with_ball", False))   # (name, drop_ball)

# Per-frame stages summed into the whole-pipeline latency, in execution order.
PIPELINE_STAGES = ("frame_load", "detect", "track", "calibrate", "project", "describe", "jersey")
CALIB_STAGES = ("calibrate", "project")     # what the plan's fast path does not wait for
LATENCY_ROWS = (
    ("frame_load", "frame load (cv2.imread of a JPEG; production uses NVDEC)"),
    ("detect", "detector end-to-end (preprocess + inference + NMS + D2H)"),
    ("track", "tracker (ByteTrack)"),
    ("calibrate", "calibration (LiveCalibrator: NBJW on fresh frames, motion + propagation otherwise)"),
    ("calibrate_nbjw", "  calibration on frames where NBJW ran inline"),
    ("calibrate_no_nbjw", "  calibration on the other frames (motion + propagation)"),
    ("project", "pitch projection"),
    ("describe", "torso descriptors + record assembly"),
    ("jersey", "jersey OCR (frames where it runs; 0 elsewhere)"),
    ("total", "WHOLE PIPELINE, per frame"),
    ("total_no_calib", "whole pipeline minus calibration + projection"),
)
PROFILE_ROWS = (
    ("det_preprocess", "detector: preprocess (CPU letterbox + H2D)"),
    ("det_infer", "detector: inference (GPU, synced)"),
    ("det_postprocess", "detector: decode + NMS + D2H"),
    ("calib_prep", "calibration: image prep (H2D of uint8 frame + on-device resize)"),
    ("calib_kp_net", "calibration: keypoint network (synced)"),
    ("calib_line_net", "calibration: line network (synced)"),
    ("calib_decode_peaks", "calibration: decode, heatmap peaks (local-max pool + top-k + D2H, both nets)"),
    ("calib_decode_dict", "calibration: decode, peaks -> keypoint dicts"),
    ("calib_decode_complete", "calibration: decode, keypoint completion (line intersections)"),
    ("calib_cam_update", "calibration: camera update (add keypoints)"),
    ("calib_ransac", "calibration: homography (RANSAC)"),
    ("calib_voting", "calibration: camera-param voting (compute_cam_params is on)"),
    ("calib_decode_other", "(not on the path) the other decode implementation, same heatmaps"),
)
# Steps that run on a real NBJW frame; their sum is compared with the main pass.
PROFILE_ON_PATH = ("calib_prep", "calib_kp_net", "calib_line_net", "calib_decode_peaks", "calib_decode_dict",
                   "calib_decode_complete", "calib_cam_update", "calib_ransac", "calib_voting")


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def _np_default(obj):
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def _json_safe(obj):
    return json.loads(json.dumps(obj, default=_np_default))


def _prf(tp: int, fp: int, fn: int) -> dict:
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    if recall is None or precision is None:
        f1 = None
    elif recall == 0.0 and precision == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall, "precision": precision, "f1": f1}


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.2f}%"


def _image_id(image_id, file_name: str) -> int:
    try:
        return int(image_id)
    except (TypeError, ValueError):
        return int(Path(file_name).stem)


def _sequence_dirs(data_root: Path, names, limit) -> list:
    if names:
        dirs = [data_root / n for n in names]
    else:
        dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    dirs = [d for d in dirs if (d / "Labels-GameState.json").exists()]
    return dirs[:limit] if limit else dirs


def _split_label(label: str):
    """athlete_roles label -> (role, team letter or None)."""
    role = label.split("_")[0]
    team = label[-1] if label.startswith(("player_team", "goalkeeper_team")) else None
    return role, team


# --------------------------------------------------------------------------------------
# environment header
# --------------------------------------------------------------------------------------

def environment() -> dict:
    info = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    for mod in ("onnxruntime", "tensorrt", "ultralytics"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import onnxruntime as ort

        info["ort_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    return info


# --------------------------------------------------------------------------------------
# stage: parity (subprocess to the existing gate script)
# --------------------------------------------------------------------------------------

def run_parity(args, seq_dirs, out_dir: Path) -> dict:
    ordered = sorted(seq_dirs, key=lambda d: (d.name not in PARITY_PREFERRED, d.name))[:5]
    if len(ordered) < 3:
        return {"skipped": f"parity gate needs >= 3 sequences, have {len(ordered)}"}

    result = {"sequences": [d.name for d in ordered], "backends": {}}
    if args.build_trt and "tensorrt" in args.parity_backends:
        print("\n=== building the TensorRT engine ===", flush=True)
        rc = subprocess.run([sys.executable, "scripts/build_trt_engine.py"], cwd=ROOT).returncode
        result["trt_build_returncode"] = rc

    for backend in args.parity_backends:
        out_json = out_dir / f"parity_{backend}.json"
        print(f"\n=== parity gate: {backend} ===", flush=True)
        cmd = [sys.executable, "scripts/parity_gate.py",
               "--frames_per_sequence", str(args.parity_frames), "--backend", backend,
               "--device", args.device, "--weights", args.weights,
               "--output_json", str(out_json), "--sequences", *[str(d) for d in ordered]]
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        entry = {"returncode": rc, "pass": rc == 0}
        if out_json.exists():
            entry["variants"] = [
                {"variant": v["variant"], "pass": v["pass"], "skipped": v.get("skipped"),
                 "worst_min_iou": min((s["min_iou"] for s in v["sequences"]), default=None),
                 "worst_conf_delta": max((s["max_conf_delta"] for s in v["sequences"]), default=None),
                 "count_mismatches": sum(s["count_mismatches"] for s in v["sequences"])}
                for v in json.loads(out_json.read_text())["variants"]
            ]
        else:
            entry["error"] = "gate did not write its report (backend unavailable or crashed)"
        result["backends"][backend] = entry
    return result


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------

class ReplayDetector:
    """Hands detections computed in the tracking pass to evaluate_detection.evaluate(), so
    the detector runs once per frame, not once for tracking and again for scoring."""

    batch_size = 8

    def __init__(self, cache: list, template, start_frame: int = 0) -> None:
        self._cache, self._cursor = cache, start_frame
        self.backend = template.backend
        self.athlete_conf = template.athlete_conf
        self.ball_conf = template.ball_conf

    def process_sequence(self, frames):
        out = self._cache[self._cursor:self._cursor + len(frames)]
        if len(out) != len(frames):
            raise RuntimeError("replay cache is shorter than the frames evaluate() asked for")
        self._cursor += len(frames)
        return out


def load_models(args, cfg, need_pipeline: bool):
    import torch
    from gsr_pipeline.detect_athlete import AthleteBallDetector

    det_cfg = cfg["detector_b"]
    detector = AthleteBallDetector(
        onnx_path=args.onnx or det_cfg["onnx_path"],
        athlete_conf=args.athlete_conf if args.athlete_conf is not None else det_cfg["athlete_conf"],
        ball_conf=args.ball_conf if args.ball_conf is not None else det_cfg["ball_conf"],
        device=args.device, backend=args.backend,
        auto_build_engine=det_cfg.get("auto_build_engine", True),
    )
    models = {"detector": detector, "kp_detector": None, "jersey": None}
    if need_pipeline:
        from gsr_pipeline.calibrate import NBJWKeypointDetector

        calib = cfg["calibration"]
        calib_device = args.calib_device or ("cuda" if torch.cuda.is_available() else "cpu")
        models["kp_detector"] = NBJWKeypointDetector(
            checkpoint_kp=calib["checkpoint_kp"], checkpoint_l=calib["checkpoint_l"],
            device=calib_device, backend=args.calib_backend or calib.get("backend", "torch"),
            onnx_kp=calib.get("onnx_kp"), onnx_l=calib.get("onnx_l"),
            fp16=calib.get("fp16", False), fast_decode=calib.get("fast_decode", True),
            auto_build_engine=calib.get("auto_build_engine", True))
        if args.jersey:
            from gsr_pipeline.jersey import EasyOCRJerseyDetector

            models["jersey"] = EasyOCRJerseyDetector(
                **{**cfg.get("jersey", {}), "gpu": torch.cuda.is_available()})
    return models


# --------------------------------------------------------------------------------------
# the end-to-end pass: detect (+ track + calibrate + describe) over one sequence
# --------------------------------------------------------------------------------------

def run_pass(seq_dir: Path, models: dict, cfg: dict, args, do_pipeline: bool) -> dict:
    import cv2
    import pandas as pd
    from gsr_pipeline.athlete_roles import torso_descriptor
    import numpy as np
    from gsr_pipeline.calibrate import project_detections
    from gsr_pipeline.calibration_live import LiveCalibrator
    from gsr_pipeline.evaluate_detection import load_gt_boxes
    from gsr_pipeline.track import ByteTracker

    images, _ = load_gt_boxes(seq_dir / "Labels-GameState.json")
    if args.max_frames:
        images = images[:args.max_frames]
    img_dir = seq_dir / "img1" if (seq_dir / "img1").is_dir() else seq_dir / "images"

    detector, kp_detector, jersey = models["detector"], models["kp_detector"], models["jersey"]
    if do_pipeline:
        tracker = ByteTracker(**cfg["tracker"])
        calib = cfg["calibration"]
        calibrator = LiveCalibrator(
            kp_detector,
            image_width=calib.get("image_width", 1920), image_height=calib.get("image_height", 1080),
            use_prev_homography=calib.get("use_prev_homography", True),
            compute_cam_params=args.cam_params or calib.get("compute_cam_params", False),
            calib_every_k=args.calib_every_k or calib.get("calib_every_k", 1),
            mode=args.calib_mode or calib.get("mode", "inline"),
            max_h_age_frames=calib.get("max_h_age_frames", 50),
            min_inlier_ratio=calib.get("min_inlier_ratio", 0.5),
            max_zoom_change=calib.get("max_zoom_change", 0.15),
            min_trigger_interval_frames=calib.get("min_trigger_interval_frames", 3),
            motion=calib.get("motion"))
        h_sources = defaultdict(int)
    nbjw_ran = []               # per frame: did an NBJW forward run inline (splits the calibrate row)

    det_cache, records, metadata = [], [], {}
    lat = defaultdict(list)     # per-frame latency in ms, one list per stage
    next_singleton = SINGLETON_ID_START

    for pos, (image_id, file_name) in enumerate(images):
        t = time.perf_counter()
        frame = cv2.imread(str(img_dir / file_name))
        lat["frame_load"].append(_ms(t))
        if frame is None:
            raise FileNotFoundError(f"could not read {img_dir / file_name}")
        iid = _image_id(image_id, file_name)

        # detect() and kp_detector.detect() both end in a device-to-host copy, so these
        # wall-clock timings include the GPU work without an explicit synchronize.
        t = time.perf_counter()
        det = detector.detect(frame)
        lat["detect"].append(_ms(t))
        det_cache.append(det)
        if not do_pipeline:
            continue

        athletes = det[det["class_id"] == ATHLETE_CLASS].reset_index(drop=True)
        ball = det[det["class_id"] == BALL_CLASS].sort_values("bbox_conf", ascending=False).head(1).copy()

        t = time.perf_counter()
        tracked = tracker.update(athletes, frame.shape[:2])
        lat["track"].append(_ms(t))

        # One timer around the whole calibration step: with calib_every_k > 1 or async mode
        # the NBJW cost lands on some frames and not others, and that spread is the point.
        t = time.perf_counter()
        boxes = np.asarray(det["bbox_ltrb"].tolist(), np.float64).reshape(-1, 4) if len(det) else None
        cres = calibrator.update(pos, frame, boxes)
        H, params = cres.H, cres.cam_params
        lat["calibrate"].append(_ms(t))
        h_sources[cres.h_source] += 1
        nbjw_ran.append(bool(cres.nbjw_ran))

        jersey_by_row, jersey_ms = {}, 0.0
        if jersey is not None and len(tracked) and pos % args.jersey_every == 0:
            t = time.perf_counter()
            jdf = tracked.copy()
            jdf["role_detection"] = "player"      # process_frame only reads player/goalkeeper rows
            jdf = jersey.process_frame(frame, jdf)
            jersey_by_row = {i: (r["jersey_number_detection"], float(r["jersey_number_confidence"]))
                             for i, r in jdf.iterrows() if r["jersey_number_detection"]}
            jersey_ms = _ms(t)
        if jersey is not None:
            lat["jersey"].append(jersey_ms)

        t = time.perf_counter()
        if len(ball):
            ball["track_id"] = BALL_TRACK_ID
            combined = pd.concat([tracked, ball], ignore_index=True)
        else:
            combined = tracked
        combined = project_detections(combined, H)
        lat["project"].append(_ms(t))

        t = time.perf_counter()
        for i, row in combined.iterrows():
            is_ball = int(row["class_id"]) == BALL_CLASS
            if is_ball:
                track_id, desc, jn = BALL_TRACK_ID, None, (None, 0.0)
            else:
                track_id = row["track_id"]
                if pd.isna(track_id):
                    track_id, next_singleton = next_singleton, next_singleton + 1
                else:
                    track_id = int(track_id)
                desc = torso_descriptor(frame, row["bbox_ltrb"])
                jn = jersey_by_row.get(i, (None, 0.0))
            pitch = row["bbox_pitch"]
            records.append({
                "image_id": iid, "pos": pos, "kind": "ball" if is_ball else "athlete",
                "track_id": track_id, "ltwh": tuple(float(v) for v in row["bbox_ltwh"]),
                "pitch": pitch if isinstance(pitch, dict) else None, "desc": desc,
                "jn": jn[0], "jn_conf": jn[1],
            })
        metadata[iid] = {"parameters": _json_safe(params), "keypoints": {}, "lines": {}}
        lat["describe"].append(_ms(t))

        # Whole-pipeline latency for this frame: every stage above, back to back.
        frame_ms = {k: lat[k][-1] for k in PIPELINE_STAGES if lat.get(k)}
        lat["total"].append(sum(frame_ms.values()))
        lat["total_no_calib"].append(sum(v for k, v in frame_ms.items() if k not in CALIB_STAGES))

        if (pos + 1) % 100 == 0:
            print(f"    {seq_dir.name}: {pos + 1}/{len(images)} frames", flush=True)

    if do_pipeline:
        calibrator.close()
    return {"det_cache": det_cache, "records": records, "metadata": metadata, "n_frames": len(images),
            "h_sources": dict(h_sources) if do_pipeline else {}, "calib_nbjw_ran": nbjw_ran,
            "latency_ms": {k: [round(v, 2) for v in vals] for k, vals in lat.items()}}


# --------------------------------------------------------------------------------------
# roles, tracklet vote, output
# --------------------------------------------------------------------------------------

def assign_roles_and_teams(records: list, cfg: dict, fit_frames: int) -> dict:
    """Fit AthleteRoleTeamClassifier, predict, then vote per tracklet (replaces the
    detector-class vote in tracklet_agg on Path B, plan section 11). Mutates `records`."""
    import numpy as np
    from gsr_pipeline.athlete_roles import AthleteRoleTeamClassifier

    athletes = [r for r in records if r["kind"] == "athlete"]
    usable = [r for r in athletes if r["pitch"] and r["desc"] is not None]
    fit_rows = [r for r in usable if fit_frames <= 0 or r["pos"] < fit_frames]

    def xy(rows):
        return np.array([(r["pitch"]["x_bottom_middle"], r["pitch"]["y_bottom_middle"]) for r in rows])

    info = {"fit_rows": len(fit_rows), "usable_rows": len(usable), "fit_ok": False, "fit_error": None}
    try:
        clf = AthleteRoleTeamClassifier(**cfg["athlete_roles"]).fit(
            np.array([r["desc"] for r in fit_rows]), xy(fit_rows), [r["pos"] for r in fit_rows])
        for r, label in zip(usable, clf.predict(np.array([r["desc"] for r in usable]), xy(usable))):
            r["label"] = label
        info["fit_ok"] = True
    except (ValueError, RuntimeError) as exc:
        info["fit_error"] = str(exc)

    by_track = defaultdict(list)
    for r in athletes:
        by_track[r["track_id"]].append(r)
    for rows in by_track.values():
        labelled = [_split_label(r["label"]) for r in rows if "label" in r]
        role, team = "player", None
        if labelled:
            role = Counter(rl for rl, _ in labelled).most_common(1)[0][0]
            teams = Counter(tm for rl, tm in labelled if tm and rl == role)
            if teams and role in ("player", "goalkeeper"):
                team = TEAM_SIDE[teams.most_common(1)[0][0]]
        votes = defaultdict(float)
        if role in ("player", "goalkeeper"):
            for r in rows:
                if r["jn"]:
                    votes[r["jn"]] += r["jn_conf"]
        jersey = max(votes.items(), key=lambda kv: kv[1])[0] if votes else None
        for r in rows:
            r["role"], r["team"], r["jersey_number"] = role, team, jersey
    for r in records:
        if r["kind"] == "ball":
            r["role"], r["team"], r["jersey_number"] = "ball", None, None
    info["role_counts"] = dict(Counter(r["role"] for r in athletes))
    return info


def write_predictions(records: list, metadata: dict, path: Path) -> None:
    import pandas as pd
    from gsr_pipeline.output import save_results_json

    df = pd.DataFrame([{
        "image_id": r["image_id"], "track_id": r["track_id"], "bbox_ltwh": r["ltwh"],
        "bbox_pitch": r["pitch"], "team": r["team"], "role": r["role"],
        "jersey_number": r["jersey_number"],
    } for r in records])
    meta_df = pd.DataFrame.from_dict(metadata, orient="index")
    meta_df.index.name = "image_id"
    save_results_json(df, meta_df, path)


# --------------------------------------------------------------------------------------
# GS-HOTA (evaluate.py, unchanged; its printed table is captured and its MEAN row parsed)
# --------------------------------------------------------------------------------------

_MEAN_ROW = re.compile(r"MEAN\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%\s*\|\s*([\d.]+)\s*%")


def score_gshota(seq_dir: Path, pred_path: Path, use_jersey: bool) -> dict:
    from gsr_pipeline.evaluate import evaluate_gs_hota, load_gt, load_preds

    gt, pred = load_gt(seq_dir / "Labels-GameState.json"), load_preds(pred_path)
    if not use_jersey:
        gt["jersey"] = None     # no OCR on this path: IdSim would otherwise zero every player
    out = {"jersey": "used" if use_jersey else "ignored"}
    for name, drop_ball in HOTA_VARIANTS:
        g = gt[gt["role"] != "ball"] if drop_ball else gt
        p = pred[pred["role"] != "ball"] if (drop_ball and not pred.empty) else pred
        if g.empty or p.empty:
            out[name] = {"det_a": 0.0, "ass_a": 0.0, "gs_hota": 0.0, "note": "empty gt or predictions"}
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            hota = evaluate_gs_hota(g.copy(), p.copy())
        m = _MEAN_ROW.search(buf.getvalue())
        out[name] = {"det_a": float(m.group(1)) / 100 if m else None,
                     "ass_a": float(m.group(2)) / 100 if m else None,
                     "gs_hota": float(hota)}
    return out


# --------------------------------------------------------------------------------------
# role-classifier eval on GT (evaluate_athlete_roles protocol)
# --------------------------------------------------------------------------------------

def score_roles(seq_dir: Path, cfg: dict, step: int, fit_frames: int) -> dict:
    import numpy as np
    from gsr_pipeline.athlete_roles import AthleteRoleTeamClassifier
    from gsr_pipeline.evaluate_athlete_roles import load_rows, score

    rows = load_rows(seq_dir, step)
    fit_rows = [r for r in rows if r["frame"] < fit_frames]
    held = [r for r in rows if r["frame"] >= fit_frames]
    if not fit_rows or not held:
        return {"error": f"fit_frames={fit_frames} leaves an empty fit or held-out set"}
    try:
        clf = AthleteRoleTeamClassifier(**cfg["athlete_roles"]).fit(
            np.array([r["descriptor"] for r in fit_rows]), np.array([r["pitch"] for r in fit_rows]),
            [r["frame"] for r in fit_rows])
    except (ValueError, RuntimeError) as exc:
        return {"error": f"fit failed: {exc}"}
    out = {}
    for name, subset in (("fit_window", fit_rows), ("held_out", held)):
        labels = clf.predict(np.array([r["descriptor"] for r in subset]), np.array([r["pitch"] for r in subset]))
        out[name] = _json_safe(score(subset, labels))
    return out


# --------------------------------------------------------------------------------------
# aggregation across sequences
# --------------------------------------------------------------------------------------

def aggregate_detection(reports: list) -> dict | None:
    reports = [r for r in reports if r]
    if not reports:
        return None
    agg = {"sequences": len(reports), "per_class": {}}
    for cls in reports[0]["per_class"]:
        tp, fp, fn = (sum(r["per_class"][cls][k] for r in reports) for k in ("tp", "fp", "fn"))
        agg["per_class"][cls] = _prf(tp, fp, fn)
    labelled = sum(r["ball"]["frames_with_ball_labelled"] for r in reports)
    found = sum(round((r["ball"]["frame_recall"] or 0) * r["ball"]["frames_with_ball_labelled"]) for r in reports)
    misses = [r["ball"]["longest_consecutive_miss_seconds"] for r in reports]
    sizes = {}
    for bucket in reports[0]["ball"]["recall_by_gt_box_size"]:
        gts = sum(r["ball"]["recall_by_gt_box_size"][bucket]["gt"] for r in reports)
        hits = sum(round((r["ball"]["recall_by_gt_box_size"][bucket]["recall"] or 0)
                         * r["ball"]["recall_by_gt_box_size"][bucket]["gt"]) for r in reports)
        sizes[bucket] = {"gt": gts, "recall": hits / gts if gts else None}
    agg["ball"] = {"frame_recall": found / labelled if labelled else None,
                   "frames_labelled": labelled,
                   "longest_miss_s_max": max(misses), "longest_miss_s_mean": sum(misses) / len(misses),
                   "recall_by_gt_box_size": sizes}
    return agg


def aggregate_roles(results: list) -> dict | None:
    held = [r["held_out"] for r in results if r and "held_out" in r]
    if not held:
        return None
    n = sum(h["n"] for h in held)
    agg = {"sequences": len(held), "n": n,
           "role_accuracy": sum((h["role_accuracy"] or 0) * h["n"] for h in held) / n if n else None}
    pairs = sum(h["team_pairs"] for h in held)
    agg["team_accuracy"] = sum((h["team_accuracy"] or 0) * h["team_pairs"] for h in held) / pairs if pairs else None
    agg["team_pairs"] = pairs
    per_role = {}
    for h in held:
        for role, m in h["per_role"].items():
            e = per_role.setdefault(role, {"n": 0, "hits": 0.0})
            e["n"] += m["n"]
            e["hits"] += m["recall"] * m["n"]
    agg["per_role_recall"] = {r: {"n": e["n"], "recall": e["hits"] / e["n"]} for r, e in per_role.items()}
    return agg


def aggregate_gshota(results: list) -> dict | None:
    rows = [r["gshota"]["hota"] for r in results if r and "gshota" in r and "hota" in r["gshota"]]
    if not rows:
        return None
    out = {"sequences": len(rows), "jersey": rows[0]["jersey"]}
    for name, _ in HOTA_VARIANTS:
        vals = [r[name] for r in rows if r[name]["det_a"] is not None]
        out[name] = {k: sum(v[k] for v in vals) / len(vals) if vals else None
                     for k in ("det_a", "ass_a", "gs_hota")}
    return out


# --------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------

def _summarise(values) -> dict | None:
    import numpy as np

    if not len(values):
        return None
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def aggregate_latency(results: list, warmup: int) -> dict | None:
    """Pool per-frame latencies across sequences, dropping each sequence's first `warmup`
    frames (CUDA/ORT autotuning and allocator warm-up land there, not in steady state)."""
    pools = defaultdict(list)
    for r in results:
        lat = r.get("latency_ms") or {}
        for stage, values in lat.items():
            pools[stage].extend(values[warmup:])
        ran = r.get("calib_nbjw_ran") or []
        if len(ran) == len(lat.get("calibrate", [])):       # absent in results cached before this split
            for flag, ms in list(zip(ran, lat["calibrate"]))[warmup:]:
                pools["calibrate_nbjw" if flag else "calibrate_no_nbjw"].append(ms)
    out = {stage: _summarise(v) for stage, v in pools.items() if v}
    return out or None


def profile_components(models: dict, seq_dir: Path, n_frames: int, warmup: int = 8,
                       cam_params: bool = False) -> dict:
    """Sub-stage latency of the detector and of calibration, which the main pass can only time
    as black boxes. Replays the same calls as AthleteBallDetector.detect,
    NBJWKeypointDetector.detect and NBJWHomography.compute on the first frames of one sequence,
    with a CUDA synchronize between steps so GPU time is attributed to the step that spent it.

    A diagnostic breakdown, kept apart from the scored run: it re-implements those bodies, so
    it can drift from them. The sum of its steps should land close to the main pass's detect and
    calibrate number on fresh (NBJW) frames; if it does not, trust the main pass.
    """
    import cv2
    import torch
    from gsr_pipeline.calibrate import NBJWHomography
    from gsr_pipeline.evaluate_detection import load_gt_boxes
    from gsr_pipeline.runtime.postprocess import postprocess, preprocess

    images, _ = load_gt_boxes(seq_dir / "Labels-GameState.json")
    img_dir = seq_dir / "img1" if (seq_dir / "img1").is_dir() else seq_dir / "images"
    frames = [cv2.imread(str(img_dir / fn)) for _, fn in images[:n_frames]]

    det, kp = models["detector"], models["kp_detector"]
    cuda = torch.cuda.is_available()

    def tick() -> float:
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    if kp is not None:
        from gsr_pipeline.calibrate import KP_THRESHOLD, LINE_THRESHOLD
        if kp.fast_decode:      # time whichever decode the pipeline actually runs
            from gsr_pipeline.nbjw_fast_decode import (
                complete_keypoints, coords_to_dict, peaks_kp as peaks_k, peaks_lines as peaks_l)
        else:
            from gsr_pipeline.nbjw_calib.utils.utils_heatmap import (
                complete_keypoints, coords_to_dict, get_keypoints_from_heatmap_batch_maxpool as peaks_k,
                get_keypoints_from_heatmap_batch_maxpool_l as peaks_l)
        homography = NBJWHomography(image_width=1920, image_height=1080)

    samples = defaultdict(list)     # stage -> seconds per frame
    for frame in frames:
        t0 = tick()
        x, params = preprocess(frame, det.artifact.height, det.artifact.width, device=det.device, out=det._input)
        t1 = tick()
        raw = det.runner.infer(x)
        t2 = tick()
        boxes, scores, class_ids = postprocess(
            raw.to(torch.float32), params, det.conf_by_class,
            iou=det.artifact.nms_iou, max_det=det.artifact.nms_max_det)
        boxes.detach().cpu(), scores.detach().cpu(), class_ids.detach().cpu()
        t3 = tick()
        samples["det_preprocess"].append(t1 - t0)
        samples["det_infer"].append(t2 - t1)
        samples["det_postprocess"].append(t3 - t2)
        if kp is None:
            continue

        t0 = tick()
        tensor = kp.preprocess(frame)
        t1 = tick()
        hm_kp = kp.net_kp.infer(tensor)
        t2 = tick()
        hm_l = kp.net_l.infer(tensor)
        t3 = tick()
        pk, pl = peaks_k(hm_kp[:, :-1, :, :]), peaks_l(hm_l[:, :-1, :, :])
        t3a = tick()
        kp_raw, l_raw = coords_to_dict(pk, threshold=KP_THRESHOLD), coords_to_dict(pl, threshold=LINE_THRESHOLD)
        t3b = tick()
        final = complete_keypoints(kp_raw, l_raw, w=tensor.size(-1), h=tensor.size(-2), normalize=True)
        keypoints = final[0] if final else {}
        t4 = tick()
        cam = homography.cam
        cam.update(keypoints)
        t5 = tick()
        h = cam.get_homography_from_ground_plane(use_ransac=50, inverse=True)
        t6 = tick()
        steps = [("calib_prep", t1 - t0), ("calib_kp_net", t2 - t1), ("calib_line_net", t3 - t2),
                 ("calib_decode_peaks", t3a - t3), ("calib_decode_dict", t3b - t3a),
                 ("calib_decode_complete", t4 - t3b), ("calib_cam_update", t5 - t4), ("calib_ransac", t6 - t5)]
        if cam_params:          # off in the pipeline unless compute_cam_params; only then is it on the path
            if h is not None:
                cam.heuristic_voting()
            steps.append(("calib_voting", tick() - t6))
        t7 = tick()
        kp.decode(hm_kp, hm_l, fast=not kp.fast_decode)
        steps.append(("calib_decode_other", tick() - t7))
        for name, dt in steps:
            samples[name].append(dt)

    return {"sequence": seq_dir.name, "frames": len(frames), "warmup_dropped": warmup,
            "fast_decode": bool(kp is not None and kp.fast_decode),
            "stages": {k: _summarise([v * 1000 for v in vals[warmup:]]) for k, vals in samples.items()}}


def build_markdown(summary: dict) -> str:
    L = ["# Path B evaluation", ""]
    env = summary["environment"]
    L += [f"- run: {env['utc']}   gpu: {env.get('gpu', 'none')}   torch {env.get('torch')}   "
          f"ort {env.get('onnxruntime')}   tensorrt {env.get('tensorrt')}",
          f"- detector backend: **{summary.get('detector_backend')}**   sequences: {summary['sequences_run']}"
          f"   max_frames: {summary['args']['max_frames']}", ""]

    if summary.get("parity"):
        L += ["## Parity gate (runner vs Ultralytics .pt, box for box)", ""]
        p = summary["parity"]
        if "skipped" in p:
            L.append(p["skipped"])
        for backend, e in p.get("backends", {}).items():
            L.append(f"- **{backend}**: {'PASS' if e['pass'] else 'FAIL'} (rc={e['returncode']})"
                     + (f" -- {e['error']}" if e.get("error") else ""))
            for v in e.get("variants", []):
                state = "skipped" if v["pass"] is None else ("PASS" if v["pass"] else "FAIL")
                L.append(f"  - {v['variant']}: {state}, worst min-IoU {v['worst_min_iou']}, "
                         f"worst conf delta {v['worst_conf_delta']}, count mismatches {v['count_mismatches']}")
        L.append("")

    det = summary.get("detection")
    if det:
        L += [f"## Detection ({det['sequences']} sequences, counts pooled)", "",
              "| class | GT | TP | FP | FN | recall | precision | F1 |", "|---|---|---|---|---|---|---|---|"]
        for cls, m in det["per_class"].items():
            L.append(f"| {cls} | {m['tp'] + m['fn']} | {m['tp']} | {m['fp']} | {m['fn']} | "
                     f"{_pct(m['recall'])} | {_pct(m['precision'])} | {_pct(m['f1'])} |")
        b = det["ball"]
        L += ["", f"Ball frame-recall {_pct(b['frame_recall'])} over {b['frames_labelled']} labelled frames; "
                  f"longest miss {b['longest_miss_s_max']:.2f}s (max), {b['longest_miss_s_mean']:.2f}s (mean).",
              "Ball recall by GT box size: " + ", ".join(
                  f"{k} {_pct(v['recall'])} (n={v['gt']})" for k, v in b["recall_by_gt_box_size"].items()), ""]
    ref = summary.get("reference")
    if ref:
        L += ["### Accuracy gate vs Ultralytics `.pt` (plan 3.5: within 1 point)", "",
              "| metric | .pt | runner | delta (pts) | |", "|---|---|---|---|---|"]
        for row in ref["rows"]:
            L.append(f"| {row['metric']} | {_pct(row['reference'])} | {_pct(row['runner'])} | "
                     f"{row['delta_pts']:+.2f} | {'PASS' if row['pass'] else 'FAIL'} |")
        L += ["", f"Gate: **{'PASS' if ref['pass'] else 'FAIL'}**", ""]

    roles = summary.get("roles")
    if roles:
        L += [f"## Role / team classifier on GT ({roles['sequences']} sequences, held-out frames)", "",
              f"- role accuracy {_pct(roles['role_accuracy'])} over {roles['n']} detections",
              f"- team accuracy {_pct(roles['team_accuracy'])} over {roles['team_pairs']} pairs "
              "(permutation-invariant: team_a/team_b are arbitrary names)",
              "- per-role recall: " + ", ".join(f"{k} {_pct(v['recall'])} (n={v['n']})"
                                               for k, v in roles["per_role_recall"].items()), ""]

    hota = summary.get("gshota")
    if hota:
        L += [f"## GS-HOTA, end-to-end Path B ({hota['sequences']} sequences, mean of per-sequence scores)", "",
              f"Jersey in IdSim: **{hota['jersey']}**.", "",
              "| variant | DetA | AssA | GS-HOTA |", "|---|---|---|---|"]
        for name, _ in HOTA_VARIANTS:
            v = hota[name]
            L.append(f"| {name} | {_pct(v['det_a'])} | {_pct(v['ass_a'])} | {_pct(v['gs_hota'])} |")
        L += ["", "`persons` excludes the ball from GT and predictions; `with_ball` keeps it.", ""]

    lat = summary.get("latency")
    if lat:
        total = lat.get("total")
        L += [f"## Latency per frame, ms (tag: `{env.get('gpu', 'cpu')}`)", "",
              f"Wall time of each stage run back to back on one thread, pooled over all frames "
              f"(first {summary['args']['latency_warmup']} frames of each sequence dropped as warm-up). "
              "GPU stages end in a device-to-host copy, so their GPU time is included. This is the "
              "sequential cost, not a pipelined production figure: plan 1.4's targets assume threaded "
              "stages, NVDEC and an RTX 30-series, so they are not comparable.", "",
              "| stage | n | mean | p50 | p95 | p99 | max | share of total |", "|---|---|---|---|---|---|---|---|"]
        for key, label in LATENCY_ROWS:
            s = lat.get(key)
            if not s:
                continue
            share = (f"{s['mean'] / total['mean'] * 100:.0f}%"
                     if total and key not in ("total", "total_no_calib") else "")
            bold = "**" if key.startswith("total") else ""
            L.append(f"| {bold}{label}{bold} | {s['n']} | {s['mean']:.1f} | {s['p50']:.1f} | "
                     f"{s['p95']:.1f} | {s['p99']:.1f} | {s['max']:.1f} | {share} |")
        if total:
            L += ["", f"Sequential throughput: **{1000 / total['mean']:.2f} fps** (mean), "
                      f"{1000 / total['p95']:.2f} fps at p95."]
            if lat.get("total_no_calib"):
                L.append(f"Without calibration and projection: {1000 / lat['total_no_calib']['mean']:.1f} fps (mean).")
        hs = Counter(summary.get("h_sources") or {})
        if hs:
            cs = summary.get("calib") or {}
            L.append(f"Calibration: backend `{summary.get('calib_backend')}`, "
                     f"calib_every_k={cs.get('calib_every_k')}, mode={cs.get('mode')}, "
                     f"min_trigger_interval_frames={cs.get('min_trigger_interval_frames')}, "
                     f"fast_decode={cs.get('fast_decode')}; homography source per frame: "
                     + ", ".join(f"{k} {v}" for k, v in sorted(hs.items())) + ".")
        cost = summary.get("roles_cost_ms")
        if cost:
            L.append(f"Role/team fit + predict + vote: {cost['per_sequence']:.0f} ms per sequence "
                     f"(~{cost['per_frame']:.2f} ms/frame amortised), not in the per-frame rows above.")
        L.append("")

    prof = summary.get("profile")
    if prof and "stages" in prof:
        L += [f"### Sub-stage breakdown ({prof['sequence']}, {prof['frames']} frames, "
              f"first {prof['warmup_dropped']} dropped; GPU steps synchronised)", "",
              "| step | mean | p50 | p95 | max |", "|---|---|---|---|---|"]
        for key, label in PROFILE_ROWS:
            s = prof["stages"].get(key)
            if s:
                L.append(f"| {label} | {s['mean']:.1f} | {s['p50']:.1f} | {s['p95']:.1f} | {s['max']:.1f} |")
        det_sum = sum(prof["stages"][k]["mean"] for k in ("det_preprocess", "det_infer", "det_postprocess")
                      if prof["stages"].get(k))
        calib_sum = sum(prof["stages"][k]["mean"] for k in PROFILE_ON_PATH if prof["stages"].get(k))
        L += ["", f"Detector steps sum to {det_sum:.1f} ms"
                  + (f"; calibration steps on an NBJW frame sum to {calib_sum:.1f} ms "
                     f"(decode: {'nbjw_fast_decode' if prof.get('fast_decode') else 'vendored reference'})."
                     if calib_sum else "."),
              "Camera-param voting appears only when compute_cam_params is on; the last row times the "
              "decode implementation the pipeline is not using, on the same heatmaps, and is not summed.",
              "Compare with the detector and calibration rows above; a large gap means this breakdown "
              "has drifted from the real code and the main-pass rows are the ones to trust.", ""]
    elif prof:
        L += ["### Sub-stage breakdown", "", "Failed: " + "; ".join((prof.get("error") or "").splitlines()[-1:]), ""]

    if summary.get("failed"):
        L += ["## Failed sequences", ""] + [f"- {k}: {v.splitlines()[-1]}" for k, v in summary["failed"].items()] + [""]
    L += ["## Scope notes", ""] + [f"- {n}" for n in summary["notes"]]
    return "\n".join(L) + "\n"


NOTES = [
    "GS-HOTA here is the repo's evaluate.py unchanged. Detections without a pitch projection score as misses.",
    "No tracklet refinement (needs ReID). Calibration is LiveCalibrator: NBJW every calib_every_k frames "
    "or on a trigger, homography propagated by camera motion in between (plan 7; K=1 inline is per-frame NBJW).",
    "Ball: single highest-confidence detection per frame, one constant track id (plan Phase 6 BallTracker not built).",
    "Unassociated detections become one-frame singleton tracklets rather than being dropped.",
    "Roles: AthleteRoleTeamClassifier fitted per sequence on the first --fit_frames frames (0 = whole sequence), "
    "then a per-tracklet majority vote. team_a (lower mean pitch x) is labelled 'left'.",
    "athlete_roles tunables were fitted on SNGS-021 only; every other sequence is a held-out test of them.",
]


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Path B end-to-end + component evaluation runner",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_root", default="data/datasets/SoccerNetGS/gamestate-2024/valid")
    p.add_argument("--output_dir", default="data/outputs/path_b_eval")
    p.add_argument("--sequences", nargs="*", default=None, help="sequence folder names (default: all)")
    p.add_argument("--limit", type=int, default=None, help="only the first N sequences")
    p.add_argument("--max_frames", type=int, default=None, help="cap frames per sequence (smoke tests)")
    p.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    p.add_argument("--config", default="gsr_pipeline/configs/default.yaml")
    p.add_argument("--onnx", default=None, help="default: detector_b.onnx_path from the config")
    p.add_argument("--weights", default="runs/yolo26s_athlete_ball/best.pt")
    p.add_argument("--backend", default=None, choices=["tensorrt", "ort_gpu", "ort_cpu"],
                   help="pin a detector tier; default walks the fallback chain")
    p.add_argument("--device", default=None, help="detector device; default cuda:0 if available else cpu")
    p.add_argument("--calib_device", default=None, help="NBJW device; default cuda if available else cpu")
    p.add_argument("--calib_backend", default=None, choices=["torch", "auto", "tensorrt", "ort_gpu", "ort_cpu"],
                   help="NBJW forward backend; default calibration.backend from the config")
    p.add_argument("--calib_every_k", type=int, default=None,
                   help="run NBJW every K frames, propagate H by camera motion between (default: config)")
    p.add_argument("--calib_mode", default=None, choices=["inline", "async"], help="default: config")
    p.add_argument("--cam_params", action="store_true",
                   help="also run NBJW heuristic_voting (camera intrinsics); the old per-frame cost")
    p.add_argument("--athlete_conf", type=float, default=None)
    p.add_argument("--ball_conf", type=float, default=None)
    p.add_argument("--fit_frames", type=int, default=250,
                   help="gshota: role classifier fit window in frames (0 = whole sequence)")
    p.add_argument("--jersey", action="store_true", help="gshota: run EasyOCR (slow) and keep jersey in IdSim")
    p.add_argument("--jersey_every", type=int, default=5, help="OCR every Nth frame when --jersey")
    p.add_argument("--det_start_frame", type=int, default=0, help="detection stage: first frame to score")
    p.add_argument("--reference", action="store_true",
                   help="detection stage: also score the Ultralytics .pt and apply the plan 3.5 gate")
    p.add_argument("--role_step", type=int, default=5, help="roles stage: use every Nth frame")
    p.add_argument("--role_fit_frames", type=int, default=250, help="roles stage: fit window (evaluate_athlete_roles)")
    p.add_argument("--parity_backends", nargs="*", default=["ort_gpu"], choices=["tensorrt", "ort_gpu", "ort_cpu"])
    p.add_argument("--parity_frames", type=int, default=30)
    p.add_argument("--build_trt", action="store_true", help="run scripts/build_trt_engine.py before the tensorrt parity")
    p.add_argument("--latency_warmup", type=int, default=3,
                   help="latency: frames dropped from the start of each sequence before computing percentiles")
    p.add_argument("--profile_frames", type=int, default=40,
                   help="latency: frames for the detector/calibration sub-stage breakdown (0 = skip; "
                        "the first 8 are warm-up)")
    p.add_argument("--force", action="store_true", help="ignore cached per-sequence results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root, out_dir = Path(args.data_root).resolve(), Path(args.output_dir).resolve()
    os.chdir(ROOT)      # config / weights / checkpoints paths are relative to cv_service/
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import yaml

    args.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seq_dirs = _sequence_dirs(data_root, args.sequences, args.limit)
    if not seq_dirs:
        raise SystemExit(f"no sequences with Labels-GameState.json under {data_root}")
    stages = set(args.stages)
    print(f"{len(seq_dirs)} sequences, stages: {sorted(stages)}, device: {args.device}", flush=True)

    calib = cfg["calibration"]
    summary = {"environment": environment(), "args": vars(args), "notes": NOTES,
               "sequences_run": len(seq_dirs), "failed": {},
               # What actually ran, after CLI overrides -- not "config".
               "calib": {"calib_every_k": args.calib_every_k or calib.get("calib_every_k", 1),
                         "mode": args.calib_mode or calib.get("mode", "inline"),
                         "compute_cam_params": bool(args.cam_params or calib.get("compute_cam_params", False)),
                         "min_trigger_interval_frames": calib.get("min_trigger_interval_frames", 3),
                         "fast_decode": calib.get("fast_decode", True)}}

    if "parity" in stages:
        summary["parity"] = run_parity(args, seq_dirs, out_dir)

    models, reference = None, None
    if stages & {"detection", "gshota"}:
        models = load_models(args, cfg, need_pipeline="gshota" in stages)
        summary["detector_backend"] = models["detector"].backend
        if models["kp_detector"] is not None:
            summary["calib_backend"] = models["kp_detector"].backend_name
        print(f"detector backend: {models['detector'].backend}", flush=True)
        if args.profile_frames > 0:
            print("profiling detector / calibration sub-stages ...", flush=True)
            try:
                summary["profile"] = profile_components(models, seq_dirs[0], args.profile_frames,
                                                            cam_params=summary["calib"]["compute_cam_params"])
            except Exception:
                summary["profile"] = {"error": traceback.format_exc()}
                print(f"  profile failed: {summary['profile']['error'].splitlines()[-1]}", flush=True)
        if args.reference and "detection" in stages:
            from gsr_pipeline.evaluate_detection import UltralyticsReferenceDetector

            art = models["detector"].artifact
            reference = UltralyticsReferenceDetector(
                weights=args.weights, imgsz=(art.height, art.width),
                athlete_conf=models["detector"].athlete_conf, ball_conf=models["detector"].ball_conf,
                device=args.device)

    results = {}
    for k, seq_dir in enumerate(seq_dirs, 1):
        name = seq_dir.name
        res_path = out_dir / name / "result.json"
        result = json.loads(res_path.read_text()) if res_path.exists() and not args.force else {}
        print(f"\n[{k}/{len(seq_dirs)}] {name}", flush=True)
        try:
            from gsr_pipeline.evaluate_detection import evaluate as evaluate_detection

            todo = [s for s in ("detection", "gshota") if s in stages and s not in result]
            if todo:
                p = run_pass(seq_dir, models, cfg, args, do_pipeline="gshota" in todo)
                result["latency_ms"] = p["latency_ms"]
                result["h_sources"] = p["h_sources"]
                result["calib_nbjw_ran"] = p["calib_nbjw_ran"]
                result["frames"] = p["n_frames"]
                if "detection" in todo:
                    result["detection"] = evaluate_detection(
                        seq_dir, ReplayDetector(p["det_cache"], models["detector"], args.det_start_frame),
                        person_iou=0.5, ball_match_px=15.0, max_frames=args.max_frames, fps=25.0,
                        two_class=True, start_frame=args.det_start_frame)
                if "gshota" in todo:
                    t = time.perf_counter()
                    info = assign_roles_and_teams(p["records"], cfg, args.fit_frames)
                    info["roles_ms"] = round(_ms(t), 1)
                    pred_path = out_dir / name / "predictions.json"
                    write_predictions(p["records"], p["metadata"], pred_path)
                    info["hota"] = score_gshota(seq_dir, pred_path, use_jersey=args.jersey)
                    result["gshota"] = info
            if reference is not None and "reference" not in result:
                result["reference"] = evaluate_detection(
                    seq_dir, reference, person_iou=0.5, ball_match_px=15.0, max_frames=args.max_frames,
                    fps=25.0, two_class=True, start_frame=args.det_start_frame)
            if "roles" in stages and "roles" not in result:
                result["roles"] = score_roles(seq_dir, cfg, args.role_step, args.role_fit_frames)
            result.pop("error", None)
        except Exception:
            result["error"] = traceback.format_exc()
            summary["failed"][name] = result["error"]
            print(f"  FAILED: {result['error'].splitlines()[-1]}", flush=True)
        res_path.parent.mkdir(parents=True, exist_ok=True)
        res_path.write_text(json.dumps(_json_safe(result), indent=2))
        results[name] = result

        line = [name]
        if "detection" in result:
            d = result["detection"]["per_class"]
            line.append(f"athlete R/P {_pct(d['athlete']['recall'])}/{_pct(d['athlete']['precision'])}"
                        f"  ball R/P {_pct(d['ball']['recall'])}/{_pct(d['ball']['precision'])}")
        if "gshota" in result and "hota" in result["gshota"]:
            line.append(f"GS-HOTA persons {_pct(result['gshota']['hota']['persons']['gs_hota'])}")
        if "roles" in result and "held_out" in result["roles"]:
            line.append(f"role acc {_pct(result['roles']['held_out']['role_accuracy'])}")
        total_ms = (result.get("latency_ms") or {}).get("total", [])[args.latency_warmup:]
        if total_ms:
            line.append(f"{sum(total_ms) / len(total_ms):.0f} ms/frame")
        print("  " + "  |  ".join(line), flush=True)

    ok = [r for r in results.values() if "error" not in r]
    summary["detection"] = aggregate_detection([r.get("detection") for r in ok])
    summary["roles"] = aggregate_roles([r.get("roles") for r in ok])
    summary["gshota"] = aggregate_gshota(ok)
    summary["latency"] = aggregate_latency(ok, args.latency_warmup)
    summary["h_sources"] = dict(sum((Counter(r.get("h_sources") or {}) for r in ok), Counter()))
    roles_ms = [(r["gshota"]["roles_ms"], r["frames"]) for r in ok if "roles_ms" in r.get("gshota", {})]
    if roles_ms:
        summary["roles_cost_ms"] = {
            "per_sequence": sum(ms for ms, _ in roles_ms) / len(roles_ms),
            "per_frame": sum(ms for ms, _ in roles_ms) / max(sum(n for _, n in roles_ms), 1)}
    refs = aggregate_detection([r.get("reference") for r in ok])
    if refs and summary["detection"]:
        rows = []
        for metric, get in (("athlete recall", lambda a: a["per_class"]["athlete"]["recall"]),
                            ("athlete precision", lambda a: a["per_class"]["athlete"]["precision"]),
                            ("ball recall", lambda a: a["per_class"]["ball"]["recall"]),
                            ("ball precision", lambda a: a["per_class"]["ball"]["precision"])):
            r_, o_ = get(refs), get(summary["detection"])
            delta = (o_ - r_) * 100
            rows.append({"metric": metric, "reference": r_, "runner": o_, "delta_pts": delta,
                         "pass": abs(delta) <= 1.0})
        summary["reference"] = {"rows": rows, "pass": all(r["pass"] for r in rows)}

    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    md = build_markdown(summary)
    (out_dir / "summary.md").write_text(md)
    print("\n" + md)
    print(f"wrote {out_dir / 'summary.json'} and summary.md; per-sequence results and predictions.json under {out_dir}")


if __name__ == "__main__":
    main()
