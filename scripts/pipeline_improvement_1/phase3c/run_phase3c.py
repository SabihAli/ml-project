"""
run_phase3c.py -- Phase 3c end to end (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md), resumable: rerun the same command
after a pause (kill) and it continues. Each stage writes <state>/<stage>.done when its command exits 0; a done stage
is skipped. Training stages continue from their last checkpoint (--resume) when one exists.

Stages (label_balls .. pseudo_seqs in order, then the two branches student_data -> det_train -> det_eval
and main_cache -> c5c_data -> c5c_train -> c5c_eval in parallel threads):
  label_balls       pseudo_label_matches.py up to --budget_gb (exits at once when the cap is already reached)
  athlete_train     athlete_teacher.py train (YOLO26s, 1920)
  athlete_choose    athlete_teacher.py choose (confidence on select; test once)
  athlete_label     athlete_teacher.py label (every pseudo window)
  pseudo_seqs       make_pseudo_seqs.py (GSR-format sequences, ball >= 0.4)
  student_data      build_student_data.py --src gsr,pseudo
  main_cache        c5c's main-detector cache over GSR + pseudo (live detector, as the runtime centres windows)
  c5c_data          c5c's crop dataset over GSR + pseudo (pseudo positives thinned, disk)
  det_train         student_detector.py train (5 ch, past frames)
  det_eval          student + baseline (live detector) on select / test
  c5c_train         c5c student, warm start from c5c, same recipe
  c5c_eval          eval_ball_temporal.py all, the c5c student and c5c on the same cache and crops

    python scripts/pipeline_improvement_1/phase3c/run_phase3c.py
    python scripts/pipeline_improvement_1/phase3c/run_phase3c.py --only det_train
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable
S3B = "scripts/pipeline_improvement_1/phase3b"
S3C = "scripts/pipeline_improvement_1/phase3c"
BT = "scripts/ball_temporal"
OUT = Path("data/outputs/pipeline_improvement_1/phase3c")
SRC = "data/gsr,data/pseudo_gsr"                # images only (main-detector cache)
SRC_T = "data/gsr_teacher,data/pseudo_gsr"      # training labels: teachers only (GSR train/pool relabelled)
CACHE = OUT / "main_cache"
C5C_DATA = OUT / "c5c_data"
C5C_OUT = Path("runs/student/b1_c5c_pseudo")
SERIAL = ["label_balls", "athlete_train", "athlete_choose", "athlete_label", "pseudo_seqs"]
BRANCHES = [["relabel_gsr", "student_data", "det_train", "det_eval"],                     # detector student
            ["main_cache", "c5c_data", "c5c_train", "c5c_eval"]]          # c5c student (in parallel)


def stages(a):
    det_last = Path("runs/student/d1_s1280_c5/weights/last.pt")
    ath_last = Path("runs/teacher/a1_s1920_ft/weights/last.pt")
    return [
        ("label_balls", [PY, "-u", f"{S3B}/pseudo_label_matches.py", "--n_games", "171", "--budget_gb", str(a.budget_gb)],
         {"HF_HUB_DOWNLOAD_TIMEOUT": "60", "HF_HUB_ETAG_TIMEOUT": "60"}),
        ("athlete_train", [PY, "-u", f"{S3C}/athlete_teacher.py", "train", "--batch", "4", "--epochs", "12",
                           "--patience", "4", "--close_mosaic", "3"] + (["--resume"] if ath_last.exists() else []), {}),
        ("athlete_choose", [PY, "-u", f"{S3C}/athlete_teacher.py", "choose"], {}),
        ("athlete_label", [PY, "-u", f"{S3C}/athlete_teacher.py", "label"], {}),
        ("pseudo_seqs", [PY, "-u", f"{S3C}/make_pseudo_seqs.py"], {}),
        ("relabel_gsr", [PY, "-u", f"{S3C}/relabel_gsr.py", "all"], {}),
        ("student_data", [PY, "-u", f"{S3C}/build_student_data.py", "--src", SRC_T], {}),
        ("main_cache", [PY, "-u", f"{BT}/cache_main_detections.py", "run", "--src", SRC, "--out", str(CACHE),
                        "--weights", "runs/yolo26s_athlete_ball/best.pt", "--device", "0", "--half", "--batch", "32",
                        "--decode_threads", "4"], {}),
        ("c5c_data", [PY, "-u", f"{BT}/build_ball_temporal_dataset.py", "build", "--src", SRC_T, "--cache", str(CACHE),
                      "--out", str(C5C_DATA), "--workers", "8", "--pseudo_easy_step", str(a.c5c_easy_step),
                      "--pseudo_hard_step", str(a.c5c_hard_step)], {}),
        ("det_train", [PY, "-u", f"{S3C}/student_detector.py", "train", "--batch", str(a.det_batch), "--epochs", str(a.det_epochs), "--patience", "5"]
         + (["--resume"] if det_last.exists() else []), {}),
        ("det_eval", None, {}),
        ("c5c_train", [PY, "-u", f"{BT}/train_ball_temporal.py", "--data", str(C5C_DATA), "--out", str(C5C_OUT),
                       "--channels", "5", "--motion_blur", "0.25", "--epochs", str(a.c5c_epochs), "--patience", "6",
                       "--init", "runs/ball_temporal/c5c/best.pt"]
         + (["--resume"] if (C5C_OUT / "last.pt").exists() else []), {}),
        ("c5c_eval", None, {}),
    ]


def run(name, cmd, env, log_dir: Path) -> bool:
    log = log_dir / f"{name}.log"
    print(f"[{time.strftime('%H:%M')}] {name}: {' '.join(cmd)}  (log {log})", flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M')} {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M')}] {name}: exit {r.returncode}", flush=True)
    return r.returncode == 0


def composite(name, log_dir):
    if name == "det_eval":
        return all(run(f"det_eval_{k}", c, {}, log_dir) for k, c in [
            ("student", [PY, "-u", f"{S3C}/student_detector.py", "eval", "--weights",
                         "runs/student/d1_s1280_c5/weights/best.pt"]),
            ("baseline", [PY, "-u", f"{S3C}/student_detector.py", "eval", "--baseline", "--weights",
                          "runs/yolo26s_athlete_ball/best.pt"])])
    if name == "c5c_eval":
        return all(run(f"c5c_eval_{k}", [PY, "-u", "scripts/ball_eval/eval_ball_temporal.py", "all", "--ckpt", ck,
                                          "--src", "data/gsr", "--cache", str(CACHE), "--data", str(C5C_DATA),
                                          "--out", str(OUT / "eval" / f"c5c_{k}")], {}, log_dir)
                   for k, ck in [("student", str(C5C_OUT / "best.pt")), ("baseline", "runs/ball_temporal/c5c/best.pt")])
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="run just these stages (still skipping done ones)")
    ap.add_argument("--budget_gb", type=float, default=230)
    ap.add_argument("--det_epochs", type=int, default=20)
    ap.add_argument("--det_batch", type=int, default=8)
    ap.add_argument("--c5c_epochs", type=int, default=15)
    ap.add_argument("--c5c_easy_step", type=int, default=9)
    ap.add_argument("--c5c_hard_step", type=int, default=2)
    ap.add_argument("--min_free_gb", type=float, default=40, help="refuse c5c_data below this free space")
    a = ap.parse_args()
    state, logs = ROOT / OUT / "state", ROOT / OUT / "logs"
    state.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    st = {name: (cmd, env) for name, cmd, env in stages(a)}

    def chain(names):
        for name in names:
            if a.only is not None and name not in a.only:
                continue
            if (state / f"{name}.done").exists():
                continue
            if name == "c5c_data":
                while not (state / "relabel_gsr.done").exists():      # needs the teacher labels of GSR
                    if errs:
                        raise SystemExit("c5c_data: relabel_gsr failed")
                    time.sleep(60)
            if name == "c5c_data" and shutil.disk_usage(ROOT).free / 1e9 < a.min_free_gb:
                raise SystemExit(f"c5c_data: less than {a.min_free_gb} GB free; free space first (voxblink/voxceleb)")
            cmd, env = st[name]
            ok = composite(name, logs) if cmd is None else run(name, cmd, env, logs)
            if not ok:
                raise SystemExit(f"stage {name} failed; see {logs / name}.log")
            (state / f"{name}.done").write_text(time.strftime("%Y-%m-%d %H:%M"))

    errs = []
    chain(SERIAL)
    # after pseudo_seqs the detector branch and the c5c branch share nothing but the GPU: run them side by side

    def branch(names):
        try:
            chain(names)
        except SystemExit as e:
            errs.append(str(e))

    ts = [threading.Thread(target=branch, args=(b,)) for b in BRANCHES]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise SystemExit("; ".join(errs))
    print("all stages done", flush=True)


if __name__ == "__main__":
    main()
