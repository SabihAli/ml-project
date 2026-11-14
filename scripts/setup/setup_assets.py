"""
setup_assets.py -- the gitignored runtime assets of a fresh checkout, for the root Makefile.
Each step is idempotent: what already exists is skipped.

  gpu          print 1 if an NVIDIA GPU is visible (nvidia-smi), else 0
  checkpoints  download the NBJW calibration checkpoints (SV_kp, SV_lines) from Zenodo, the same
               URLs gsr_pipeline/calibrate.py uses, to the paths in configs/default.yaml
  engines      build the TensorRT FP16 engines of the committed detector and c5c ONNX files (the
               runtime's ensure_engine; engines are per GPU/driver/TRT, so they are never committed)
               and report the backend each model ends up on
  data         download SoccerNet GSR (gamestate-2024, no password) and extract to data/gsr/<split>

    python scripts/setup/setup_assets.py checkpoints
    python scripts/setup/setup_assets.py engines --device cuda:0
    python scripts/setup/setup_assets.py data --splits valid
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "gsr_pipeline" / "configs" / "default.yaml"


def _cfg():
    import yaml
    return yaml.safe_load(CONFIG.read_text())


def gpu(_a):
    print(1 if shutil.which("nvidia-smi") else 0)


def checkpoints(_a):
    from gsr_pipeline.calibrate import _KP_URL, _LINES_URL, _ensure_download
    c = _cfg().get("calibration", {})
    for url, key in ((_KP_URL, "checkpoint_kp"), (_LINES_URL, "checkpoint_l")):
        p = ROOT / c[key]
        _ensure_download(url, p)
        print(f"{p.relative_to(ROOT)}: {p.stat().st_size / 1e6:.1f} MB")


def engines(a):
    from gsr_pipeline.ball_specialist import BallSpecialist
    from gsr_pipeline.detect_athlete import AthleteBallDetector
    cfg = _cfg()
    d = AthleteBallDetector(onnx_path=str(ROOT / cfg["detector"]["onnx_path"]), device=a.device, auto_build_engine=True)
    s = BallSpecialist(onnx_path=str(ROOT / cfg["ball_specialist"]["onnx_path"]), device=a.device, auto_build_engine=True)
    got = {"detector": str(d.backend), "c5c": s.backend.value}
    print("backends:", got)
    if any("tensorrt" not in v for v in got.values()):
        sys.exit("a model is not on TensorRT: check the `tensorrt` package and the log above")


def data(a):
    from SoccerNet.Downloader import SoccerNetDownloader
    dl, out = ROOT / "data" / "gsr_download", ROOT / "data" / "gsr"
    for split in a.splits:
        dest = out / split
        if dest.exists() and any(dest.rglob("Labels-GameState.json")):
            print(f"{dest.relative_to(ROOT)}: present")
            continue
        SoccerNetDownloader(LocalDirectory=str(dl)).downloadDataTask(task="gamestate-2024", split=[split])
        zp = next(dl.rglob(f"{split}.zip"))
        print(f"extracting {zp.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["gpu", "checkpoints", "engines", "data"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--splits", nargs="+", default=["valid"], choices=["train", "valid", "test", "challenge"])
    a = ap.parse_args()
    {"gpu": gpu, "checkpoints": checkpoints, "engines": engines, "data": data}[a.step](a)


if __name__ == "__main__":
    main()
