"""
gsr_pipeline — Lightweight SoccerNet Game State Reconstruction baseline.

Key differences from the original sn-gamestate baseline:
- YOLOv8n (nano) instead of YOLOv8m / YOLO11m
- torchreid OSNet instead of PRTReId / BPBreID
- EasyOCR instead of MMOCR for jersey-number reading
- ByteTrack (built-in to ultralytics) instead of StrongSORT
- No tracklab / Hydra dependency — plain Python orchestration
"""

from gsr_pipeline.pipeline import GSRPipeline

__all__ = ["GSRPipeline"]
__version__ = "0.1.0"
