"""
evaluate_athlete_roles.py — score athlete_roles.AthleteRoleTeamClassifier against SoccerNet-GSR labels.

Uses ground-truth boxes and ground-truth pitch positions, so it measures the classification
step alone (detector errors and calibration errors are excluded). The classifier is fitted on
the first --fit_frames frames and scored on the later, held-out frames; the fit window is
reported too, for reference.

Usage:
    python -m gsr_pipeline.evaluate_athlete_roles \\
        --sequence_dir data/datasets/SoccerNetGS/gamestate-2024/valid/SNGS-021
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

from .athlete_roles import AthleteRoleTeamClassifier, torso_descriptor

PERSON_ROLES = ("player", "goalkeeper", "referee", "other")


def load_rows(sequence_dir: Path, step: int) -> List[dict]:
    with open(sequence_dir / "Labels-GameState.json", "r") as f:
        data = json.load(f)
    images = sorted(((img["image_id"], img["file_name"]) for img in data["images"] if img.get("is_labeled", True)),
                    key=lambda pair: pair[1])
    by_image: Dict[str, list] = defaultdict(list)
    for ann in data["annotations"]:
        attrs = ann.get("attributes") or {}
        box, pitch = ann.get("bbox_image"), ann.get("bbox_pitch")
        if attrs.get("role") in PERSON_ROLES and box and pitch:
            by_image[ann["image_id"]].append((attrs["role"], attrs.get("team"), box, pitch))

    img_dir = sequence_dir / "img1" if (sequence_dir / "img1").is_dir() else sequence_dir / "images"
    rows = []
    for frame_idx, (image_id, file_name) in list(enumerate(images))[::step]:
        frame = cv2.imread(str(img_dir / file_name))
        for role, team, box, pitch in by_image.get(image_id, []):
            ltrb = (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"])
            descriptor = torso_descriptor(frame, ltrb)
            if descriptor is not None:
                rows.append({"frame": frame_idx, "role": role, "team": team, "descriptor": descriptor,
                             "pitch": (pitch["x_bottom_middle"], pitch["y_bottom_middle"])})
    return rows


def _role(label: str) -> str:
    return label.split("_")[0]


def _team(label: str):
    return label[-1] if label.startswith(("player_team", "goalkeeper_team")) else None


def score(rows: List[dict], labels: List[str]) -> dict:
    confusion = Counter((r["role"], _role(p)) for r, p in zip(rows, labels))
    per_role = {}
    for role in PERSON_ROLES:
        total = sum(n for (g, _), n in confusion.items() if g == role)
        if total:
            per_role[role] = {"n": total, "recall": confusion[(role, role)] / total}
    pairs = [(r["team"], _team(p)) for r, p in zip(rows, labels)
             if r["team"] in ("left", "right") and _team(p) in ("a", "b")]
    agree = sum(1 for gt, pr in pairs if (gt == "left") == (pr == "a"))
    return {
        "n": len(rows),
        "role_accuracy": sum(n for (g, p), n in confusion.items() if g == p) / len(rows) if rows else None,
        "per_role": per_role,
        "confusion": {f"{g}->{p}": n for (g, p), n in sorted(confusion.items())},
        "team_accuracy": max(agree, len(pairs) - agree) / len(pairs) if pairs else None,
        "team_pairs": len(pairs),
    }


def _fmt(value) -> str:
    return "  n/a " if value is None else f"{value * 100:5.1f}%"


def print_report(title: str, result: dict) -> None:
    print(f"\n{title}  ({result['n']} detections)")
    print(f"  role accuracy   {_fmt(result['role_accuracy'])}")
    for role, m in result["per_role"].items():
        print(f"    {role:<11} recall {_fmt(m['recall'])}  (n={m['n']})")
    print(f"  team accuracy   {_fmt(result['team_accuracy'])}  (players + goalkeepers predicted with a team, n={result['team_pairs']})")
    print("  confusion (GT -> predicted): " + "  ".join(f"{k}={v}" for k, v in result["confusion"].items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score athlete role/team classification against SoccerNet-GSR labels")
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--config", default="gsr_pipeline/configs/default.yaml")
    parser.add_argument("--step", type=int, default=5, help="use every Nth frame")
    parser.add_argument("--fit_frames", type=int, default=250, help="fit on frames before this index, score the rest")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f).get("athlete_roles", {})
    rows = load_rows(Path(args.sequence_dir), args.step)
    fit_rows = [r for r in rows if r["frame"] < args.fit_frames]
    held_rows = [r for r in rows if r["frame"] >= args.fit_frames]
    if not fit_rows or not held_rows:
        raise SystemExit(f"--fit_frames {args.fit_frames} leaves an empty fit or held-out set")

    classifier = AthleteRoleTeamClassifier(**cfg).fit(
        np.array([r["descriptor"] for r in fit_rows]),
        np.array([r["pitch"] for r in fit_rows]),
        [r["frame"] for r in fit_rows],
    )
    for title, subset in (("Fit window", fit_rows), ("Held-out frames", held_rows)):
        labels = classifier.predict(np.array([r["descriptor"] for r in subset]), np.array([r["pitch"] for r in subset]))
        print_report(title, score(subset, labels))


if __name__ == "__main__":
    main()
