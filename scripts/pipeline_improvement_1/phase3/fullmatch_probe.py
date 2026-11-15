"""
fullmatch_probe.py -- before pseudo-labelling SoccerNet full matches at scale (Phase 3b,
docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md), download a few `videos-HQ` matches and answer:

  1. resolution: is HQ really 1080p for these matches (ffprobe: width, height, fps, codec, duration)?
  2. video.ini: where does each half start inside the HQ file (HQ files include pre-match footage)?
  3. leakage: does a downloaded match contain one of our held-out GSR clips (`select` / `test`)?
     GSR's `game_id` is a local index, not a SoccerNet match name, so this is checked from the
     pixels: each GSR clip's game time (info.game_time_start, "1 - 01:16") plus the half's
     video.ini offset gives where it would sit in the HQ file; frames around there are dHashed
     and compared with the clip's frames. A match that holds a held-out clip goes to exclude.txt.

Steps (each skippable):
  plan      pick --n_games SoccerNet games, spread over leagues and seasons, deterministic (--seed);
            or take --games (a text file, one SoccerNet game path per line). No network.
  download  1_HQ.mkv, 2_HQ.mkv, video.ini per game via SoccerNet's downloader (Hugging Face,
            repo SoccerNet/SoccerNet_raw_HQ, branch videos-HQ, ~4.4 GB per match). The repo is
            gated: accept its terms on huggingface.co and log in first (`huggingface-cli login`
            or HF_TOKEN).
  probe     ffprobe + video.ini per file.
  leakage   dHash comparison against GSR held-out clips (--map_all: every GSR clip, which also gives
            the clip -> full-match map needed for cross-fitting by game; ~5x slower).

    python scripts/pipeline_improvement_1/phase3/fullmatch_probe.py --n_games 3
    python scripts/pipeline_improvement_1/phase3/fullmatch_probe.py --games my_games.txt --skip_download

Outputs (--out): plan.csv, probe.csv, leakage.csv (one row per match x GSR clip checked),
exclude.txt (matches holding a select/test clip), summary.md.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "ball_temporal"))

HQ_FILES = ["1_HQ.mkv", "2_HQ.mkv", "video.ini"]
HELD_OUT = ("select", "test")
CLIP_PROBES_S = (0.0, 6.0, 12.0, 18.0, 24.0)       # seconds into a 30 s GSR clip


# -- plan ---------------------------------------------------------------------------------

def plan_games(n: int, seed: int, splits) -> list[tuple[str, str]]:
    """(soccernet_split, game) for n games, round-robin over (league, season) so they spread."""
    from SoccerNet.utils import getListGames
    groups = defaultdict(list)
    for spl in splits:
        for g in getListGames(spl):
            g = g.replace("\\", "/")
            league, season = g.split("/")[:2]
            groups[(league, season)].append((spl, g))
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    out = []
    while len(out) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def games_from_file(path: Path) -> list[tuple[str, str]]:
    from SoccerNet.utils import getListGames
    split_of = {g.replace("\\", "/"): s for s in ("train", "valid", "test") for g in getListGames(s)}
    out = []
    for line in path.read_text().splitlines():
        g = line.strip().replace("\\", "/")
        if g:
            out.append((split_of.get(g, "train"), g))
    return out


# -- download -----------------------------------------------------------------------------

def download(games, root: Path) -> None:
    from SoccerNet.Downloader import SoccerNetDownloader
    dl = SoccerNetDownloader(LocalDirectory=str(root))
    for i, (spl, g) in enumerate(games, 1):
        missing = [f for f in HQ_FILES if not (root / g / f).exists()]
        if not missing:
            print(f"[{i}/{len(games)}] {g}: already here", flush=True)
            continue
        print(f"[{i}/{len(games)}] {g}: downloading {missing}", flush=True)
        dl.downloadGame(game=g, files=missing, spl=spl)


# -- probe --------------------------------------------------------------------------------

def read_ini(path: Path) -> dict:
    """video.ini -> {file: {key: value}}; keys as found (expected start_time_second, duration_second)."""
    if not path.exists():
        return {}
    cp = configparser.ConfigParser()
    cp.read(path)
    return {s: dict(cp[s]) for s in cp.sections()}


def ffprobe(path: Path) -> dict:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,avg_frame_rate,codec_name:format=duration", "-of", "json",
                            str(path)], capture_output=True, text=True, timeout=120)
        d = json.loads(r.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return {"error": repr(e)}
    s = (d.get("streams") or [{}])[0]
    num, _, den = str(s.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {"width": s.get("width"), "height": s.get("height"), "fps": round(fps, 3),
            "codec": s.get("codec_name"), "duration_s": float((d.get("format") or {}).get("duration", 0) or 0)}


def start_offset_s(ini: dict, file: str) -> float:
    sec = ini.get(file, {})
    for k in ("start_time_second", "start_time", "start"):
        if k in sec:
            try:
                return float(sec[k])
            except ValueError:
                pass
    return 0.0


# -- leakage ------------------------------------------------------------------------------

def dhash(img, size: int = 8) -> int:
    """phase0_report.py's dHash (the same near-duplicate test as the split-integrity check)."""
    import cv2
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size), interpolation=cv2.INTER_AREA)
    v = 0
    for bit in (g[:, 1:] > g[:, :-1]).flatten():
        v = (v << 1) | int(bit)
    return v


def gsr_clips(gsr_root: Path, map_all: bool) -> list[dict]:
    """GSR clips to look for: half, game-time second, and dHashes of frames CLIP_PROBES_S into the clip."""
    from gsr_io import SeqReader, decode, discover, parse_labels
    clips = []
    for ref in discover(gsr_root):
        if not map_all and ref.split not in HELD_OUT:
            continue
        rd = SeqReader(ref)
        try:
            d = rd.labels()
            info = d.get("info", {})
            half, _, clock = str(info.get("game_time_start", "")).partition(" - ")
            mm, _, ss = clock.partition(":")
            if not (half.strip().isdigit() and mm.strip().isdigit() and ss.strip().isdigit()):
                print(f"  {ref.name}: no usable game_time_start ({info.get('game_time_start')!r}), skipped")
                continue
            fps = float(info.get("frame_rate", 25))
            im_dir, files, *_ = parse_labels(d)
            names = sorted(files)
            hashes = []
            for t in CLIP_PROBES_S:
                k = min(len(names) - 1, int(round(t * fps)))
                hashes.append(dhash(decode(rd.read(f"{im_dir}/{files[names[k]]}"))))
            clips.append({"seq": ref.name, "split": ref.split, "half": int(half), "t0": int(mm) * 60 + int(ss),
                          "hashes": hashes})
        finally:
            rd.close()
    return clips


def scan_half(video: Path, offset_s: float, clips: list[dict], window_s: float, sample_fps: float) -> list[dict]:
    """For each clip of this half: best Hamming distance of each probe frame to frames sampled
    around where the clip should be in the HQ file."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return [{"seq": c["seq"], "error": "cannot open video"} for c in clips]
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(vfps / sample_fps)))
    rows = []
    for c in clips:
        t_lo = max(0.0, offset_s + c["t0"] - window_s)
        t_hi = offset_s + c["t0"] + CLIP_PROBES_S[-1] + window_s
        cap.set(cv2.CAP_PROP_POS_MSEC, t_lo * 1000.0)
        found = []
        i = 0
        while True:
            if not cap.grab():
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    found.append((cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, dhash(frame)))
                if found and found[-1][0] > t_hi:
                    break
            i += 1
        best = []
        for h in c["hashes"]:
            d, t = min(((bin(h ^ fh).count("1"), ft) for ft, fh in found), default=(64, None))
            best.append((d, t))
        rows.append({"seq": c["seq"], "split": c["split"], "frames_scanned": len(found),
                     "best_dist": [d for d, _ in best], "best_time_s": [None if t is None else round(t, 2) for _, t in best]})
    cap.release()
    return rows


# -- main ---------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_games", type=int, default=3)
    ap.add_argument("--games", type=Path, help="text file, one SoccerNet game path per line (overrides --n_games)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                    help="SoccerNet (action spotting) splits to draw games from; challenge lives in another repo")
    ap.add_argument("--video_root", type=Path, default=ROOT / "data" / "soccernet_fullmatch")
    ap.add_argument("--gsr_root", type=Path, default=ROOT / "data" / "gsr")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "outputs" / "pipeline_improvement_1" / "phase3b" / "fullmatch_probe")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_leakage", action="store_true")
    ap.add_argument("--map_all", action="store_true", help="look for every GSR clip, not only select/test")
    ap.add_argument("--window_s", type=float, default=30.0, help="search +- this around the expected clip time")
    ap.add_argument("--sample_fps", type=float, default=5.0)
    ap.add_argument("--max_dist", type=int, default=8, help="dHash Hamming distance (of 64) that counts as the same frame")
    ap.add_argument("--min_hits", type=int, default=3, help="probe frames (of 5) that must match to call a clip found")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    games = games_from_file(a.games) if a.games else plan_games(a.n_games, a.seed, a.splits)
    with open(a.out / "plan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["soccernet_split", "game"])
        w.writerows(games)
    print(f"{len(games)} games:", *[g for _, g in games], sep="\n  ")

    if not a.skip_download:
        download(games, a.video_root)

    probe = []
    for _, g in games:
        ini = read_ini(a.video_root / g / "video.ini")
        for half in (1, 2):
            fn = f"{half}_HQ.mkv"
            p = a.video_root / g / fn
            row = {"game": g, "file": fn, "present": p.exists(), "ini_start_s": start_offset_s(ini, fn),
                   "ini": json.dumps(ini.get(fn, {}))}
            row.update(ffprobe(p) if p.exists() else {})
            probe.append(row)
    keys = ["game", "file", "present", "width", "height", "fps", "codec", "duration_s", "ini_start_s", "ini", "error"]
    with open(a.out / "probe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(probe)

    leak_rows, exclude = [], set()
    if not a.skip_leakage:
        clips = gsr_clips(a.gsr_root, a.map_all)
        print(f"{len(clips)} GSR clips to look for ({'all' if a.map_all else 'select + test'})", flush=True)
        for _, g in games:
            ini = read_ini(a.video_root / g / "video.ini")
            for half in (1, 2):
                fn = f"{half}_HQ.mkv"
                p = a.video_root / g / fn
                todo = [c for c in clips if c["half"] == half]
                if not p.exists() or not todo:
                    continue
                print(f"  scanning {g}/{fn} for {len(todo)} clips", flush=True)
                for r in scan_half(p, start_offset_s(ini, fn), todo, a.window_s, a.sample_fps):
                    hits = sum(d <= a.max_dist for d in r.get("best_dist", []))
                    r.update(game=g, file=fn, hits=hits, found=hits >= a.min_hits)
                    leak_rows.append(r)
                    if r["found"] and r["split"] in HELD_OUT:
                        exclude.add(g)
        keys = ["game", "file", "seq", "split", "found", "hits", "best_dist", "best_time_s", "frames_scanned", "error"]
        with open(a.out / "leakage.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(leak_rows)
    (a.out / "exclude.txt").write_text("".join(f"{g}\n" for g in sorted(exclude)))

    res = defaultdict(int)
    for r in probe:
        if r.get("height"):
            res[f"{r['width']}x{r['height']}@{r['fps']}"] += 1
    found = [r for r in leak_rows if r["found"]]
    L = ["# Full-match probe (videos-HQ)", "",
         f"{len(games)} games, {sum(r['present'] for r in probe)}/{len(probe)} HQ files present.", "",
         "Resolutions: " + (", ".join(f"{k} x{v}" for k, v in sorted(res.items())) or "none probed") + ".", "",
         "| game | file | resolution | fps | duration s | video.ini start s |", "|---|---|---|---|---|---|"]
    for r in probe:
        L.append(f"| {r['game']} | {r['file']} | {r.get('width')}x{r.get('height')} | {r.get('fps')} | "
                 f"{r.get('duration_s', 0):.0f} | {r['ini_start_s']:.1f} |")
    L += ["", f"Leakage: {'skipped' if a.skip_leakage else f'{len(found)} GSR clip(s) found'} "
          f"(dHash <= {a.max_dist} on >= {a.min_hits}/5 probe frames, +-{a.window_s:.0f} s window). "
          f"Matches to exclude (hold a select/test clip): {len(exclude)}."]
    for r in found:
        L.append(f"- {r['seq']} ({r['split']}) in {r['game']}/{r['file']}: distances {r['best_dist']} at {r['best_time_s']} s")
    L += ["", "Review found rows by eye (the timestamps say where to look) before trusting exclude.txt; a near-miss "
          "row (2 hits) is worth a look too."]
    (a.out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
