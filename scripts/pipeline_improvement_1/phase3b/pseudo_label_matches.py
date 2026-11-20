"""
pseudo_label_matches.py -- pseudo-label SoccerNet full matches (videos-HQ) with the Phase 3b teacher, inside a disk
budget (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md, Phase 3b, "After the gate").

Per match, one at a time (the next match downloads while the current one is labelled):
  1. download 1_HQ.mkv, 2_HQ.mkv, video.ini (fullmatch_probe.download; gated repo, `hf auth login` first)
  2. check each half: 1920x1080 at ~25 fps, else the half is skipped (the decode knobs are px/frame at 1080p 25 fps)
  3. leakage: dHash scan for every select/test GSR clip (fullmatch_probe.scan_half); a match holding one is
     skipped and listed in exclude.txt
  4. --windows_per_half windows of --seconds each, spread over the half after its video.ini start, jittered
     (seeded by the game name); extracted at native rate to <out>/<game>/<win>/frames, labelled by
     pseudo_label.label_dir (labels.json + yolo/*.txt beside frames/)
  5. delete the match's video files; write <out>/<game>/done.json (resume: done games are skipped)

Stops when <out> reaches --budget_gb, or G: has less than --min_free_gb free. Games come from
fullmatch_probe.plan_games (same seed -> the probe's 3 games come first, their downloads are reused).

    python scripts/pipeline_improvement_1/phase3b/pseudo_label_matches.py --n_games 3 --budget_gb 150
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase3"))
import fullmatch_probe as fp  # noqa: E402
import pseudo_label as pl  # noqa: E402

ROOT = fp.ROOT


def dir_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9 if p.exists() else 0.0


def windows(game: str, start_s: float, dur_s: float, n: int, seconds: float) -> list[float]:
    """n window starts spread over [start + 60, end - seconds - 60], one per equal slot, jittered inside it."""
    lo, hi = start_s + 60.0, dur_s - seconds - 60.0
    if hi <= lo:
        return []
    rng = random.Random(int(hashlib.md5(game.encode()).hexdigest()[:8], 16))
    slot = (hi - lo) / n
    return [round(lo + i * slot + rng.uniform(0, max(0.0, slot - seconds)), 1) for i in range(n)]


def remote_size(game: str) -> tuple[int, int] | None:
    """(width, height) of the game's 1_HQ.mkv read over HTTP by ffprobe (headers only, no download)."""
    import subprocess
    from huggingface_hub import get_token, hf_hub_url
    url = hf_hub_url("SoccerNet/SoccerNet_raw_HQ", f"{game}/1_HQ.mkv", repo_type="dataset", revision="videos-HQ")
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-headers", f"Authorization: Bearer {get_token()}\r\n",
                            "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", url],
                           capture_output=True, text=True, timeout=120)
        st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
        return (st["width"], st["height"]) if "width" in st else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def screen(games: list[tuple[str, str]], cache: Path, workers: int = 8) -> dict:
    """{game: [w, h] | None} for every game, cached in `cache` (only 1080p games are worth downloading)."""
    from concurrent.futures import ThreadPoolExecutor
    res = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [g for _, g in games if g not in res]
    with ThreadPoolExecutor(workers) as ex:
        for k, (g, wh) in enumerate(zip(todo, ex.map(remote_size, todo)), 1):
            res[g] = list(wh) if wh else None
            if k % 50 == 0 or k == len(todo):
                cache.write_text(json.dumps(res, indent=0))
                print(f"  screened {k}/{len(todo)}", flush=True)
    return res


def download_retry(game: tuple[str, str], root: Path, tries: int = 4) -> None:
    """fp.download with retries; a network error must not kill the prefetch thread silently."""
    for k in range(1, tries + 1):
        try:
            fp.download([game], root)
            return
        except Exception as e:  # noqa: BLE001 -- any download error: log, wait, retry
            print(f"  download {game[1]} try {k}/{tries} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(60 * k)


def slug(game: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in game)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_games", type=int, default=100, help="games to plan (stops earlier at the budget)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows_per_half", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--budget_gb", type=float, default=150.0, help="cap on <out> (frames + labels)")
    ap.add_argument("--min_free_gb", type=float, default=25.0, help="stop if the drive has less free than this")
    ap.add_argument("--video_root", type=Path, default=ROOT / "data" / "soccernet_fullmatch")
    ap.add_argument("--gsr_root", type=Path, default=ROOT / "data" / "gsr")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "pseudo_labels" / "phase3b_teacher")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--keep_video", action="store_true", help="do not delete the match videos after labelling")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    allg = fp.plan_games(10**6, a.seed, ["train", "valid", "test"])  # every game, in the probe's spread order
    sizes = screen(allg, a.out / "screen.json")
    hd = [(s, g) for s, g in allg if sizes.get(g) == [1920, 1080]]
    print(f"screen: {len(hd)}/{len(allg)} games are 1080p", flush=True)
    games = hd[:a.n_games]
    (a.out / "plan.txt").write_text("".join(f"{s}\t{g}\n" for s, g in games))
    clips = fp.gsr_clips(a.gsr_root, map_all=False)
    print(f"{len(games)} games planned; {len(clips)} select/test GSR clips to screen for", flush=True)
    exclude_f = a.out / "exclude.txt"
    exclude = set(exclude_f.read_text().split()) if exclude_f.exists() else set()

    todo = [(s, g) for s, g in games if not (a.out / slug(g) / "done.json").exists() and g not in exclude]
    prefetch: dict[str, threading.Thread] = {}

    def fetch(i: int) -> None:
        if i < len(todo) and todo[i][1] not in prefetch:
            t = threading.Thread(target=download_retry, args=(todo[i], a.video_root), daemon=True)
            t.start()
            prefetch[todo[i][1]] = t

    fetch(0)
    for i, (spl, g) in enumerate(todo):
        used, free = dir_gb(a.out), shutil.disk_usage(a.out).free / 1e9
        if used >= a.budget_gb or free < a.min_free_gb:
            print(f"stop: {used:.1f} GB used of {a.budget_gb}, {free:.0f} GB free", flush=True)
            break
        t0 = time.time()
        prefetch[g].join()
        fetch(i + 1)                                    # next match downloads while this one is labelled
        gdir, vdir = a.out / slug(g), a.video_root / g
        if not any((vdir / f).exists() for f in fp.HQ_FILES[:2]):
            print(f"[{i + 1}/{len(todo)}] {g}: download failed, skipped (not marked done; a rerun retries it)", flush=True)
            continue
        ini = fp.read_ini(vdir / "video.ini")
        info = {"game": g, "split": spl, "halves": {}, "windows": []}
        leaked = []
        for half in (1, 2):
            fn = f"{half}_HQ.mkv"
            p = vdir / fn
            if not p.exists():
                info["halves"][fn] = "missing"
                continue
            pr = fp.ffprobe(p)
            info["halves"][fn] = pr
            if (pr.get("width"), pr.get("height")) != (1920, 1080) or abs((pr.get("fps") or 0) - 25) > 0.5:
                print(f"  {g}/{fn}: {pr} -> skipped (not 1080p25)", flush=True)
                continue
            off = fp.start_offset_s(ini, fn)
            hc = [c for c in clips if c["half"] == half]
            for r in (fp.scan_half(p, off, hc, 30.0, 5.0) if hc else []):
                if sum(d <= 8 for d in r.get("best_dist", [])) >= 3:
                    leaked.append(r["seq"])
            if leaked:
                break
            for s0 in windows(g, off, pr["duration_s"], a.windows_per_half, a.seconds):
                wdir = gdir / f"h{half}_{int(s0):05d}"
                pl.extract(p, s0, a.seconds, wdir / "frames")
                _, preds = pl.label_dir(wdir / "frames", wdir, a.device, a.batch)
                info["windows"].append({"half": half, "start_s": s0, "frames": len(preds),
                                        "labelled": sum(v is not None for v in preds.values())})
        if leaked:
            print(f"  {g}: holds held-out GSR clip(s) {leaked} -> excluded", flush=True)
            exclude.add(g)
            exclude_f.write_text("".join(f"{x}\n" for x in sorted(exclude)))
            shutil.rmtree(gdir, ignore_errors=True)
        else:
            gdir.mkdir(parents=True, exist_ok=True)
            info["minutes"] = round((time.time() - t0) / 60, 1)
            (gdir / "done.json").write_text(json.dumps(info, indent=1))
        if not a.keep_video:
            for f in fp.HQ_FILES[:2]:
                (vdir / f).unlink(missing_ok=True)
        print(f"[{i + 1}/{len(todo)}] {g}: {len(info['windows'])} windows, {(time.time() - t0) / 60:.1f} min, "
              f"out {dir_gb(a.out):.1f} GB", flush=True)


if __name__ == "__main__":
    main()
