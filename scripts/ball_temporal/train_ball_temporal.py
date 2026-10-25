"""
train_ball_temporal.py -- step 3 of docs/path_b/plans/BALL_TEMPORAL_PLAN.md §9: train the temporal
heatmap specialist (heatmap_model.py) on the step 2 dataset (build_ball_temporal_dataset.py).

Single GPU. Standalone apart from heatmap_model.py (same folder). Needs torch, torchvision,
numpy, opencv-python, matplotlib.

  * `outside` crops (the runtime window misses the labelled ball) are not used for training and
    are not counted as empty crops: they are centred on a main-detector sighting, which is often a
    real but unlabelled ball (a spare ball by the boards). Labelled empty, they penalised correct
    detections (2026-10-14). Validation reports how often they fire, for reference.
  * Train on `train` + `pool`, sampled with replacement in proportion to the manifest's `repeat`
    (hard pool positives x3). One epoch = sum(repeat) samples.
  * Augmentation (plan §5): horizontal flip; the same brightness/contrast/colour gain on all three
    frames (on the GPU); a small translation (+-16 px, exposed border filled grey); and in 5% of samples
    t-1 and t-2 replaced by t (sequence start or a cut, plan §4). Optional (--motion_blur p, added
    2026-10-15, off by default so earlier runs reproduce): in a share p of samples, the same linear
    motion blur (3-9 px, random direction) on all three frames, like a fast pan or a fast ball.
  * Validation on `select` every epoch. Distance recall = peak within 15 px of a ball (plan §8),
    over positive crops. The selection score is its mean over the thresholds at which 1%, 2%, ...,
    10% of empty crops (`neg_hard`, `neg_random`) fire: a partial recall-vs-FPR area. Recall at a
    single 5%-FPR threshold swung 16-41% between epochs on noise in the empty crops' top scores, so
    it is reported but not selected on. Early stopping on the score, not on the loss.
  * AMP, AdamW, linear warm-up then cosine decay.

Output (<out>/): best.pt, last.pt (model + optimiser + epoch; resumable with --resume), history.csv,
curves.png, args.json, and DONE (finished: epochs or early stop) or INCOMPLETE (stopped by
--time_budget_h; run again with --resume).

Added for docs/plans/BALL_IMPROVEMENT_PLAN.md item 3 (2026-10-25):
  * --init <ckpt>: warm start from another checkpoint's weights (e.g. c5c/best.pt); optimiser and
    schedule start fresh. Use a lower --lr than the original run.
  * Multi-GPU: launch with `torchrun --nproc_per_node 2 train_ball_temporal.py ...`. Each GPU
    draws its own weighted sample stream (1/world of an epoch); validation, checkpoints and logs
    on rank 0 only.
  * --time_budget_h: stop after the epoch that would otherwise run past the budget (Kaggle's 12 h
    session limit), leaving last.pt to resume from.

    python scripts/ball_temporal/train_ball_temporal.py --data data/outputs/ball_temporal/dataset \
        --out runs/ball_temporal/c9 --channels 9
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heatmap_model import (BallHeatmapNet, build_input, decode, focal_loss,  # noqa: E402
                           make_targets, offset_loss)

CROP = 640
MAX_BALLS = 4
MATCH_PX = 15.0
EMPTY_FPR = 0.05            # reported operating point: this share of empty crops fire
SCORE_FPRS = np.linspace(0.01, 0.10, 10)   # selection score: mean recall over these empty-crop FPRs
METRIC = "score_v3"         # checkpoints selected on an older metric are re-scored on --resume
SHIFT = 16
REPEAT_T = 0.05
KINDS = ("hard", "easy", "outside", "neg_hard", "neg_random")


def read_manifest(data: Path, splits) -> list[dict]:
    with open(data / "manifest.csv", newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] in splits]


def parse_balls(s: str) -> np.ndarray:
    pts = [list(map(float, b.split())) for b in s.split(";") if b]
    return np.array(pts, np.float32).reshape(-1, 3)


def motion_blur(frames: np.ndarray, rng) -> np.ndarray:
    """The same linear blur kernel (length 3-9 px, random angle) on each frame (3, H, W, 3)."""
    k = int(rng.integers(3, 10))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2] = 1.0
    rot = cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return np.stack([cv2.filter2D(f, -1, kernel) for f in frames])


class TemporalCrops(Dataset):
    def __init__(self, data: Path, rows: list[dict], augment: bool, seed: int = 0, blur_p: float = 0.0):
        self.data, self.rows, self.augment, self.seed = data, rows, augment, seed
        self.blur_p = blur_p

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(str(self.data / r["file"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.data / r["file"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames = np.stack([img[:, k * CROP:(k + 1) * CROP] for k in range(3)])   # (3, H, W, 3)
        pts = parse_balls(r["balls"])
        if self.augment:
            rng = np.random.default_rng()
            frames, pts = self._augment(frames, pts, rng)
            if self.blur_p and rng.random() < self.blur_p:
                frames = motion_blur(np.ascontiguousarray(frames), rng)
        out = np.full((MAX_BALLS, 3), -1, np.float32)
        out[:len(pts)] = pts[:MAX_BALLS]
        x = torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2).reshape(9, CROP, CROP)))
        return x, torch.from_numpy(out), KINDS.index(r["kind"]), i

    @staticmethod
    def _augment(frames, pts, rng):
        if rng.random() < REPEAT_T:
            frames = np.stack([frames[2]] * 3)
        if rng.random() < 0.5:
            frames = frames[:, :, ::-1]
            pts = pts.copy()
            pts[:, 0] = CROP - pts[:, 0]
        f = frames                                      # colour jitter runs on the GPU (color_jitter)
        if rng.random() < 0.5:
            dx, dy = (int(v) for v in rng.integers(-SHIFT, SHIFT + 1, 2))
            g = np.full_like(f, 114)
            sx, tx = (slice(0, CROP - dx), slice(dx, CROP)) if dx >= 0 else (slice(-dx, CROP), slice(0, CROP + dx))
            sy, ty = (slice(0, CROP - dy), slice(dy, CROP)) if dy >= 0 else (slice(-dy, CROP), slice(0, CROP + dy))
            g[:, ty, tx] = f[:, sy, sx]
            f = g
            pts = pts.copy()
            pts[:, 0] += dx
            pts[:, 1] += dy
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < CROP) & (pts[:, 1] >= 0) & (pts[:, 1] < CROP)]
        return f, pts


def color_jitter(x: torch.Tensor) -> torch.Tensor:
    """Per sample, the same contrast/brightness/colour gain on all three frames. uint8 (B, 9, H, W) in and out."""
    B, dev = x.shape[0], x.device
    a = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.75, 1.25)       # contrast
    b = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(-25, 25)          # brightness
    gain = torch.empty(B, 1, 3, 1, 1, device=dev).uniform_(0.92, 1.08)    # colour
    f = x.view(B, 3, 3, *x.shape[-2:]).float()
    f = ((f - 128) * a + 128 + b) * gain
    return f.clamp_(0, 255).round_().to(torch.uint8).view_as(x)


def step_loss(model, x, pts, channels):
    logits, off = model(build_input(x, channels))
    hm, ind, off_gt, mask = make_targets(pts, logits.shape[-1])
    lh = focal_loss(logits, hm)
    lo = offset_loss(off, ind, off_gt, mask)
    return lh + lo, lh, lo, logits, off


@torch.no_grad()
def validate(model, loader, channels, dev, amp):
    model.eval()
    confs, hits, pos, kinds, losses = [], [], [], [], []
    for x, pts, kind, _ in loader:
        x, pts = x.to(dev, non_blocking=True), pts.to(dev, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            loss, *_rest, logits, off = step_loss(model, x, pts, channels)
        losses.append(loss.item())
        p = decode(logits, off)
        d = torch.cdist(p[:, None, :2], pts[..., :2]).squeeze(1)              # (B, K)
        d[pts[..., 2] <= 0] = float("inf")
        hits.append((d.min(1).values <= MATCH_PX).cpu())
        confs.append(p[:, 2].cpu())
        pos.append((pts[..., 2] > 0).any(1).cpu())
        kinds.append(kind)
    conf, hit, pos, kind = (torch.cat(v) for v in (confs, hits, pos, kinds))
    hard = kind == KINDS.index("hard")
    neg = kind >= KINDS.index("neg_hard")
    outside = conf[kind == KINDS.index("outside")]
    empty = conf[neg]
    thr = float(torch.quantile(empty, 1 - EMPTY_FPR)) if len(empty) else 0.5

    def rec(m, t):
        return float(((conf >= t) & hit & m).sum() / m.sum().clamp(min=1))

    def score(m):
        if not len(empty):
            return rec(m, 0.5)
        return float(np.mean([rec(m, float(torch.quantile(empty, 1 - f))) for f in SCORE_FPRS]))
    return {"val_loss": float(np.mean(losses)), "score": score(pos), "score_hard": score(pos & hard),
            "score_easy": score(pos & ~hard), "thr": thr,
            "recall": rec(pos, thr), "recall_hard": rec(pos & hard, thr),
            "recall_easy": rec(pos & ~hard, thr),
            "recall@0.5": rec(pos, 0.5),
            "outside_fire": float((outside >= thr).float().mean()) if len(outside) else 0.0, "empty_fpr@0.5": float((empty >= 0.5).float().mean()) if len(empty) else 0.0}


def plot(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="select")
    ax[0].set_title("loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    for k in ("score", "score_hard", "score_easy", "recall", "recall@0.5"):
        pts = [(h["epoch"], float(h[k])) for h in hist if h.get(k) not in (None, "")]
        if pts:
            ax[1].plot(*zip(*pts), marker=".", label=k)
    ax[1].set_title(f"select ({MATCH_PX:.0f} px): score = mean recall at 1-10% empty-crop FPR")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _init_distributed():
    """(world size, rank, local rank or None). torchrun sets WORLD_SIZE / RANK / LOCAL_RANK."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 1, 0, None
    import torch.distributed as dist
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))     # rank 0 validates while others wait
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return world, dist.get_rank(), local


def _agree(flag: bool, dev, world: int) -> bool:
    """Rank 0's decision, on every rank."""
    if world <= 1:
        return flag
    import torch.distributed as dist
    t = torch.tensor([int(flag)], device=dev)
    dist.broadcast(t, 0)
    return bool(t.item())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/outputs/ball_temporal/dataset"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--channels", type=int, choices=[5, 9], default=9)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8, help="epochs without a better select score")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=1.0, help="warm-up length in epochs")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true", help="skip the ImageNet weights download")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: samples per epoch and per validation")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--motion_blur", type=float, default=0.0, help="share of training samples given motion blur")
    ap.add_argument("--init", type=Path, default=None, help="warm start: load this checkpoint's weights (not optimiser)")
    ap.add_argument("--time_budget_h", type=float, default=0.0,
                    help="stop before an epoch would end past this many hours (0 = no limit); resume with --resume")
    a = ap.parse_args()

    world, rank, local = _init_distributed()
    main_rank = rank == 0
    log = (lambda *s, **k: print(*s, **k, flush=True)) if main_rank else (lambda *s, **k: None)
    t_start = time.time()
    torch.manual_seed(a.seed + rank)
    torch.backends.cudnn.benchmark = True
    dev = torch.device(f"cuda:{local}") if local is not None else torch.device(a.device)
    amp = not a.no_amp and dev.type == "cuda"
    a.out.mkdir(parents=True, exist_ok=True)
    if main_rank:
        for marker in ("DONE", "INCOMPLETE"):
            (a.out / marker).unlink(missing_ok=True)
        (a.out / "args.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, "world": world}, indent=1))

    tr_rows = [r for r in read_manifest(a.data, ("train", "pool")) if r["kind"] != "outside"]
    va_rows = read_manifest(a.data, ("select",))
    if a.limit:
        va_rows = [va_rows[i] for i in np.random.default_rng(a.seed).choice(len(va_rows), min(a.limit, len(va_rows)), replace=False)]
    weights = torch.tensor([float(r["repeat"]) for r in tr_rows])
    n_epoch = a.limit or int(weights.sum())
    sampler = WeightedRandomSampler(weights, max(n_epoch // world, a.batch), replacement=True,
                                    generator=torch.Generator().manual_seed(a.seed * 1000 + rank))
    tr = DataLoader(TemporalCrops(a.data, tr_rows, True, blur_p=a.motion_blur), a.batch, sampler=sampler, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    va = DataLoader(TemporalCrops(a.data, va_rows, False), a.batch, shuffle=False, num_workers=a.workers,
                    pin_memory=True, persistent_workers=a.workers > 0) if main_rank else None
    log(f"train {len(tr_rows)} crops ({n_epoch} per epoch over {world} GPU(s)), select {len(va_rows)}, "
        f"{a.channels} channels, batch {a.batch} per GPU, amp {amp}")

    net = BallHeatmapNet(a.channels, pretrained=not (a.no_pretrained or a.init)).to(dev)   # channels_last was 4x slower (3090, torch 2.6)
    if a.init and not (a.resume and (a.out / "last.pt").exists()):
        ck0 = torch.load(a.init, map_location=dev, weights_only=False)
        if int(ck0.get("channels", a.channels)) != a.channels:
            raise SystemExit(f"--init {a.init} has {ck0.get('channels')} channels, this run {a.channels}")
        net.load_state_dict(ck0["model"])
        log(f"warm start from {a.init} (epoch {ck0.get('epoch')})")
    model = net
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = len(tr)
    total, warm = a.epochs * steps_per_epoch, int(a.warmup * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))) * 0.99 + 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    hist, best, start, bad = [], -1.0, 1, 0
    if a.resume and (a.out / "last.pt").exists():
        ck = torch.load(a.out / "last.pt", map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        hist, best, bad, start = ck["hist"], ck["best"], ck["bad"], ck["epoch"] + 1
        if ck.get("metric") != METRIC and main_rank:   # selected on the old single-threshold recall: re-score
            print("re-scoring best.pt and last.pt with the current selection score", flush=True)
            v_last = validate(model, va, a.channels, dev, amp)
            bk = torch.load(a.out / "best.pt", map_location=dev, weights_only=False)
            tmp = BallHeatmapNet(a.channels, pretrained=False).to(dev)
            tmp.load_state_dict(bk["model"])
            v_best = validate(tmp, va, a.channels, dev, amp)
            del tmp
            for e, v in ((bk["epoch"], v_best), (ck["epoch"], v_last)):
                for h in hist:
                    if h["epoch"] == e:
                        h.update({k: v[k] for k in ("score", "score_hard", "score_easy")})
            print(f"  best.pt (epoch {bk['epoch']}) score {v_best['score']:.2%}, "
                  f"last.pt (epoch {ck['epoch']}) score {v_last['score']:.2%}", flush=True)
            if v_last["score"] > v_best["score"]:
                best, bad = v_last["score"], 0
                torch.save({"model": model.state_dict(), "channels": a.channels, "epoch": ck["epoch"],
                            "metrics": {**hist[-1], **v_last}}, a.out / "best.pt")
            else:
                best, bad = v_best["score"], ck["epoch"] - bk["epoch"]
        log(f"resumed at epoch {start}, best select score {best:.4f}, {bad} epochs since")

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local])
    epoch_min = []
    stopped_by_budget = False
    for ep in range(start, a.epochs + 1):
        model.train()
        t0, tl, n = time.time(), 0.0, 0
        for i, (x, pts, _, _) in enumerate(tr, 1):
            x = color_jitter(x.to(dev, non_blocking=True))
            pts = pts.to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=amp):
                loss, lh, lo, *_ = step_loss(model, x, pts, a.channels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl, n = tl + loss.item(), n + 1
            if i % 100 == 0 or i == steps_per_epoch:
                per_it = (time.time() - t0) / i
                eta_ep = per_it * (steps_per_epoch - i) / 60
                mem = f" mem {torch.cuda.max_memory_allocated(dev) / 2**30:.1f}G" if dev.type == "cuda" else ""
                log(f"  ep {ep}/{a.epochs} {i}/{steps_per_epoch} loss {tl / n:.4f} (hm {lh.item():.3f} off {lo.item():.3f}) "
                    f"lr {sched.get_last_lr()[0]:.2e} {per_it:.2f}s/it, epoch ETA {eta_ep:.1f} min{mem}")
        stop = False
        if main_rank:
            v = validate(net, va, a.channels, dev, amp)
            h = {"epoch": ep, "train_loss": tl / max(n, 1), **v, "minutes": (time.time() - t0) / 60}
            hist.append(h)
            epoch_min.append(h["minutes"])
            improved = v["score"] > best
            if improved:
                best, bad = v["score"], 0
                torch.save({"model": net.state_dict(), "channels": a.channels, "epoch": ep, "metrics": h},
                           a.out / "best.pt")
            else:
                bad += 1
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "channels": a.channels, "epoch": ep, "hist": hist,
                        "best": best, "bad": bad, "metric": METRIC}, a.out / "last.pt")
            with open(a.out / "history.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(h), restval="")
                w.writeheader()
                w.writerows(hist)
            plot(hist, a.out / "curves.png")
            left = (a.epochs - ep) * float(np.mean(epoch_min[-3:]))
            log(f"epoch {ep}/{a.epochs}: train {h['train_loss']:.4f} select loss {v['val_loss']:.4f} | score {v['score']:.2%} "
                f"(hard {v['score_hard']:.2%}, easy {v['score_easy']:.2%}) | recall {v['recall']:.2%} "
                f"(hard {v['recall_hard']:.2%}, easy {v['recall_easy']:.2%}) at thr {v['thr']:.3f} | "
                f"outside fire {v['outside_fire']:.0%} | recall@0.5 {v['recall@0.5']:.2%}, empty FPR@0.5 {v['empty_fpr@0.5']:.2%} | "
                f"{h['minutes']:.1f} min{'  *best*' if improved else ''} | run ETA {left / 60:.1f} h")
            if bad >= a.patience:
                log(f"early stop: no better select score for {a.patience} epochs")
                stop = True
            elif a.time_budget_h and ep < a.epochs:
                used_h = (time.time() - t_start) / 3600
                next_h = 1.15 * max(epoch_min[-3:]) / 60
                if used_h + next_h > a.time_budget_h:
                    log(f"time budget: {used_h:.2f} h used, next epoch ~{next_h:.2f} h, budget {a.time_budget_h} h -> "
                        f"stopping after epoch {ep}; continue with --resume")
                    stop = stopped_by_budget = True
        stop = _agree(stop, dev, world)
        if stop:
            break
    if main_rank:
        (a.out / ("INCOMPLETE" if stopped_by_budget else "DONE")).write_text(
            json.dumps({"best_score": best, "epochs_run": len(hist), "hours": (time.time() - t_start) / 3600}))
        log(f"best select score {best:.2%}; {a.out / 'best.pt'} ({'INCOMPLETE' if stopped_by_budget else 'DONE'})")
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
