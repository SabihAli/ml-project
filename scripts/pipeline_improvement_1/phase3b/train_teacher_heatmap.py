"""
train_teacher_heatmap.py -- Phase 3b teacher T1 (docs/plans/PIPELINE_IMPROVEMENT_PLAN_1.md, Phase 3b):
multi-frame heatmap ball detector at native resolution, the c5c recipe scaled up for an offline teacher.

Model: ResNet encoder (torchvision, ImageNet weights; resnet18/34/50), first conv widened to 5 frames x RGB
(15 channels, t-2..t+2, each frame given the RGB weights / 5), U-Net decoder to a stride-2 heatmap + sub-pixel
offset (as scripts/ball_temporal/heatmap_model.py). Point output: the label is a ball centre.
Loss: CenterNet focal loss on the heatmap (Gaussian sigma from the ball size) + L1 offset at the peak.

Train: native-scale crops (default 960x544) of 5 consecutive frames, centred near the ball with
--pos_prob, else anywhere (negatives); horizontal flip; per-sample contrast/brightness/colour gain identical
on all 5 frames. Frames t-2..t+2 are clamped at clip ends. GSR `train` + `pool` (every --stride-th centre frame);
`select` for checkpoint choice; `test` untouched until the dump.
Validate (every epoch): full frames of `select` every --val_stride-th frame, top-1 per frame, scored as the
eval (recall @15 px at the select-chosen threshold, precision >= 90%); best.pt keeps the best recall.
Dump: top-k peaks per full frame on `select` and `test`, written as train_teacher_yolo_motion.py dump does,
so trajectory_pass.py scores it (raw row = T1 alone) and runs the pass.

    python scripts/pipeline_improvement_1/phase3b/train_teacher_heatmap.py train --name t1_r34
    python scripts/pipeline_improvement_1/phase3b/train_teacher_heatmap.py dump --weights runs/teacher/t1_r34/best.pt
    python scripts/pipeline_improvement_1/phase3b/trajectory_pass.py --cands data/outputs/pipeline_improvement_1/phase3b/teacher_yolo/candidates/t1_r34
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_teacher_yolo as base  # noqa: E402
from gsr_io import SeqReader, discover, parse_labels  # noqa: E402

N_FR = 5                          # t-2 .. t+2
HALF = N_FR // 2
STRIDE = 2
MEAN = (0.485, 0.456, 0.406)      # RGB
STD = (0.229, 0.224, 0.225)
MAX_BALLS = 4
PAD_TO = 32


# -- data ---------------------------------------------------------------------------------

class Seq:
    def __init__(self, ref):
        rd = SeqReader(ref)
        try:
            im_dir, files, balls, *_rest, (W, H) = parse_labels(rd.labels())
        finally:
            rd.close()
        self.name, self.W, self.H = ref.name, W, H
        self.frames = sorted(files)
        self.paths = {f: str(Path(ref.container) / im_dir / files[f]) for f in self.frames}
        self.balls = {f: [(cx, cy, max(w, h, 2.0)) for cx, cy, w, h, _ in balls.get(f, [])] for f in self.frames}

    def window(self, k: int) -> list[int]:
        """Frame ids t-2..t+2 around index k, clamped at the clip ends."""
        n = len(self.frames)
        return [self.frames[min(max(k + d, 0), n - 1)] for d in range(-HALF, HALF + 1)]


def load_seqs(src: Path, splits) -> list[Seq]:
    refs = [r for r in discover(src) if r.split in splits]
    if any(r.kind != "dir" for r in refs):
        raise SystemExit("needs extracted sequence folders (data/gsr/<split>/SNGS-xxx/img1)")
    return [Seq(r) for r in refs]


class Crops(Dataset):
    def __init__(self, seqs, stride, crop_w, crop_h, pos_prob, seed=0):
        self.seqs, self.cw, self.ch, self.pos_prob = seqs, crop_w, crop_h, pos_prob
        self.items = [(i, k) for i, s in enumerate(seqs) for k in range(0, len(s.frames), stride)]
        self.seed = seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rng = random.Random(hash((self.seed, idx, time.time_ns())))
        si, k = self.items[idx]
        s = self.seqs[si]
        win = s.window(k)
        imgs = [cv2.imread(s.paths[f], cv2.IMREAD_COLOR) for f in win]
        H, W = imgs[0].shape[:2]
        balls = s.balls[s.frames[k]]
        cw, ch = min(self.cw, W), min(self.ch, H)
        if balls and rng.random() < self.pos_prob:
            bx, by, _ = rng.choice(balls)
            x0 = int(np.clip(bx - rng.uniform(0.1, 0.9) * cw, 0, W - cw))
            y0 = int(np.clip(by - rng.uniform(0.1, 0.9) * ch, 0, H - ch))
        else:
            x0, y0 = rng.randint(0, W - cw), rng.randint(0, H - ch)
        x = np.concatenate([im[y0:y0 + ch, x0:x0 + cw, ::-1] for im in imgs], 2)     # RGB x5, (ch, cw, 15)
        pts = np.zeros((MAX_BALLS, 3), np.float32)
        j = 0
        for bx, by, sz in balls:
            px, py = bx - x0, by - y0
            if 0 <= px < cw and 0 <= py < ch and j < MAX_BALLS:
                pts[j] = (px, py, sz)
                j += 1
        if rng.random() < 0.5:
            x = x[:, ::-1]
            pts[:j, 0] = cw - 1 - pts[:j, 0]
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1), torch.from_numpy(pts)


def jitter(x: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 15, H, W) -> same contrast / brightness / colour gain on all frames of a sample."""
    B = x.shape[0]
    a = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(0.75, 1.25)
    b = torch.empty(B, 1, 1, 1, 1, device=x.device).uniform_(-25, 25)
    g = torch.empty(B, 1, 3, 1, 1, device=x.device).uniform_(0.92, 1.08)
    f = x.view(B, N_FR, 3, *x.shape[-2:]).float()
    return (((f - 128) * a + 128 + b) * g).clamp_(0, 255).view_as(x)


def normalise(x: torch.Tensor) -> torch.Tensor:
    """float or uint8 (B, 15, H, W) RGB x5 in [0, 255] -> ImageNet-normalised float."""
    B = x.shape[0]
    f = x.float().view(B, N_FR, 3, *x.shape[-2:]) / 255
    m = f.new_tensor(MEAN).view(1, 1, 3, 1, 1)
    sd = f.new_tensor(STD).view(1, 1, 3, 1, 1)
    return ((f - m) / sd).view(B, N_FR * 3, *x.shape[-2:])


# -- model --------------------------------------------------------------------------------

class _Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class MultiFrameHeatmapNet(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        r = getattr(tvm, backbone)(weights="IMAGENET1K_V1" if pretrained else None)
        w = r.conv1.weight.data                                    # (64, 3, 7, 7), RGB
        conv1 = nn.Conv2d(3 * N_FR, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv1.weight.copy_(torch.cat([w] * N_FR, 1) / N_FR)
        self.stem = nn.Sequential(conv1, r.bn1, r.relu)             # stride 2
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        c = [64, 64, 128, 256, 512] if backbone in ("resnet18", "resnet34") else [64, 256, 512, 1024, 2048]
        self.up4 = _Up(c[4], c[3], 256)
        self.up3 = _Up(256, c[2], 128)
        self.up2 = _Up(128, c[1], 96)
        self.up1 = _Up(96, c[0], 64)                                # stride 2
        self.hm = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))
        self.off = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 2, 1))
        nn.init.constant_(self.hm[-1].bias, -4.6)                  # sigmoid ~0.01: a full frame is mostly empty

    def forward(self, x):
        s2 = self.stem(x)
        s4 = self.l1(self.pool(s2))
        s8 = self.l2(s4)
        s16 = self.l3(s8)
        s32 = self.l4(s16)
        d = self.up1(self.up2(self.up3(self.up4(s32, s16), s8), s4), s2)
        return self.hm(d), self.off(d)


def targets(pts: torch.Tensor, hh: int, ww: int):
    """pts (B, K, 3) px (x, y, size), size <= 0 unused -> heatmap (B,1,hh,ww), index (B,K), offset (B,K,2), mask."""
    valid = pts[..., 2] > 0
    c = pts[..., :2] / STRIDE
    cx, cy = c[..., 0].floor().clamp(0, ww - 1), c[..., 1].floor().clamp(0, hh - 1)
    off = torch.stack([c[..., 0] - cx, c[..., 1] - cy], -1)
    sigma = (pts[..., 2] / STRIDE / 4).clamp(min=1.5)
    ys = torch.arange(hh, device=pts.device, dtype=torch.float32).view(1, 1, hh, 1)
    xs = torch.arange(ww, device=pts.device, dtype=torch.float32).view(1, 1, 1, ww)
    d2 = (xs - cx[..., None, None]) ** 2 + (ys - cy[..., None, None]) ** 2
    g = torch.exp(-d2 / (2 * sigma[..., None, None] ** 2)) * valid[..., None, None]
    return g.amax(1, keepdim=True), (cy * ww + cx).long(), off, valid


def focal_loss(logits, gt):
    p = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    pos_l = torch.log(p) * (1 - p) ** 2 * pos
    neg_l = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * (1 - pos)
    return -(pos_l.sum() + neg_l.sum()) / pos.sum().clamp(min=1)


def offset_loss(off, ind, off_gt, mask):
    B, _, hh, ww = off.shape
    pred = off.float().view(B, 2, hh * ww).gather(2, ind.clamp(0, hh * ww - 1)[:, None].expand(-1, 2, -1))
    m = mask[..., None].float()
    return (F.l1_loss(pred.permute(0, 2, 1), off_gt, reduction="none") * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def peaks(logits, off, k: int, min_conf: float, nms: int = 7):
    """Top-k local maxima per image -> list of [(x, y, conf)] in input pixels, best first."""
    p = logits.float().sigmoid()
    keep = p == F.max_pool2d(p, nms, 1, nms // 2)
    p = (p * keep).flatten(1)
    conf, idx = p.topk(k, 1)
    ww = logits.shape[-1]
    o = off.float().flatten(2)
    out = []
    for b in range(p.shape[0]):
        row = []
        for c, i in zip(conf[b].tolist(), idx[b].tolist()):
            if c < min_conf:
                break
            row.append(((i % ww + o[b, 0, i].item()) * STRIDE, (i // ww + o[b, 1, i].item()) * STRIDE, c))
        out.append(row)
    return out


# -- full-frame inference -----------------------------------------------------------------

@torch.no_grad()
def infer_seq(model, s: Seq, dev, batch: int, k: int, min_conf: float, every: int = 1,
              flip: bool = False, scale: float = 1.0) -> dict:
    """{frame: [(x, y, conf), ...]} on full frames (padded to a multiple of 32), every `every`-th frame.
    Test-time augmentation: `flip` mirrors the input horizontally, `scale` resizes it; points are mapped back."""
    model.eval()
    cache: dict[int, np.ndarray] = {}

    def rgb(f):
        if f not in cache:
            cache[f] = cv2.imread(s.paths[f], cv2.IMREAD_COLOR)[:, :, ::-1]
        return cache[f]

    ks = list(range(0, len(s.frames), every))
    out = {}
    for b0 in range(0, len(ks), batch):
        kk = ks[b0:b0 + batch]
        x = np.stack([np.concatenate([rgb(f) for f in s.window(k)], 2) for k in kk])     # (B, H, W, 15)
        lo = s.frames[max(kk[0] - HALF, 0)]
        for f in [f for f in cache if f < lo]:
            del cache[f]
        x = torch.from_numpy(x).to(dev).permute(0, 3, 1, 2)
        H0, W0 = x.shape[-2:]
        if flip:
            x = x.flip(-1)
        if scale != 1.0:
            x = F.interpolate(x.float(), scale_factor=scale, mode="bilinear", align_corners=False)
        H, W = x.shape[-2:]
        ph, pw = (-H) % PAD_TO, (-W) % PAD_TO
        x = F.pad(normalise(x), (0, pw, 0, ph))
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            logits, off = model(x)
        for kidx, row in zip(kk, peaks(logits, off, k, min_conf)):
            pts = [(px / scale, py / scale, c) for px, py, c in row if px < W and py < H]
            out[s.frames[kidx]] = [((W0 - 1 - px) if flip else px, py, c) for px, py, c in pts]
    return out


def score_select(model, seqs, dev, a) -> dict:
    preds, gt = {}, {}
    for s in seqs:
        c = infer_seq(model, s, dev, a.val_batch, 1, 0.01, a.val_stride)
        preds[s.name] = {f: (r[0] if r else None) for f, r in c.items()}
        gt[s.name] = {f: [(x, y) for x, y, _ in s.balls[f]] for f in c}
    curve = [base.score(preds, gt, float(t)) for t in np.round(np.arange(0.01, 0.96, 0.01), 2)]
    ok = [r for r in curve if r["precision"] >= 0.90]
    return max(ok, key=lambda r: r["recall"]) if ok else max(curve, key=lambda r: r["f1"]) | {"below_floor": True}


# -- steps --------------------------------------------------------------------------------

def train(a) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() and a.device != "cpu" else "cpu")
    torch.manual_seed(a.seed)
    run = a.project / a.name
    run.mkdir(parents=True, exist_ok=True)
    tr = load_seqs(a.src, {"train", "pool"})
    sel = load_seqs(a.src, {"select"})
    if a.limit_seqs:
        tr, sel = tr[:a.limit_seqs], sel[:max(1, a.limit_seqs // 4)]
    ds = Crops(tr, a.stride, a.crop_w, a.crop_h, a.pos_prob, a.seed)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                    pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    model = MultiFrameHeatmapNet(a.backbone, pretrained=not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = min(len(dl), a.max_steps or len(dl))
    total = a.epochs * steps_per_epoch
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    start, best, hist = 0, -1.0, []
    if a.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        start, best, hist = ck["epoch"] + 1, ck["best"], ck["hist"]
    (run / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    print(f"train {len(ds)} centre frames / {len(tr)} seqs, select {len(sel)} seqs, {steps_per_epoch} steps/epoch, "
          f"{a.backbone}, crop {a.crop_w}x{a.crop_h}, batch {a.batch}, {dev}", flush=True)
    bad = 0
    for ep in range(start, a.epochs):
        model.train()
        t0, lh_sum, lo_sum, n = time.time(), 0.0, 0.0, 0
        for i, (x, pts) in enumerate(dl):
            if i >= steps_per_epoch:
                break
            x, pts = x.to(dev, non_blocking=True), pts.to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                logits, off = model(normalise(jitter(x)))
            hm, ind, off_gt, mask = targets(pts, *logits.shape[-2:])
            lh, lo = focal_loss(logits, hm), offset_loss(off, ind, off_gt, mask)
            loss = lh + lo
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            lh_sum, lo_sum, n = lh_sum + lh.item(), lo_sum + lo.item(), n + 1
            if i % 50 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
                print(f"  ep {ep + 1} step {i}/{steps_per_epoch} hm {lh_sum / n:.3f} off {lo_sum / n:.3f} "
                      f"{n * a.batch / (time.time() - t0):.1f} img/s {mem:.1f} GB", flush=True)
        v = score_select(model, sel, dev, a)
        h = {"epoch": ep + 1, "hm_loss": lh_sum / max(n, 1), "off_loss": lo_sum / max(n, 1),
                   "train_min": (time.time() - t0) / 60, **{f"select_{k}": v[k] for k in
                                                         ("thr", "recall", "precision", "f1", "recall_5px")}}
        hist.append(h)
        improved = v["recall"] > best and v["precision"] >= 0.90
        if improved:
            best, bad = v["recall"], 0
            torch.save({"model": model.state_dict(), "backbone": a.backbone, "epoch": ep, "select": v}, run / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": best, "hist": hist, "backbone": a.backbone}, run / "last.pt")
        (run / "history.json").write_text(json.dumps(hist, indent=1))
        print(f"epoch {ep + 1}: select recall {v['recall']:.2%} / precision {v['precision']:.2%} @ thr {v['thr']:.2f} "
              f"(R@5 {v['recall_5px']:.2%}){'  * best' if improved else ''}  [{h['train_min']:.0f} min]", flush=True)
        if a.patience and bad >= a.patience:
            print(f"early stop: no select gain in {a.patience} epochs", flush=True)
            break


def dump(a) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() and a.device != "cpu" else "cpu")
    ck = torch.load(a.weights, map_location=dev, weights_only=False)
    model = MultiFrameHeatmapNet(ck.get("backbone", a.backbone), pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    out = base.DATA_OUT / "candidates" / (a.dump_name or a.weights.parent.name)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("select", "test"):
        seqs = load_seqs(a.src, {split})
        if a.limit_seqs:
            seqs = seqs[:a.limit_seqs]
        doc = {"weights": str(a.weights), "topk": a.topk, "conf": a.dump_conf, "seqs": {}}
        for i, s in enumerate(seqs, 1):
            t0 = time.time()
            c = infer_seq(model, s, dev, a.val_batch, a.topk, a.dump_conf, flip=a.tta_flip, scale=a.tta_scale)
            doc["seqs"][s.name] = {"cands": {str(f): r for f, r in c.items()},
                                   "gt": {str(f): [(x, y) for x, y, _ in s.balls[f]] for f in s.frames}}
            print(f"  [{i}/{len(seqs)}] {s.name}: {len(s.frames) / (time.time() - t0):.1f} fps", flush=True)
        (out / f"{split}.json").write_text(json.dumps(doc))
        print(f"{split} -> {out / f'{split}.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3b teacher T1: multi-frame heatmap at native resolution")
    ap.add_argument("step", choices=["train", "dump"])
    ap.add_argument("--tta_flip", action="store_true", help="dump: mirror frames horizontally")
    ap.add_argument("--tta_scale", type=float, default=1.0, help="dump: resize frames by this factor")
    ap.add_argument("--src", type=Path, default=base.ROOT / "data" / "gsr")
    ap.add_argument("--backbone", default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--crop_w", type=int, default=960)
    ap.add_argument("--crop_h", type=int, default=544)
    ap.add_argument("--pos_prob", type=float, default=0.8)
    ap.add_argument("--stride", type=int, default=3, help="train centre frames: every n-th")
    ap.add_argument("--val_stride", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=0, help="cap steps per epoch (smoke tests)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val_batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_seqs", type=int, default=0, help="smoke tests only")
    ap.add_argument("--project", type=Path, default=base.ROOT / "runs" / "teacher")
    ap.add_argument("--name", default="t1_r34")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--dump_conf", type=float, default=0.001)
    ap.add_argument("--dump_name", default=None)
    a = ap.parse_args()
    if a.step == "train":
        train(a)
    else:
        if a.weights is None:
            raise SystemExit("dump needs --weights")
        dump(a)


if __name__ == "__main__":
    main()
