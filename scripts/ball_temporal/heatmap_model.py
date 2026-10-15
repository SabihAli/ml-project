"""
heatmap_model.py -- the temporal ball specialist (docs/path_b/plans/BALL_TEMPORAL_PLAN.md §4-§5):
model, input construction, targets, loss and decode. Shared by training, evaluation and export.

Input: 3 frames t-2, t-1, t of one 640x640 window, as uint8 RGB (B, 9, 640, 640) in that order.
  9 channels: the three RGB crops, ImageNet-normalised.
  5 channels: RGB at t (normalised) + grey(t) - grey(t-1) and grey(t-1) - grey(t-2), scaled by 1/0.226.
Output: heatmap logits (B, 1, 320, 320) and sub-pixel offset (B, 2, 320, 320), stride 2.
A ball at crop pixel (x, y) peaks at heatmap cell (floor(x/2), floor(y/2)) with offset (x/2, y/2) - cell.

Needs torch and torchvision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 2
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DIFF_SCALE = 1 / 0.226
PRIOR = 0.1                       # initial heatmap sigmoid output, CenterNet-style


def build_input(frames: torch.Tensor, channels: int) -> torch.Tensor:
    """uint8 (B, 9, H, W) frames t-2|t-1|t RGB -> float model input with `channels` (5 or 9) channels."""
    x = frames.float() / 255
    mean = x.new_tensor(MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(STD).view(1, 3, 1, 1)
    f2, f1, f0 = x[:, 0:3], x[:, 3:6], x[:, 6:9]
    if channels == 9:
        return torch.cat([(f - mean) / std for f in (f2, f1, f0)], 1)
    if channels == 5:
        g = [f.mean(1, keepdim=True) for f in (f2, f1, f0)]
        return torch.cat([(f0 - mean) / std, (g[2] - g[1]) * DIFF_SCALE, (g[1] - g[0]) * DIFF_SCALE], 1)
    raise ValueError(f"channels must be 5 or 9, got {channels}")


class _Up(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class BallHeatmapNet(nn.Module):
    """ResNet-18 encoder (first conv widened to `channels`), U-Net decoder to a stride-2 heatmap + offset."""

    def __init__(self, channels: int = 9, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        w = r.conv1.weight.data                                   # (64, 3, 7, 7)
        conv1 = nn.Conv2d(channels, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            if channels == 9:      # each frame gets the RGB weights, /3 so activations keep their size
                conv1.weight.copy_(torch.cat([w, w, w], 1) / 3)
            else:                  # RGB at t as pretrained; diff channels see a grey image -> summed weights
                g = w.sum(1, keepdim=True)
                conv1.weight.copy_(torch.cat([w, g * 0.5, g * 0.5], 1))
        self.channels = channels
        self.stem = nn.Sequential(conv1, r.bn1, r.relu)            # stride 2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4   # 4, 8, 16, 32
        self.up4 = _Up(512, 256, 256)
        self.up3 = _Up(256, 128, 128)
        self.up2 = _Up(128, 64, 64)
        self.up1 = _Up(64, 64, 64)                                 # -> stride 2
        self.hm = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))
        self.off = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 2, 1))
        nn.init.constant_(self.hm[-1].bias, -2.19)                 # sigmoid ~ PRIOR

    def forward(self, x):
        s2 = self.stem(x)
        s4 = self.l1(self.pool(s2))
        s8 = self.l2(s4)
        s16 = self.l3(s8)
        s32 = self.l4(s16)
        d = self.up1(self.up2(self.up3(self.up4(s32, s16), s8), s4), s2)
        return self.hm(d), self.off(d)


def make_targets(points: torch.Tensor, hm_size: int):
    """points (B, K, 3) crop px (x, y, size), rows with size <= 0 unused ->
    heatmap (B, 1, S, S), peak cell index (B, K), offset (B, K, 2), mask (B, K)."""
    B, K, _ = points.shape
    dev = points.device
    valid = points[..., 2] > 0
    c = points[..., :2] / STRIDE
    cell = c.floor().clamp(0, hm_size - 1)
    off = c - cell
    sigma = (points[..., 2] / STRIDE / 4).clamp(min=1.5)
    ys = torch.arange(hm_size, device=dev, dtype=torch.float32).view(1, 1, hm_size, 1)
    xs = torch.arange(hm_size, device=dev, dtype=torch.float32).view(1, 1, 1, hm_size)
    d2 = (xs - cell[..., 0, None, None]) ** 2 + (ys - cell[..., 1, None, None]) ** 2
    g = torch.exp(-d2 / (2 * sigma[..., None, None] ** 2)) * valid[..., None, None]
    hm = g.amax(1, keepdim=True) if K else torch.zeros(B, 1, hm_size, hm_size, device=dev)
    ind = (cell[..., 1] * hm_size + cell[..., 0]).long()
    return hm, ind, off, valid


def focal_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss, normalised by the number of peaks (min 1)."""
    p = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def offset_loss(off: torch.Tensor, ind: torch.Tensor, off_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, _, S, _ = off.shape
    pred = off.float().view(B, 2, S * S).gather(2, ind.clamp(0, S * S - 1)[:, None].expand(-1, 2, -1))
    pred = pred.permute(0, 2, 1)                                   # (B, K, 2)
    m = mask[..., None].float()
    return (F.l1_loss(pred, off_gt, reduction="none") * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def decode(logits: torch.Tensor, off: torch.Tensor):
    """-> (B, 3): x, y in crop pixels and confidence of the single strongest peak."""
    B, _, S, _ = logits.shape
    p = logits.float().sigmoid().view(B, -1)
    conf, i = p.max(1)
    iy, ix = (i // S).float(), (i % S).float()
    o = off.float().view(B, 2, -1).gather(2, i[:, None, None].expand(-1, 2, 1))[..., 0]
    return torch.stack([(ix + o[:, 0]) * STRIDE, (iy + o[:, 1]) * STRIDE, conf], 1)
