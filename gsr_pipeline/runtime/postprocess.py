"""
postprocess.py -- preprocessing and post-processing shared by every backend tier.

One implementation for TensorRT, ONNX Runtime GPU and ONNX Runtime CPU
(docs/path_b/plans/IMPLEMENTATION_PLAN.md sections 3.2-3.3). The tiers therefore
differ only in how the forward pass runs, which is what lets the parity gate (section 3.5)
attribute a discrepancy to the backend rather than to arithmetic that drifted between
copies of the same logic.

Two things here have to match Ultralytics closely or the parity gate fails for reasons that
look like a model problem:

  * `preprocess` reproduces Ultralytics' LetterBox: scale by min(ratio), round with
    `round(x * r)`, pad the remainder symmetrically with 114/255 grey.
  * `postprocess` applies NMS with the TRAINING config's iou=0.7 and max_det=300 (from
    args.yaml), not detect.py's old 0.45.

The checkpoint's head is a plain `Detect` (end2end=False), so NMS is required -- see plan
section 3 on why an earlier revision wrongly called it NMS-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Ultralytics' LetterBox fill value, as a float in [0, 1] once normalised.
PAD_VALUE = 114.0 / 255.0


@dataclass(frozen=True)
class LetterboxParams:
    """How a source frame was fitted into the model's fixed input.

    `postprocess` needs these to map boxes back to frame pixels. Kept separate from the
    tensors so the live path can compute them once per stream rather than per frame.
    """

    scale: float          # source pixel -> model pixel
    pad_x: float          # left padding, in model pixels
    pad_y: float          # top padding, in model pixels
    src_h: int
    src_w: int


def letterbox_params(src_h: int, src_w: int, dst_h: int, dst_w: int) -> LetterboxParams:
    """Compute the letterbox transform Ultralytics' LetterBox would apply.

    Scale is the smaller of the two ratios so the whole frame fits; the leftover is split
    evenly between both sides. 1920x1080 -> 1280x736 gives scale 0.6667 with 8px of
    top/bottom padding, i.e. the frame becomes 1280x720 inside a 1280x736 input.
    """
    scale = min(dst_h / src_h, dst_w / src_w)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    return LetterboxParams(
        scale=scale,
        pad_x=(dst_w - new_w) / 2,
        pad_y=(dst_h - new_h) / 2,
        src_h=src_h,
        src_w=src_w,
    )


def preprocess(
    frame_bgr: np.ndarray | torch.Tensor,
    dst_h: int,
    dst_w: int,
    device: torch.device | str = "cpu",
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, LetterboxParams]:
    """BGR HWC uint8 frame -> normalised RGB NCHW float32 tensor, plus its letterbox params.

    `out`, when given, is written in place -- that is how the live path avoids allocating a
    new input buffer per frame (plan principle 4, "allocate once"). On the GPU tiers the
    caller passes the frame as a CUDA tensor and no full frame crosses the PCIe bus.

    Two resize paths, because matching Ultralytics exactly and staying on the GPU are not
    the same requirement. Measured on SNGS-021 frame 200 against Ultralytics `.pt` at
    imgsz=(736, 1280), 17 boxes both ways (2026-09-26):

        cv2.INTER_LINEAR on uint8  ->  mean IoU 1.00000, max conf delta 0.000001
        torch bilinear, no AA      ->  mean IoU 0.99709, max conf delta 0.033
        torch bilinear, antialias  ->  mean IoU 0.98840, max conf delta 0.012

    So a numpy frame takes the cv2 path and is bit-for-bit comparable with the reference;
    that is what the parity gate (plan 3.5) scores. A tensor that is already on the GPU
    takes the torch path with antialias=False, because cv2.INTER_LINEAR does NOT antialias
    when downscaling -- turning antialiasing on moves *away* from the reference, which is
    the opposite of the intuition. The residual gap on that path is uint8-rounding versus
    float arithmetic, and it is re-measured as an accuracy gate in Phase 2 when GPU
    preprocessing lands.
    """
    params_from = lambda h, w: letterbox_params(h, w, dst_h, dst_w)

    if isinstance(frame_bgr, np.ndarray):
        src_h, src_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        params = params_from(src_h, src_w)
        new_w, new_h = round(src_w * params.scale), round(src_h * params.scale)
        top, left = int(round(params.pad_y - 0.1)), int(round(params.pad_x - 0.1))

        import cv2  # already a hard dependency of the pipeline

        resized = (
            cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if (new_h, new_w) != (src_h, src_w)
            else frame_bgr
        )
        # Pad in uint8 with Ultralytics' 114 grey, exactly as LetterBox does, before any
        # float conversion -- converting first and padding with 114/255 is arithmetically
        # equal but leaves the resize on the float path, which is what loses the parity.
        canvas = np.full((dst_h, dst_w, 3), 114, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized
        chw = (
            torch.from_numpy(canvas)
            .to(device)
            .permute(2, 0, 1)
            .flip(0)
            .unsqueeze(0)
            .to(torch.float32)
            .div_(255.0)
        )
        if out is None:
            return chw, params
        out.copy_(chw)
        return out, params

    # Tensor input: stay on-device, never round-trip a full frame through host memory.
    img = frame_bgr.to(device)
    src_h, src_w = int(img.shape[0]), int(img.shape[1])
    params = params_from(src_h, src_w)

    chw = img.permute(2, 0, 1).flip(0).unsqueeze(0).to(torch.float32).div_(255.0)
    new_w, new_h = round(src_w * params.scale), round(src_h * params.scale)
    if (new_h, new_w) != (src_h, src_w):
        chw = torch.nn.functional.interpolate(
            chw, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=False
        )

    if out is None:
        out = torch.full((1, 3, dst_h, dst_w), PAD_VALUE, dtype=torch.float32, device=device)
    else:
        out.fill_(PAD_VALUE)

    top, left = int(round(params.pad_y - 0.1)), int(round(params.pad_x - 0.1))
    out[:, :, top:top + new_h, left:left + new_w] = chw
    return out, params


def postprocess(
    raw: torch.Tensor,
    params: LetterboxParams,
    conf_by_class: dict[int, float],
    iou: float,
    max_det: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one raw head output into frame-pixel detections.

    Args:
        raw: (1, 4 + nc, num_anchors) -- boxes in xywh centre form in MODEL pixel space,
            followed by per-class scores. The layout is asserted by the caller against the
            sidecar's `output.channels` rather than assumed here.
        params: the letterbox transform used for this frame.
        conf_by_class: per-class score floor, e.g. {0: 0.25, 1: 0.15}. Applied BEFORE NMS,
            which is what keeps NMS cheap enough to stay inside the section 1.4 budget.
        iou, max_det: must be the training config's 0.7 / 300 for the parity gate to mean
            anything (plan section 3.5).

    Returns:
        (boxes (N,4) ltrb float32 in frame pixels, scores (N,), class_ids (N,) int64),
        ordered by NMS (descending score). All three stay on `raw`'s device.
    """
    from torchvision.ops import batched_nms

    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"expected (1, 4 + nc, anchors); got {tuple(raw.shape)}")

    pred = raw[0].transpose(0, 1)              # (anchors, 4 + nc)
    xywh, scores_all = pred[:, :4], pred[:, 4:]
    num_classes = scores_all.shape[1]

    scores, class_ids = scores_all.max(dim=1)

    # Per-class thresholds: build the floor each anchor must clear given its own best class.
    # Classes absent from conf_by_class get +inf, i.e. they are dropped rather than silently
    # admitted at whatever the lowest threshold happens to be.
    floors = torch.full((num_classes,), float("inf"), device=raw.device, dtype=scores.dtype)
    for cls_idx, floor in conf_by_class.items():
        if 0 <= cls_idx < num_classes:
            floors[cls_idx] = floor
    keep = scores > floors[class_ids]
    if not bool(keep.any()):
        empty_boxes = torch.zeros((0, 4), device=raw.device, dtype=torch.float32)
        return (
            empty_boxes,
            torch.zeros((0,), device=raw.device, dtype=scores.dtype),
            torch.zeros((0,), device=raw.device, dtype=torch.long),
        )

    xywh, scores, class_ids = xywh[keep], scores[keep], class_ids[keep]

    # Centre form -> corner form, still in model pixels.
    half = xywh[:, 2:] / 2
    boxes = torch.cat((xywh[:, :2] - half, xywh[:, :2] + half), dim=1)

    # Class-aware NMS: a ball overlapping an athlete must not suppress either one, which a
    # class-agnostic NMS would do -- and the ball is exactly the class we cannot afford to
    # lose (blockers doc section 2).
    kept = batched_nms(boxes, scores, class_ids, iou)[:max_det]
    boxes, scores, class_ids = boxes[kept], scores[kept], class_ids[kept]

    # Model pixels -> frame pixels: undo the padding, then the scale.
    boxes = boxes.clone()
    boxes[:, [0, 2]] -= params.pad_x
    boxes[:, [1, 3]] -= params.pad_y
    boxes /= params.scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, params.src_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, params.src_h)

    return boxes.to(torch.float32), scores, class_ids
