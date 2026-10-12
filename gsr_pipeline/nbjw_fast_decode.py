"""
nbjw_fast_decode.py -- NBJW heatmap decode without the reference's per-frame overheads
(docs/path_b/plans/OPTIMIZATION_PLAN.md, Phase 1).

Drop-in replacements for four functions in nbjw_calib/utils/utils_heatmap.py, which stays
untouched so scripts/parity_pipeline.py can compare the two on the same heatmaps:

  get_keypoints_from_heatmap_batch_maxpool[_l]  -> peaks_kp / peaks_lines
  coords_to_dict                                -> coords_to_dict
  complete_keypoints                            -> complete_keypoints

Where the time went in the reference, and what changed:

  * Local-maximum pooling used a square 31x31 (keypoints) / 21x21 (lines) window at stride 1,
    about 1.8 G comparisons per frame. A max over a square window equals a row max followed
    by a column max, and max only selects values (it never rounds), so the two separable
    passes give identical output with ~15x fewer comparisons.
  * The top-k results went through Python loops, nested lists and torch.tensor(), then one
    .item() per value. Here they are one numpy array.
  * Each missing keypoint was completed by intersecting two lines fitted with
    scipy.stats.linregress, up to ~45 calls per frame, each on just 2 points. The same least-
    squares formula (linregress: cov with bias=1, slope = s_xy / s_xx) is evaluated inline.
    It may differ from scipy in the last bit; the result is rounded to whole pixels, so
    parity is checked on real frames rather than assumed.

Everything else (thresholds, the +1e-7 nudges, the pixel scale, the dict layout and key
order, and the reference's skipped last auxiliary pair) is reproduced deliberately: this is
an optimisation, not a change to the accuracy reference.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

_SCALE = 2      # heatmap px -> network-input px (heatmaps are half resolution)


def _separable_max_pool(x: torch.Tensor, kernel: int, pad: int) -> torch.Tensor:
    """Square max-pool, stride 1, as a row pass then a column pass (identical values)."""
    x = F.max_pool2d(x, (1, kernel), stride=1, padding=(0, pad))
    return F.max_pool2d(x, (kernel, 1), stride=1, padding=(pad, 0))


def _topk_peaks(heatmap: torch.Tensor, pooled: torch.Tensor, k: int) -> np.ndarray:
    """Top-k local maxima per channel -> float32 (B, C, k, 3) array of [x, y, score]."""
    b, c, _, width = heatmap.shape
    peaks = heatmap * (pooled == heatmap)
    scores, idx = torch.topk(peaks.reshape(b, c, -1), k, sorted=True)
    idx, scores = idx.cpu().numpy(), scores.cpu().numpy()
    out = np.empty((b, c, k, 3), dtype=np.float32)
    out[..., 0] = (idx % width) * _SCALE
    out[..., 1] = (idx // width) * _SCALE
    out[..., 2] = scores
    return out


def peaks_kp(heatmap: torch.Tensor, max_keypoints: int = 1, min_distance: int = 15) -> np.ndarray:
    """get_keypoints_from_heatmap_batch_maxpool, as a numpy array.

    The reference pads with 1.0 first so peaks within `min_distance` of the border never
    count as maxima; same here, then both passes run unpadded.
    """
    kernel = 2 * min_distance + 1
    padded = F.pad(heatmap, (min_distance,) * 4, mode="constant", value=1.0)
    return _topk_peaks(heatmap, _separable_max_pool(padded, kernel, 0), max_keypoints)


def peaks_lines(heatmap: torch.Tensor, max_keypoints: int = 2, min_distance: int = 10) -> np.ndarray:
    """get_keypoints_from_heatmap_batch_maxpool_l, as a numpy array (implicit -inf padding)."""
    kernel = 2 * min_distance + 1
    return _topk_peaks(heatmap, _separable_max_pool(heatmap, kernel, min_distance), max_keypoints)


def coords_to_dict(coords: np.ndarray, threshold: float) -> List[Dict]:
    """coords_to_dict for the (B, C, K, 3) arrays above (ground_plane_only=False).

    The reference compares float32 tensor values against a Python float, which torch does
    in float32; the threshold is cast the same way so borderline scores land identically.
    """
    thr = np.float32(threshold)
    vals = coords.tolist()      # float32 -> Python floats, as .item() did
    out = []
    for b in range(coords.shape[0]):
        keypoints = {}
        for c in range(coords.shape[1]):
            if coords.shape[2] == 1:
                if coords[b, c, 0, 2] > thr:
                    x, y, p = vals[b][c][0]
                    keypoints[c + 1] = {"x": x, "y": y, "p": p}
            elif coords[b, c, 0, 2] > thr and coords[b, c, 1, 2] > thr:
                (x1, y1, p1), (x2, y2, p2) = vals[b][c][0], vals[b][c][1]
                keypoints[c + 1] = {"x_1": x1, "y_1": y1, "p_1": p1,
                                    "x_2": x2, "y_2": y2, "p_2": p2}
        out.append(keypoints)
    return out


# --- keypoint completion ------------------------------------------------------------------

_LINES = ["Big rect. left bottom", "Big rect. left main", "Big rect. left top", "Big rect. right bottom",
          "Big rect. right main", "Big rect. right top", "Goal left crossbar", "Goal left post left ",
          "Goal left post right", "Goal right crossbar", "Goal right post left", "Goal right post right",
          "Middle line", "Side line bottom", "Side line left", "Side line right", "Side line top",
          "Small rect. left bottom", "Small rect. left main", "Small rect. left top", "Small rect. right bottom",
          "Small rect. right main", "Small rect. right top"]

_KEYPOINT_LINES = [['Side line top', 'Side line left'], ['Side line top', 'Middle line'],
                   ['Side line right', 'Side line top'], ['Side line left', 'Big rect. left top'],
                   ['Big rect. left top', 'Big rect. left main'], ['Big rect. right top', 'Big rect. right main'],
                   ['Side line right', 'Big rect. right top'], ['Side line left', 'Small rect. left top'],
                   ['Small rect. left top', 'Small rect. left main'], ['Small rect. right top', 'Small rect. right main'],
                   ['Side line right', 'Small rect. right top'], ['Goal left crossbar', 'Goal left post right'],
                   ['Side line left', 'Goal left post right'], ['Side line right', 'Goal right post left'],
                   ['Goal right crossbar', 'Goal right post left'], ['Goal left crossbar', 'Goal left post left '],
                   ['Side line left', 'Goal left post left '], ['Side line right', 'Goal right post right'],
                   ['Goal right crossbar', 'Goal right post right'], ['Side line left', 'Small rect. left bottom'],
                   ['Small rect. left bottom', 'Small rect. left main'], ['Small rect. right bottom', 'Small rect. right main'],
                   ['Side line right', 'Small rect. right bottom'], ['Side line left', 'Big rect. left bottom'],
                   ['Big rect. left bottom', 'Big rect. left main'], ['Big rect. right main', 'Big rect. right bottom'],
                   ['Side line right', 'Big rect. right bottom'], ['Side line left', 'Side line bottom'],
                   ['Side line bottom', 'Middle line'], ['Side line bottom', 'Side line right']]

_AUX_PAIRS = [['Small rect. left main', 'Side line top'], ['Big rect. left main', 'Side line top'],
              ['Big rect. right main', 'Side line top'], ['Small rect. right main', 'Side line top'],
              ['Small rect. left main', 'Big rect. left top'], ['Big rect. right top', 'Small rect. right main'],
              ['Small rect. left top', 'Big rect. left main'], ['Small rect. right top', 'Big rect. right main'],
              ['Small rect. left bottom', 'Big rect. left main'], ['Small rect. right bottom', 'Big rect. right main'],
              ['Small rect. left main', 'Big rect. left bottom'], ['Small rect. right main', 'Big rect. right bottom'],
              ['Small rect. left main', 'Side line bottom'], ['Big rect. left main', 'Side line bottom'],
              ['Big rect. right main', 'Side line bottom'], ['Small rect. right main', 'Side line bottom']]

# (line key, line key) per target, resolved once instead of two list.index() calls each time.
_KP_LINE_KEYS = [(_LINES.index(a) + 1, _LINES.index(b) + 1) for a, b in _KEYPOINT_LINES]
# The reference loops range(1, len(pairs)), so the last pair is never used. Kept: changing it
# would change the accuracy reference.
_AUX_LINE_KEYS = [(_LINES.index(a) + 1, _LINES.index(b) + 1) for a, b in _AUX_PAIRS[:-1]]


def _fit(x0: float, x1: float, y0: float, y1: float):
    """scipy.stats.linregress on two points: slope = cov_xy / var_x (bias=1)."""
    xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx0, dx1, dy0, dy1 = x0 - xm, x1 - xm, y0 - ym, y1 - ym
    ssxm = (dx0 * dx0 + dx1 * dx1) / 2.0
    ssxym = (dx0 * dy0 + dx1 * dy1) / 2.0
    slope = ssxym / ssxm if ssxm != 0.0 else np.nan
    return slope, ym - slope * xm


def _intersection(l1: Dict, l2: Dict):
    # The reference nudges the second x of each line by 1e-7 so a vertical line still fits.
    s1, i1 = _fit(l1["x_1"], l1["x_2"] + 1e-7, l1["y_1"], l1["y_2"])
    s2, i2 = _fit(l2["x_1"], l2["x_2"] + 1e-7, l2["y_1"], l2["y_2"])
    x = (i2 - i1) / (s1 - s2 + 1e-7)
    return x, s1 * x + i1


def complete_keypoints(kp_dict: List[Dict], lines_dict: List[Dict], w: int, h: int,
                       normalize: bool = False) -> List[Dict]:
    """complete_keypoints: fill missing keypoints with line intersections, same rules."""
    w_extra, h_extra = 0.5 * w, 0.5 * h

    def inside(p):
        return -w_extra < p[0] < w_extra + w and -h_extra < p[1] < h_extra + h

    out = []
    for kps, lines in zip(kp_dict, lines_dict):
        done = copy.deepcopy(kps)
        for key, (a, b) in enumerate(_KP_LINE_KEYS, start=1):
            if key not in kps and a in lines and b in lines:
                p = _intersection(lines[a], lines[b])
                if inside(p):
                    done[key] = {"x": round(p[0], 0), "y": round(p[1], 0), "p": 1.}
        for key, (a, b) in enumerate(_AUX_LINE_KEYS, start=1):
            if a in lines and b in lines:
                p = _intersection(lines[a], lines[b])
                if inside(p):
                    done[key + 57] = {"x": round(p[0], 0), "y": round(p[1], 0), "p": 1.}
        if normalize:
            for kp in done.values():
                kp["x"] /= w
                kp["y"] /= h
        out.append(dict(sorted(done.items())))
    return out
