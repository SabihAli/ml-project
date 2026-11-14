"""BallSpecialist._preprocess (GPU-side flip/stack) must equal the old CPU path exactly."""

import numpy as np
import pytest
import torch

from gsr_pipeline.ball_specialist import BallSpecialist


def _old_cpu_path(frames, x0, y0, c):
    crops = []
    for f in frames:
        crop = f[y0:y0 + c, x0:x0 + c, ::-1]
        if crop.shape[:2] != (c, c):
            padded = np.zeros((c, c, 3), np.uint8)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded
        crops.append(crop)
    x = np.ascontiguousarray(np.concatenate(crops, axis=2).transpose(2, 0, 1))[None]
    return torch.from_numpy(x).to(torch.float32)


def _specialist(device, crop=64):
    s = BallSpecialist.__new__(BallSpecialist)          # no artifact / runner needed
    s.crop, s.device, s._host_buf = crop, torch.device(device), None
    return s


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("x0,y0", [(10, 7), (0, 0), (100, 50), (130, 90)])   # last two pad: frame 180x150
def test_preprocess_matches_cpu_reference(device, x0, y0):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (150, 180, 3), dtype=np.uint8) for _ in range(3)]
    s = _specialist(device)
    for _ in range(2):                                   # second call reuses the host buffer
        got = s._preprocess(frames, x0, y0).cpu()
        assert torch.equal(got, _old_cpu_path(frames, x0, y0, s.crop))
