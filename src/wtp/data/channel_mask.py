"""Post-normalization channel ablation masks (neutral constant fill).

Policy: after canonical preprocessing (sat z-score, FirePred z-score+clip, BA 0/1),
excluded channels are set to **0.0** for every spatial location and timestep.
For z-scored FirePred/sat channels this equals substituting the training-set
channel mean in raw space. Land-cover (categorical stored as float and z-scored)
uses the same numeric policy — it is **not** one-hot in this pipeline.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from wtp.data.channel_map import ABLATION_GROUPS


def channels_for_ablation(name: str) -> list[int]:
    if name not in ABLATION_GROUPS:
        raise KeyError(f"Unknown ablation '{name}'. Known: {sorted(ABLATION_GROUPS)}")
    return list(ABLATION_GROUPS[name]["model_channels"])


def apply_channel_mask_numpy(
    x: np.ndarray, channels: Sequence[int], fill: float = 0.0
) -> np.ndarray:
    """x: (C,T,H,W) or (C,H,W). Returns a copy with selected channels filled."""
    out = np.array(x, copy=True)
    if out.ndim == 4:
        for c in channels:
            out[c, :, :, :] = fill
    elif out.ndim == 3:
        for c in channels:
            out[c, :, :] = fill
    else:
        raise ValueError(f"expected 3D/4D x, got {out.shape}")
    return out


def apply_channel_mask_torch(
    x: torch.Tensor, channels: Sequence[int], fill: float = 0.0
) -> torch.Tensor:
    """x: (B,C,T,H,W) or (C,T,H,W). Clones and fills."""
    out = x.clone()
    if out.ndim == 5:
        for c in channels:
            out[:, c, :, :, :] = fill
    elif out.ndim == 4:
        for c in channels:
            out[c, :, :, :] = fill
    else:
        raise ValueError(f"expected 4D/5D x, got {tuple(out.shape)}")
    return out


class MaskedPreloadedDataset(torch.utils.data.Dataset):
    """Wrap PreloadedMatchedDataset windows with a fixed channel mask."""

    def __init__(self, base_dataset, channels: Sequence[int], fill: float = 0.0):
        self.base = base_dataset
        self.channels = list(channels)
        self.fill = float(fill)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        # PreloadedMatchedDataset returns (x, y) or (x, y, meta)
        if len(item) == 3:
            x, y, meta = item
            x = apply_channel_mask_torch(x, self.channels, self.fill)
            return x, y, meta
        x, y = item
        x = apply_channel_mask_torch(x, self.channels, self.fill)
        return x, y
