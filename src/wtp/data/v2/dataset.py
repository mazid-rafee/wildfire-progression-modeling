"""V2 datasets with directional-aware augmentation."""

from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from wtp.data.v2.directional_aug import apply_geometric_aug
from wtp.data.v2.windows import build_x_for_row_v2, label_crop_v2


class PreloadedMatchedDatasetV2(Dataset):
    """In-RAM windows with V2 directional augmentation correction."""

    def __init__(
        self,
        windows: Sequence[dict],
        augment: bool = False,
        fp_mean: Optional[np.ndarray] = None,
        fp_std: Optional[np.ndarray] = None,
        correct_directional: bool = True,
        seed: Optional[int] = None,
    ):
        self.windows = list(windows)
        self.augment = augment
        self.fp_mean = fp_mean
        self.fp_std = fp_std
        self.correct_directional = correct_directional
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        x = w["x"].astype(np.float32)
        y = w["y"].astype(np.float32)
        if self.augment:
            x, y, _ = apply_geometric_aug(
                x,
                y,
                rng=self._rng,
                fp_mean=self.fp_mean,
                fp_std=self.fp_std,
                correct_directional=self.correct_directional,
            )
        meta = w.get("target_key", w["fire_id"])
        return (
            torch.from_numpy(x),
            torch.from_numpy(y).unsqueeze(0),
            meta,
        )


def preload_matched_windows_v2(
    rows: Sequence[dict],
    fire_by_id: dict,
    ts: int,
    fp_mean: np.ndarray,
    fp_std: np.ndarray,
    split_name: str = "",
    verbose: bool = True,
    max_windows: Optional[int] = None,
) -> list[dict]:
    """Build float16 windows with V2 geo-aligned preprocessing."""
    windows: list[dict] = []
    n = len(rows) if max_windows is None else min(len(rows), max_windows)
    t0 = __import__("time").time()
    for i, r in enumerate(rows[:n]):
        info = fire_by_id[r["fire_id"]]
        x = build_x_for_row_v2(info, r, ts, fp_mean, fp_std)
        y = label_crop_v2(info, r)
        windows.append(
            {
                "x": x.astype(np.float16),
                "y": y.astype(np.float16),
                "fire_id": r["fire_id"],
                "target_key": r["target_key"],
                "n_pos": float(r.get("n_burn_px") or y.sum()),
                "protocol": "V2",
            }
        )
        if verbose and ((i + 1) % 25 == 0 or i + 1 == n):
            mb = sum(w["x"].nbytes + w["y"].nbytes for w in windows) / 1e6
            print(
                f"\r  V2 preload {split_name} TS={ts}: {i+1}/{n} | {mb:.0f}MB",
                end="",
                flush=True,
            )
    if verbose:
        mb = sum(w["x"].nbytes + w["y"].nbytes for w in windows) / 1e6
        print(
            f"\n  V2 preload {split_name} TS={ts} done: {len(windows)} | "
            f"{mb:.0f}MB | {(__import__('time').time()-t0)/60:.1f} min",
            flush=True,
        )
    return windows
