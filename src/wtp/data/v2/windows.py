"""V2 window assembly: geo-aligned FirePred + shared VIIRS center-crop."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from wtp.data.v2.constants import CROP_SIZE, FP_BAND_IDX
from wtp.data.v2.geo_align import align_firepred_to_viirs, center_crop_hw
from wtp.data.v2.io import (
    load_ba_mask_v2,
    load_firepred_raw_with_georef,
    load_frame_sat_v2,
)
from wtp.data.v2.normalize import normalize_firepred


def build_window_tensor_v2(
    fire_dir: str,
    input_days: Sequence[str],
    ba_last: np.ndarray,
    fp_mean: np.ndarray,
    fp_std: np.ndarray,
    cs: int = CROP_SIZE,
) -> Optional[np.ndarray]:
    """Assemble (27, T, cs, cs) with FirePred resampled onto each frame's VIIRS grid."""
    ts = len(input_days)
    frames = []
    for t_idx in range(ts):
        sat, viirs_ref = load_frame_sat_v2(fire_dir, input_days[t_idx])
        H, W = viirs_ref.height, viirs_ref.width
        if H < cs or W < cs:
            return None
        sat_crop = center_crop_hw(sat, cs, H, W)

        fp_raw, fp_ref, _ = load_firepred_raw_with_georef(fire_dir, input_days[t_idx])
        if fp_raw is None or fp_ref is None:
            fp_crop = np.zeros((len(FP_BAND_IDX), cs, cs), dtype=np.float32)
        else:
            fp_raw = np.nan_to_num(fp_raw, nan=0.0, posinf=0.0, neginf=0.0)
            fp_aln = align_firepred_to_viirs(fp_raw, fp_ref, viirs_ref)
            fp_norm = normalize_firepred(fp_aln, fp_mean, fp_std)
            fp_crop = center_crop_hw(fp_norm, cs, H, W)

        if t_idx == ts - 1:
            ba_t = ba_last
        else:
            ba_t = load_ba_mask_v2(input_days[t_idx])
        if ba_t is not None:
            ba_crop = center_crop_hw(ba_t, cs, H, W)[np.newaxis]
        else:
            ba_crop = np.zeros((1, cs, cs), dtype=np.float32)

        frames.append(np.concatenate([sat_crop, fp_crop, ba_crop], axis=0))
    return np.stack(frames, axis=1).astype(np.float32)


def label_crop_v2(info: dict, row: dict, cs: int = CROP_SIZE) -> np.ndarray:
    days = info["day_files"]
    L = int(row["label_idx"])
    ba_last = load_ba_mask_v2(days[L - 1])
    ba_next = load_ba_mask_v2(days[L])
    if ba_last is None or ba_next is None:
        raise RuntimeError(f"BA missing for {row['target_key']}")
    H, W = ba_last.shape
    lab = np.clip(ba_next - ba_last, 0, 1)
    return center_crop_hw(lab, cs, H, W).astype(np.float32)


def build_x_for_row_v2(
    info: dict,
    row: dict,
    ts: int,
    fp_mean: np.ndarray,
    fp_std: np.ndarray,
) -> np.ndarray:
    days = info["day_files"]
    L = int(row["label_idx"])
    if L < ts:
        raise RuntimeError(f"TS={ts} insufficient history for {row['target_key']}")
    input_days = days[L - ts : L]
    if len(input_days) != ts:
        raise RuntimeError(f"bad input slice for {row['target_key']}")
    ba_last = load_ba_mask_v2(input_days[-1])
    if ba_last is None:
        raise RuntimeError(f"ba_last missing {row['target_key']}")
    x = build_window_tensor_v2(
        info["fire_dir"], input_days, ba_last, fp_mean, fp_std
    )
    if x is None:
        raise RuntimeError(f"build_window_tensor_v2 failed {row['target_key']}")
    if x.shape[1] != ts:
        raise RuntimeError(f"expected T={ts}, got {x.shape}")
    return x.astype(np.float32)
