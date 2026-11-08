"""Read-only GeoTIFF IO for V2 (does not write under DATA_ROOT)."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from wtp.data.v2.constants import FP_BAND_IDX, SAT_MEAN, SAT_STD
from wtp.data.v2.geo_align import GeoRef, read_georef

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False


def load_frame_sat_v2(fire_dir: str, day_path: str) -> Tuple[np.ndarray, GeoRef]:
    """Load 8 satellite channels (6 day + 2 night), z-scored with frozen SAT stats."""
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required")
    ref = read_georef(day_path)
    with rasterio.open(day_path) as src:
        bands = src.read([1, 2, 3, 4, 5, 6]).astype(np.float32)
    night_dir = os.path.join(fire_dir, "VIIRS_Night")
    night_fname = os.path.basename(day_path).replace("_VIIRS_Day", "_VIIRS_Night")
    night_path = os.path.join(night_dir, night_fname)
    H, W = bands.shape[1], bands.shape[2]
    if os.path.exists(night_path):
        with rasterio.open(night_path) as src:
            n = src.read([1, 2]).astype(np.float32)
        nb = n[:2] if n.shape[0] >= 2 else np.zeros((2, H, W), dtype=np.float32)
    else:
        nb = np.zeros((2, H, W), dtype=np.float32)
    sat = np.concatenate([bands, nb], axis=0)
    mean = np.asarray(SAT_MEAN, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(SAT_STD, dtype=np.float32).reshape(-1, 1, 1)
    sat = (sat - mean) / (std + 1e-8)
    sat = np.nan_to_num(sat, nan=0.0, posinf=0.0, neginf=0.0)
    return sat, ref


def load_firepred_raw_with_georef(
    fire_dir: str, day_path: str
) -> tuple[Optional[np.ndarray], Optional[GeoRef], Optional[str]]:
    """Return (fp[C=18], georef, path) or (None, None, None) if missing."""
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required")
    fp_dir = os.path.join(fire_dir, "FirePred")
    fp_fname = os.path.basename(day_path).replace("_VIIRS_Day", "_FirePred")
    fp_path = os.path.join(fp_dir, fp_fname)
    if not os.path.exists(fp_path):
        return None, None, None
    with rasterio.open(fp_path) as src:
        if src.count < 19:
            return None, None, None
        indexes = [i + 1 for i in FP_BAND_IDX]
        arr = src.read(indexes).astype(np.float32)
        ref = GeoRef(
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            path=fp_path,
        )
    return arr, ref, fp_path


def load_ba_mask_v2(day_path: str) -> Optional[np.ndarray]:
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio required")
    with rasterio.open(day_path) as src:
        if src.count < 8:
            return None
        band8 = src.read(8).astype(np.float32)
        return np.isfinite(band8).astype(np.float32)
