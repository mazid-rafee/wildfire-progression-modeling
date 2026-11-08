"""Verified geographic alignment: resample FirePred onto the VIIRS grid.

V1 bug (GEO-MISREG): independent center-crops on VIIRS vs FirePred when grids
differ (4 train fires; ~1.33× coarser FP). V2 reprojects FP to the VIIRS
georeference before the shared 256 center-crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from wtp.data.v2.constants import CATEGORICAL_FP_SLOTS, CROP_SIZE, FP_BAND_IDX

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False


@dataclass(frozen=True)
class GeoRef:
    crs: object
    transform: object
    width: int
    height: int
    path: str

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def same_grid(self, other: "GeoRef", atol: float = 1e-12) -> bool:
        if self.width != other.width or self.height != other.height:
            return False
        if self.crs != other.crs:
            return False
        a = np.array(self.transform, dtype=np.float64)
        b = np.array(other.transform, dtype=np.float64)
        return bool(np.allclose(a, b, atol=atol, rtol=0.0))


def read_georef(path: str) -> GeoRef:
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required for V2 georeferencing")
    with rasterio.open(path) as src:
        return GeoRef(
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            path=path,
        )


def center_crop_hw(arr: np.ndarray, cs: int, H: int, W: int) -> np.ndarray:
    """Pixel center-crop using source H/W (same formula as V1)."""
    r0 = max(0, (H - cs) // 2)
    c0 = max(0, (W - cs) // 2)
    if arr.ndim == 3:
        return arr[:, r0 : r0 + cs, c0 : c0 + cs]
    if arr.ndim == 2:
        return arr[r0 : r0 + cs, c0 : c0 + cs]
    raise ValueError(f"Unexpected ndim={arr.ndim}")


def crop_window_from_hw(H: int, W: int, cs: int = CROP_SIZE) -> tuple[int, int, int, int]:
    """Return (row_off, col_off, height, width) for the shared center crop."""
    r0 = max(0, (H - cs) // 2)
    c0 = max(0, (W - cs) // 2)
    return r0, c0, cs, cs


def _resampling_for_slot(slot: int):
    if slot in CATEGORICAL_FP_SLOTS:
        return Resampling.nearest
    return Resampling.bilinear


def reproject_band_to_grid(
    src_band: np.ndarray,
    src_transform,
    src_crs,
    dst_height: int,
    dst_width: int,
    dst_transform,
    dst_crs,
    resampling,
    dst_dtype=np.float32,
) -> np.ndarray:
    """Reproject a single 2D band onto an explicit destination grid."""
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio is required for V2 reprojection")
    src = np.asarray(src_band, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f"expected 2D band, got shape {src.shape}")
    dst = np.zeros((dst_height, dst_width), dtype=dst_dtype)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=0.0,
    )
    return dst


def align_firepred_to_viirs(
    fp_raw: np.ndarray,
    fp_ref: GeoRef,
    viirs_ref: GeoRef,
    categorical_slots: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Resample FirePred (C,H,W) onto the VIIRS Day grid.

    Continuous bands → bilinear; categorical (land cover) → nearest.
    If grids already match, returns ``fp_raw`` unchanged (copy-free view ok).
    """
    if fp_raw.ndim != 3:
        raise ValueError(f"fp_raw must be (C,H,W), got {fp_raw.shape}")
    if fp_raw.shape[0] != len(FP_BAND_IDX):
        raise ValueError(
            f"expected {len(FP_BAND_IDX)} FP bands, got {fp_raw.shape[0]}"
        )
    cat = set(categorical_slots) if categorical_slots is not None else set(
        CATEGORICAL_FP_SLOTS
    )
    if fp_ref.same_grid(viirs_ref):
        return fp_raw.astype(np.float32, copy=False)

    out = np.zeros(
        (fp_raw.shape[0], viirs_ref.height, viirs_ref.width), dtype=np.float32
    )
    for slot in range(fp_raw.shape[0]):
        resampling = Resampling.nearest if slot in cat else Resampling.bilinear
        out[slot] = reproject_band_to_grid(
            fp_raw[slot],
            fp_ref.transform,
            fp_ref.crs,
            viirs_ref.height,
            viirs_ref.width,
            viirs_ref.transform,
            viirs_ref.crs,
            resampling,
        )
    return out


def crop_center_aligned(
    sat: np.ndarray,
    fp_on_viirs: np.ndarray,
    ba: Optional[np.ndarray],
    H: int,
    W: int,
    cs: int = CROP_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one shared center-crop to sat, FP (already on VIIRS), and BA."""
    if H < cs or W < cs:
        raise ValueError(f"grid {H}x{W} smaller than crop {cs}")
    sat_c = center_crop_hw(sat, cs, H, W)
    fp_c = center_crop_hw(fp_on_viirs, cs, H, W)
    if ba is None:
        ba_c = np.zeros((cs, cs), dtype=np.float32)
    else:
        ba_c = center_crop_hw(ba, cs, H, W)
    return sat_c, fp_c, ba_c


def world_center_of_crop(
    transform, H: int, W: int, cs: int = CROP_SIZE
) -> tuple[float, float]:
    """Geographic (x,y) of the center pixel of the shared crop."""
    r0, c0, _, _ = crop_window_from_hw(H, W, cs)
    # center of crop window
    row = r0 + cs / 2.0
    col = c0 + cs / 2.0
    x = transform.c + col * transform.a + row * transform.b
    y = transform.f + col * transform.d + row * transform.e
    return float(x), float(y)


def alignment_report(
    viirs_path: str, firepred_path: str, cs: int = CROP_SIZE
) -> dict:
    """Diagnostic: V1-style independent crop centers vs V2 shared VIIRS crop."""
    v = read_georef(viirs_path)
    f = read_georef(firepred_path)
    same = v.same_grid(f)
    vx, vy = world_center_of_crop(v.transform, v.height, v.width, cs)
    fx, fy = world_center_of_crop(f.transform, f.height, f.width, cs)
    # Approximate metres via equirectangular at mid-lat
    lat = 0.5 * (vy + fy)
    m_per_deg_x = 111_320.0 * np.cos(np.deg2rad(lat))
    m_per_deg_y = 110_540.0
    dx_m = (fx - vx) * m_per_deg_x
    dy_m = (fy - vy) * m_per_deg_y
    return {
        "same_grid": same,
        "viirs": {"width": v.width, "height": v.height, "crs": str(v.crs)},
        "firepred": {"width": f.width, "height": f.height, "crs": str(f.crs)},
        "v1_independent_crop_center_offset_m": {
            "dx_m": float(dx_m),
            "dy_m": float(dy_m),
            "hypot_m": float(np.hypot(dx_m, dy_m)),
        },
        "v2_strategy": "reproject_FP_to_VIIRS_then_shared_center_crop",
        "resampling": {
            "continuous": "bilinear",
            "categorical_slots": sorted(CATEGORICAL_FP_SLOTS),
            "categorical": "nearest",
        },
    }
