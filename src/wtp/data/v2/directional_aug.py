"""Correct geometric augmentation for directional FirePred channels.

V1 bug (AUG-DIRECTIONAL): flips/rot90 move pixels but leave wind direction
(ch 11, 23) and aspect (ch 17) angle values unchanged.

V2: convert angles → unit vectors, apply the same linear transform as the
spatial axes, convert back. Operates on **denormalized** degrees when
``fp_mean``/``fp_std`` are provided; otherwise assumes raw degrees in ``x``.

Image convention (VIIRS GeoTIFF): +column = east, +row = south.
Geographic wind/aspect unit vector: (u_east, v_north).
"""

from __future__ import annotations

import random
from typing import Optional, Sequence, Tuple

import numpy as np

from wtp.data.v2.constants import DIRECTIONAL_MODEL_CHANNELS


def deg_to_unit_uv(degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological / aspect 'from' or 'facing' degrees → (u_east, v_north).

    Unit vector points in the compass direction indicated by ``degrees``
    (0=N, 90=E, 180=S, 270=W), matching both aspect-facing and the
    meteorological FROM azimuth used to orient wind.
    """
    rad = np.deg2rad(degrees.astype(np.float64))
    u = np.sin(rad)  # east
    v = np.cos(rad)  # north
    return u.astype(np.float32), v.astype(np.float32)


def unit_uv_to_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Inverse of deg_to_unit_uv; result in [0, 360)."""
    deg = np.rad2deg(np.arctan2(u.astype(np.float64), v.astype(np.float64)))
    deg = np.mod(deg, 360.0)
    # Signed-zero / -eps from atan2 can yield 360 after mod; snap to 0.
    deg = np.where(deg >= (360.0 - 1e-6), 0.0, deg)
    return deg.astype(np.float32)


def transform_uv(
    u: np.ndarray,
    v: np.ndarray,
    *,
    hflip: bool = False,
    vflip: bool = False,
    rot90_k: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply geographic vector transforms matching np.flip / np.rot90 on rasters.

    Spatial ops (image, +x east, +y south):
      hflip (axis=-1): u → -u
      vflip (axis=-2): v_north → -v_north
      rot90 CCW k=1:   (u, v) → (-v, u)   [and repeat]
    """
    for _ in range(int(rot90_k) % 4):
        u, v = -v, u
    if hflip:
        u = -u
    if vflip:
        v = -v
    return u, v


def remap_angle_field(
    degrees: np.ndarray,
    *,
    hflip: bool = False,
    vflip: bool = False,
    rot90_k: int = 0,
) -> np.ndarray:
    """Remap a 2D/ND angle field under the same geometric ops as the raster."""
    u, v = deg_to_unit_uv(degrees)
    u, v = transform_uv(u, v, hflip=hflip, vflip=vflip, rot90_k=rot90_k)
    return unit_uv_to_deg(u, v)


def _denorm_channel(
    x_ch: np.ndarray, mean: float, std: float
) -> np.ndarray:
    s = max(float(std), 0.1)
    return x_ch.astype(np.float32) * s + float(mean)


def _renorm_channel(
    raw: np.ndarray, mean: float, std: float
) -> np.ndarray:
    s = max(float(std), 0.1)
    out = (raw.astype(np.float32) - float(mean)) / s
    return np.clip(out, -5.0, 5.0)


def apply_directional_remap_on_tensor(
    x: np.ndarray,
    *,
    hflip: bool,
    vflip: bool,
    rot90_k: int,
    fp_mean: Optional[np.ndarray] = None,
    fp_std: Optional[np.ndarray] = None,
    channels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """In-place-safe remap of directional model channels after spatial transform.

    ``x`` layout: (C, T, H, W) or (C, H, W). Spatial flip/rot must already
    have been applied to all channels; this corrects angle *values*.
    """
    if not (hflip or vflip or (rot90_k % 4)):
        return x
    out = x
    chans = list(channels) if channels is not None else list(
        DIRECTIONAL_MODEL_CHANNELS.keys()
    )
    for ch in chans:
        meta = DIRECTIONAL_MODEL_CHANNELS[ch]
        slot = int(meta["fp_slot"])
        field = out[ch]
        if fp_mean is not None and fp_std is not None:
            raw = _denorm_channel(field, float(fp_mean[slot]), float(fp_std[slot]))
            remapped = remap_angle_field(
                raw, hflip=hflip, vflip=vflip, rot90_k=rot90_k
            )
            corrected = _renorm_channel(
                remapped, float(fp_mean[slot]), float(fp_std[slot])
            )
        else:
            corrected = remap_angle_field(
                field, hflip=hflip, vflip=vflip, rot90_k=rot90_k
            )
        out[ch] = corrected
    return out


def apply_geometric_aug(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hflip: Optional[bool] = None,
    vflip: Optional[bool] = None,
    rot90_k: Optional[int] = None,
    rng: Optional[random.Random] = None,
    fp_mean: Optional[np.ndarray] = None,
    fp_std: Optional[np.ndarray] = None,
    correct_directional: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Flip/rot90 with optional V2 directional correction.

    Returns (x, y, ops_dict).
    """
    rng = rng or random
    do_h = bool(rng.random() > 0.5) if hflip is None else bool(hflip)
    do_v = bool(rng.random() > 0.5) if vflip is None else bool(vflip)
    k = int(rng.randint(0, 3) if rot90_k is None else rot90_k) % 4

    x = x.astype(np.float32, copy=True)
    y = y.astype(np.float32, copy=True)
    if do_h:
        x = np.flip(x, axis=-1).copy()
        y = np.flip(y, axis=-1).copy()
    if do_v:
        x = np.flip(x, axis=-2).copy()
        y = np.flip(y, axis=-2).copy()
    if k:
        x = np.rot90(x, k, axes=(-2, -1)).copy()
        y = np.rot90(y, k, axes=(-2, -1) if y.ndim >= 3 else (0, 1)).copy()

    if correct_directional:
        x = apply_directional_remap_on_tensor(
            x, hflip=do_h, vflip=do_v, rot90_k=k, fp_mean=fp_mean, fp_std=fp_std
        )
    ops = {"hflip": do_h, "vflip": do_v, "rot90_k": k, "correct_directional": correct_directional}
    return x, y, ops
