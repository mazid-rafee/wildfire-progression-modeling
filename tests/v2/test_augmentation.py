"""Regression: directional channels remap under flips / rot90."""

from __future__ import annotations

import numpy as np

from wtp.data.v2.directional_aug import (
    apply_geometric_aug,
    deg_to_unit_uv,
    remap_angle_field,
    transform_uv,
    unit_uv_to_deg,
)


def test_hflip_maps_east_to_west():
    # Constant east wind (90°)
    field = np.full((8, 8), 90.0, dtype=np.float32)
    out = remap_angle_field(field, hflip=True)
    # 90 → 270
    assert np.allclose(out, 270.0, atol=1e-4)


def test_vflip_maps_north_to_south():
    field = np.full((8, 8), 0.0, dtype=np.float32)
    out = remap_angle_field(field, vflip=True)
    assert np.allclose(out, 180.0, atol=1e-4)


def test_rot90_ccw_maps_east_to_north():
    field = np.full((8, 8), 90.0, dtype=np.float32)
    out = remap_angle_field(field, rot90_k=1)
    # (u,v)=(1,0) → (-v,u)=(0,1) → north = 0°
    assert np.allclose(out, 0.0, atol=1e-4)


def test_uv_roundtrip():
    for deg in (0, 45, 90, 135, 180, 225, 270, 315):
        u, v = deg_to_unit_uv(np.array([deg], dtype=np.float32))
        back = unit_uv_to_deg(u, v)[0]
        assert abs((back - deg + 180) % 360 - 180) < 1e-3


def test_apply_geometric_aug_corrects_model_channels():
    # x: (C=27,T=1,H,W) with raw degrees in ch 11/17/23 (no norm)
    x = np.zeros((27, 1, 16, 16), dtype=np.float32)
    y = np.zeros((16, 16), dtype=np.float32)
    x[11] = 90.0  # east
    x[17] = 0.0  # north aspect
    x[23] = 90.0
    x2, y2, ops = apply_geometric_aug(
        x, y, hflip=True, vflip=False, rot90_k=0, correct_directional=True
    )
    assert ops["hflip"] is True
    assert np.allclose(x2[11], 270.0, atol=1e-3)
    assert np.allclose(x2[23], 270.0, atol=1e-3)
    # aspect north unchanged under hflip
    assert np.allclose(x2[17], 0.0, atol=1e-3)


def test_v1_bug_reproducer_angles_would_stay_wrong_without_correction():
    x = np.zeros((27, 1, 8, 8), dtype=np.float32)
    y = np.zeros((8, 8), dtype=np.float32)
    x[11] = 90.0
    x_bad, _, _ = apply_geometric_aug(
        x, y, hflip=True, vflip=False, rot90_k=0, correct_directional=False
    )
    assert np.allclose(x_bad[11], 90.0)  # V1 behaviour


def test_transform_uv_rot90_composition():
    u = np.array([1.0], dtype=np.float32)
    v = np.array([0.0], dtype=np.float32)
    u2, v2 = transform_uv(u, v, rot90_k=1)
    assert np.allclose(u2, 0.0) and np.allclose(v2, 1.0)
