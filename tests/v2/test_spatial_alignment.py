"""Regression: V2 geo alignment uses shared VIIRS crop after FP resample."""

from __future__ import annotations

import numpy as np
import pytest

from wtp.data.v2.constants import CATEGORICAL_FP_SLOTS, FP_BAND_IDX
from wtp.data.v2.geo_align import center_crop_hw, crop_window_from_hw


def test_center_crop_matches_v1_formula():
    H, W, cs = 595, 595, 256
    r0, c0, h, w = crop_window_from_hw(H, W, cs)
    assert (r0, c0, h, w) == ((H - cs) // 2, (W - cs) // 2, cs, cs)
    arr = np.arange(H * W, dtype=np.float32).reshape(1, H, W)
    crop = center_crop_hw(arr, cs, H, W)
    assert crop.shape == (1, cs, cs)
    assert crop[0, 0, 0] == arr[0, r0, c0]


def test_categorical_slots_are_landcover_only():
    assert CATEGORICAL_FP_SLOTS == {FP_BAND_IDX.index(13)}


def test_align_firepred_identity_when_same_grid():
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from wtp.data.v2.geo_align import GeoRef, align_firepred_to_viirs

    transform = from_origin(-120.0, 45.0, 0.01, 0.01)
    ref = GeoRef(crs="EPSG:4326", transform=transform, width=64, height=64, path="x")
    fp = np.random.randn(len(FP_BAND_IDX), 64, 64).astype(np.float32)
    out = align_firepred_to_viirs(fp, ref, ref)
    assert out is fp or np.allclose(out, fp)


def test_align_firepred_resamples_coarser_grid():
    pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from wtp.data.v2.geo_align import GeoRef, align_firepred_to_viirs

    # VIIRS fine grid
    v_transform = from_origin(-120.0, 45.0, 0.01, 0.01)
    viirs = GeoRef(crs="EPSG:4326", transform=v_transform, width=40, height=40, path="v")
    # FP coarser (~1.33x)
    f_transform = from_origin(-120.0, 45.0, 0.0133, 0.0133)
    fp_ref = GeoRef(crs="EPSG:4326", transform=f_transform, width=30, height=30, path="f")
    fp = np.zeros((len(FP_BAND_IDX), 30, 30), dtype=np.float32)
    fp[:, 10:20, 10:20] = 5.0
    # land cover categorical slot: integer classes
    lc = FP_BAND_IDX.index(13)
    fp[lc] = 3.0
    fp[lc, 10:20, 10:20] = 7.0
    out = align_firepred_to_viirs(fp, fp_ref, viirs)
    assert out.shape == (len(FP_BAND_IDX), 40, 40)
    # Continuous bands should have intermediate values somewhere (bilinear)
    cont = out[0]
    assert cont.max() > 0
    # Categorical nearest: values should stay near integer classes
    uniq = set(np.unique(out[lc]).tolist())
    assert uniq <= {0.0, 3.0, 7.0} or all(abs(u - round(u)) < 1e-5 for u in uniq)
