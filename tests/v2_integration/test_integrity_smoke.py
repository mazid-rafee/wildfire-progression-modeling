"""Integrity helpers + optional DATA_ROOT-backed smoke (skips if unavailable)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wtp.protocol.integrity import (
    check_augmentation_numeric,
    check_channel_mapping,
    check_forecast_availability,
    check_train_locks,
)


def test_aug_channel_locks_cpu():
    assert check_augmentation_numeric()["ok"]
    assert check_channel_mapping()["ok"]
    assert check_train_locks()["ok"]


def test_forecast_blocker_documented():
    r = check_forecast_availability()
    assert r["forecast_availability_established"] is False
    assert r["affected_channels"] == [22, 23, 24, 25]
    assert r["conservative_alternative"]["forecast_policy"] == "mask_zero"
    assert r["blocker"]


def test_data_backed_integrity_if_available():
    pytest.importorskip("rasterio")
    from wtp.utils.paths import load_paths

    paths = load_paths()
    root = Path(paths["data"]["root"])
    stats = Path(__file__).resolve().parents[2] / "outputs" / "v2_protocol" / "fp_stats_matched_v2.npz"
    if not root.is_dir() or not stats.exists():
        pytest.skip("DATA_ROOT or V2 stats unavailable")
    from wtp.protocol.integrity import run_all_integrity_checks

    report = run_all_integrity_checks(paths)
    assert report["essential_ok"] is True
    assert report["forecast_availability_established"] is False
