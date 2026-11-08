"""Regression: matched 983/135/561 split integrity under V2 population helper."""

from __future__ import annotations

import pytest

from wtp.data.v2.constants import EXPECTED_COUNTS


def test_expected_counts_constants():
    assert EXPECTED_COUNTS == {
        "train": 983,
        "val": 135,
        "test24": 561,
        "test18": 553,
        "test16": 531,
    }


def test_matched_population_reconstructs_if_data_reachable():
    pytest.importorskip("yaml")
    from pathlib import Path

    from wtp.utils.paths import load_paths

    try:
        paths = load_paths()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"paths unavailable: {e}")
    root = Path(paths["data"]["root"])
    if not root.is_dir():
        pytest.skip("DATA_ROOT not mounted")
    shared = Path(paths["phase4b"]["run_dir"]) / "shared_populations"
    if not (shared / "population_train_seed42.json").exists():
        pytest.skip("Phase4B shared populations missing")

    from wtp.data.v2.population import verify_matched_population

    report = verify_matched_population(paths)
    assert report["ok"] is True
    assert report["counts"] == EXPECTED_COUNTS
    assert report["fire_id_disjoint"] is True
