"""Read-only adapters that import frozen DeepFire Phase 3A/4B modules by absolute path.

Never writes under those trees. Callers must treat DATA_ROOT and checkpoints as immutable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from wtp.utils.paths import load_paths


def deepfire_root(paths: dict | None = None) -> Path:
    cfg = paths or load_paths()
    return Path(cfg["external_repos"]["deepfire_forecaster"])


def phase3a_dir(paths: dict | None = None) -> Path:
    cfg = paths or load_paths()
    return Path(cfg["phase3a"]["root"])


def phase4b_dir(paths: dict | None = None) -> Path:
    return deepfire_root(paths) / "phase4b_matched_ts_experiment"


def ensure_phase3a_on_path(paths: dict | None = None) -> Path:
    d = phase3a_dir(paths)
    s = str(d)
    if s not in sys.path:
        sys.path.insert(0, s)
    return d


def ensure_phase4b_on_path(paths: dict | None = None) -> Path:
    d = phase4b_dir(paths)
    s = str(d)
    if s not in sys.path:
        sys.path.insert(0, s)
    ensure_phase3a_on_path(paths)
    return d


def import_phase3a_modules(paths: dict | None = None) -> dict[str, Any]:
    ensure_phase3a_on_path(paths)
    from phase3a_fp_config import CFG, apply_data_root  # type: ignore
    from phase3a_fp_data import (  # type: ignore
        center_crop,
        discover_fires,
        load_ba_mask,
        load_firepred_raw,
        load_frame_sat,
        normalize_firepred,
    )
    from phase3a_fp_model import SEUNet3DPred, count_params  # type: ignore

    return {
        "CFG": CFG,
        "apply_data_root": apply_data_root,
        "center_crop": center_crop,
        "discover_fires": discover_fires,
        "load_ba_mask": load_ba_mask,
        "load_firepred_raw": load_firepred_raw,
        "load_frame_sat": load_frame_sat,
        "normalize_firepred": normalize_firepred,
        "SEUNet3DPred": SEUNet3DPred,
        "count_params": count_params,
    }


def import_phase4b_modules(paths: dict | None = None) -> dict[str, Any]:
    ensure_phase4b_on_path(paths)
    from phase4b_config import EXPECTED, MATCHED_CSV  # type: ignore
    from phase4b_matched_data import (  # type: ignore
        apply_seed42_neg_sampling,
        build_populations,
        build_x_for_row,
        fire_index_maps,
        label_crop_for_row,
        load_matched_manifest,
        order_like_discover,
    )

    return {
        "EXPECTED": EXPECTED,
        "MATCHED_CSV": MATCHED_CSV,
        "apply_seed42_neg_sampling": apply_seed42_neg_sampling,
        "build_populations": build_populations,
        "build_x_for_row": build_x_for_row,
        "fire_index_maps": fire_index_maps,
        "label_crop_for_row": label_crop_for_row,
        "load_matched_manifest": load_matched_manifest,
        "order_like_discover": order_like_discover,
    }
