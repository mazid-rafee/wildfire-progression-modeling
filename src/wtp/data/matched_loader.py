"""Matched Phase 4A/4B population helpers (read-only; no dataset duplication)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wtp.utils.external import import_phase3a_modules, import_phase4b_modules
from wtp.utils.paths import load_paths


def load_matched_context(paths: dict | None = None) -> dict[str, Any]:
    """Rebuild Phase 4B matched populations + FP stats without writing anywhere."""
    cfg_paths = paths or load_paths()
    p3 = import_phase3a_modules(cfg_paths)
    p4 = import_phase4b_modules(cfg_paths)

    CFG = p3["CFG"]
    # Set DATA_ROOT only — do NOT call apply_data_root (it mkdir's Phase 3A SAVE_DIR).
    CFG.DATA_ROOT = str(cfg_paths["data"]["root"])

    fire_by_id, split_of, train_f, val_f, test_f = p4["fire_index_maps"](
        cfg_paths["data"]["root"]
    )
    train_order = [
        f["fire_id"]
        for f in train_f
        if f["fire_id"] not in set(CFG.FP_TRAIN_EXCLUDE)
    ]
    val_order = [
        f["fire_id"] for f in val_f if f["fire_id"] not in set(CFG.FP_VAL_EXCLUDE)
    ]
    test_order = [f["fire_id"] for f in test_f]

    manifest = p4["load_matched_manifest"]()
    pops = p4["build_populations"](manifest)
    train_pre = p4["order_like_discover"](pops["train_matched_pre_neg"], train_order)
    train_pre = [r for r in train_pre if r["fire_id"] not in set(CFG.FP_TRAIN_EXCLUDE)]
    train_rows = p4["apply_seed42_neg_sampling"](train_pre, seed=CFG.SEED)
    val_rows = p4["order_like_discover"](pops["val_matched"], val_order)
    test24 = p4["order_like_discover"](pops["test24_ba_matched"], test_order)
    test18 = p4["order_like_discover"](pops["test18_ba_matched"], test_order)
    test16 = p4["order_like_discover"](pops["test16_ba_matched"], test_order)

    stats_path = Path(cfg_paths["phase4b"]["run_dir"]) / "shared_populations" / "fp_stats_matched.npz"
    st = np.load(stats_path)
    fp_mean = st["mean"].astype(np.float32)
    fp_std = st["std"].astype(np.float32)

    return {
        "CFG": CFG,
        "p3": p3,
        "p4": p4,
        "fire_by_id": fire_by_id,
        "split_of": split_of,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test24_rows": test24,
        "test18_rows": test18,
        "test16_rows": test16,
        "fp_mean": fp_mean,
        "fp_std": fp_std,
        "stats_path": stats_path,
        "expected": p4["EXPECTED"],
    }


def build_window(
    ctx: dict[str, Any], row: dict, ts: int
) -> np.ndarray:
    return ctx["p4"]["build_x_for_row"](
        ctx["fire_by_id"][row["fire_id"]], row, ts, ctx["fp_mean"], ctx["fp_std"]
    )
