"""Read-only bridge to Phase 4B matched preload helpers (no writes to DeepFire)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wtp.data.matched_loader import load_matched_context
from wtp.utils.paths import load_paths

EXPECTED_POS_WEIGHT = 112.87821151853076
EXPECTED_COUNTS = {
    "train": 983,
    "val": 135,
    "test24": 561,
    "test18": 553,
    "test16": 531,
}


def phase4b_shared_dir(paths: dict | None = None) -> Path:
    cfg = paths or load_paths()
    return Path(cfg["phase4b"]["run_dir"]) / "shared_populations"


def load_phase4b_population_keys(paths: dict | None = None) -> dict[str, list[str]]:
    shared = phase4b_shared_dir(paths)
    mapping = {
        "train": "population_train_seed42.json",
        "val": "population_val.json",
        "test24": "population_test24_ba.json",
        "test18": "population_test18_ba.json",
        "test16": "population_test16_ba.json",
    }
    out = {}
    for name, fname in mapping.items():
        obj = json.loads((shared / fname).read_text(encoding="utf-8"))
        out[name] = list(obj["target_keys"])
    return out


def verify_population_keys(ctx: dict[str, Any], paths: dict | None = None) -> dict:
    frozen = load_phase4b_population_keys(paths)
    ours = {
        "train": [r["target_key"] for r in ctx["train_rows"]],
        "val": [r["target_key"] for r in ctx["val_rows"]],
        "test24": [r["target_key"] for r in ctx["test24_rows"]],
        "test18": [r["target_key"] for r in ctx["test18_rows"]],
        "test16": [r["target_key"] for r in ctx["test16_rows"]],
    }
    report = {"ok": True, "mismatches": {}}
    for k in frozen:
        if ours[k] != frozen[k]:
            report["ok"] = False
            # diagnose set vs order
            set_eq = set(ours[k]) == set(frozen[k])
            report["mismatches"][k] = {
                "ours_n": len(ours[k]),
                "frozen_n": len(frozen[k]),
                "set_equal": set_eq,
                "order_equal": ours[k] == frozen[k],
                "first_diff": next(
                    (
                        i
                        for i, (a, b) in enumerate(zip(ours[k], frozen[k]))
                        if a != b
                    ),
                    None,
                ),
            }
        if len(ours[k]) != EXPECTED_COUNTS[k]:
            report["ok"] = False
            report["mismatches"].setdefault(k, {})["count_expected"] = EXPECTED_COUNTS[k]
    return report


def import_preload(paths: dict | None = None):
    from wtp.utils.external import ensure_phase4b_on_path

    ensure_phase4b_on_path(paths)
    from phase4b_matched_data import (  # type: ignore
        PreloadedMatchedDataset,
        preload_matched_windows,
    )

    return preload_matched_windows, PreloadedMatchedDataset


def load_fp_stats(paths: dict | None = None) -> tuple[np.ndarray, np.ndarray, Path]:
    stats_path = phase4b_shared_dir(paths) / "fp_stats_matched.npz"
    st = np.load(stats_path)
    return st["mean"].astype(np.float32), st["std"].astype(np.float32), stats_path


def build_matched_ctx(paths: dict | None = None) -> dict[str, Any]:
    cfg = paths or load_paths()
    ctx = load_matched_context(cfg)
    # Force shared Phase 4B FP stats (already done in load_matched_context)
    fp_mean, fp_std, stats_path = load_fp_stats(cfg)
    ctx["fp_mean"] = fp_mean
    ctx["fp_std"] = fp_std
    ctx["stats_path"] = stats_path
    ctx["pos_weight"] = EXPECTED_POS_WEIGHT
    return ctx
