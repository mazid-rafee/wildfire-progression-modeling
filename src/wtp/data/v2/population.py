"""Verify matched 983/135/561 population is preserved under V2 preprocessing."""

from __future__ import annotations

from typing import Any, Optional

from wtp.data.phase4b_bridge import (
    EXPECTED_COUNTS,
    build_matched_ctx,
    load_phase4b_population_keys,
    verify_population_keys,
)
from wtp.data.v2.constants import CROP_SIZE, EXPECTED_COUNTS as V2_COUNTS, GEO_MISMATCH_TRAIN_FIRES
from wtp.data.v2.io import load_ba_mask_v2, load_firepred_raw_with_georef, load_frame_sat_v2
from wtp.data.v2.windows import label_crop_v2


def verify_matched_population(paths: dict | None = None) -> dict[str, Any]:
    """Rebuild matched ctx and confirm exact key membership vs frozen Phase 4B."""
    ctx = build_matched_ctx(paths)
    key_report = verify_population_keys(ctx, paths)
    frozen = load_phase4b_population_keys(paths)
    ours = {
        "train": [r["target_key"] for r in ctx["train_rows"]],
        "val": [r["target_key"] for r in ctx["val_rows"]],
        "test24": [r["target_key"] for r in ctx["test24_rows"]],
        "test18": [r["target_key"] for r in ctx["test18_rows"]],
        "test16": [r["target_key"] for r in ctx["test16_rows"]],
    }
    counts = {k: len(v) for k, v in ours.items()}
    count_ok = all(counts[k] == EXPECTED_COUNTS[k] == V2_COUNTS[k] for k in EXPECTED_COUNTS)

    # Cross-split disjointness of target keys / fire ids
    train_f = {r["fire_id"] for r in ctx["train_rows"]}
    val_f = {r["fire_id"] for r in ctx["val_rows"]}
    test_f = {r["fire_id"] for r in ctx["test24_rows"]}
    fire_disjoint = not (train_f & val_f or train_f & test_f or val_f & test_f)

    return {
        "ok": bool(key_report["ok"] and count_ok and fire_disjoint),
        "counts": counts,
        "expected": dict(EXPECTED_COUNTS),
        "key_report": key_report,
        "fire_id_disjoint": fire_disjoint,
        "n_train_fires": len(train_f),
        "n_val_fires": len(val_f),
        "n_test24_fires": len(test_f),
        "geo_mismatch_train_fires_still_in_train": sorted(
            set(GEO_MISMATCH_TRAIN_FIRES) & train_f
        ),
        "note": (
            "V2 keeps the same matched target keys; geo alignment changes pixel "
            "content for mismatched fires but not membership."
        ),
        "ctx_counts_detail": {
            "train_keys_head": ours["train"][:3],
            "val_keys_head": ours["val"][:3],
            "test24_keys_head": ours["test24"][:3],
        },
    }


def probe_v2_buildable(
    ctx: dict,
    n_per_split: int = 2,
    ts: int = 2,
) -> dict[str, Any]:
    """Spot-check that V2 window/label construction succeeds on sample rows.

    Does not require V2 fp stats — uses ctx fp_mean/std for a smoke build, or
    zeros if only checking shapes via label path.
    """
    from wtp.data.v2.windows import build_x_for_row_v2

    results = {}
    for split, key in (
        ("train", "train_rows"),
        ("val", "val_rows"),
        ("test24", "test24_rows"),
    ):
        rows = ctx[key][:n_per_split]
        split_out = []
        for r in rows:
            info = ctx["fire_by_id"][r["fire_id"]]
            y = label_crop_v2(info, r)
            # Prefer building x; may be slow — keep n_per_split small
            x = build_x_for_row_v2(
                info, r, ts, ctx["fp_mean"], ctx["fp_std"]
            )
            day = info["day_files"][int(r["label_idx"]) - 1]
            sat, vref = load_frame_sat_v2(info["fire_dir"], day)
            fp, fref, _ = load_firepred_raw_with_georef(info["fire_dir"], day)
            split_out.append(
                {
                    "target_key": r["target_key"],
                    "x_shape": list(x.shape),
                    "y_shape": list(y.shape),
                    "y_sum": float(y.sum()),
                    "viirs_hw": [vref.height, vref.width],
                    "fp_hw": None if fref is None else [fref.height, fref.width],
                    "grids_match": None
                    if fref is None
                    else bool(vref.same_grid(fref)),
                    "fire_id": r["fire_id"],
                    "in_geo_mismatch_list": r["fire_id"] in GEO_MISMATCH_TRAIN_FIRES,
                }
            )
            del sat, fp
        results[split] = split_out
    return {
        "ok": all(
            p["x_shape"] == [27, ts, CROP_SIZE, CROP_SIZE] for v in results.values() for p in v
        ),
        "probes": results,
    }


def assert_population_preserved(report: Optional[dict] = None) -> dict[str, Any]:
    report = report or verify_matched_population()
    if not report["ok"]:
        raise AssertionError(f"Matched population not preserved: {report}")
    return report
