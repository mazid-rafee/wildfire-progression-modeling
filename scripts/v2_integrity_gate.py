#!/usr/bin/env python3
"""Run V2 integration integrity checks (CPU-only) and write gate report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Never touch GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from wtp.protocol.config import ARMS, build_resolved_config, compare_train_locks  # noqa: E402
from wtp.protocol.integrity import run_all_integrity_checks  # noqa: E402
from wtp.utils.paths import load_paths  # noqa: E402

OUT = ROOT / "outputs" / "v2_integration"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    paths = load_paths()
    print("=== V2 integrity checks (CPU) ===", flush=True)
    integrity = run_all_integrity_checks(paths)
    (out / "integrity_checks.json").write_text(
        json.dumps(integrity, indent=2, default=float), encoding="utf-8"
    )

    # Resolved configs + hashes for all arms
    resolved_table = []
    for arm_id in ARMS:
        r = build_resolved_config(arm_id)
        resolved_table.append(
            {
                "arm_id": arm_id,
                "protocol_id": r["protocol_id"],
                "preprocessing_version": r["preprocessing_version"],
                "ts_length": r["ts_length"],
                "model": r["model"]["name"],
                "config_sha256": r["config_sha256"],
                "shared_train_lock_id": r["shared_train_lock_id"],
                "forecast_policy": r["forecast_policy"],
                "threshold": {
                    "min": r["eval"]["threshold_min"],
                    "max": r["eval"]["threshold_max"],
                    "step": r["eval"]["threshold_step"],
                },
                "seed": r["train"]["seed"],
                "lr": r["train"]["lr"],
                "pos_weight": r["train"]["pos_weight"],
            }
        )
        (out / "resolved_configs" / f"{arm_id}.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "resolved_configs" / f"{arm_id}.json").write_text(
            json.dumps(r, indent=2, default=float), encoding="utf-8"
        )

    locks = compare_train_locks()
    comparison = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": resolved_table,
        "train_locks": locks,
        "v1_protocol": build_resolved_config("V2-E1", protocol_name="v1")["protocol_id"],
        "note": "V1 remains selectable via --protocol v1; default arms use v2_corrected",
    }
    (out / "config_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=float), encoding="utf-8"
    )

    # Markdown comparison table
    lines = [
        "# Exact Configuration Comparison (V2 P0 Arms)",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Arm | Model | TS | Protocol | config_sha256 | thr grid | seed | lr | pos_weight |",
        "|-----|-------|----|----------|---------------|----------|------|----|------------|",
    ]
    for a in resolved_table:
        lines.append(
            f"| {a['arm_id']} | {a['model']} | {a['ts_length']} | {a['protocol_id']} | "
            f"`{a['config_sha256'][:12]}…` | "
            f"{a['threshold']['min']}–{a['threshold']['max']}/{a['threshold']['step']} | "
            f"{a['seed']} | {a['lr']} | {a['pos_weight']} |"
        )
    lines += [
        "",
        f"Shared train lock OK: **{locks['ok']}** (`{locks['shared_train_lock_id']}`)",
        "",
        "## V1 vs V2 protocol selection",
        "",
        "| Key | Loader | Stats | Geo | Dir-aug | Thr grid |",
        "|-----|--------|-------|-----|---------|----------|",
        "| `v1` / `v1_frozen` | phase4b V1 | fp_stats_matched.npz | independent crop | no remap | 0.05–0.75 |",
        "| `v2` / `v2_corrected` | wtp.data.v2 | fp_stats_matched_v2.npz | FP→VIIRS | remap 11/17/23 | 0.01–0.99 |",
        "| `v2_mask_gfs` | wtp.data.v2 + mask | same V2 stats | same | same | 0.01–0.99 |",
        "",
    ]
    (out / "CONFIG_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"essential_ok": integrity["essential_ok"], "forecast_established": integrity["forecast_availability_established"]}, indent=2), flush=True)
    return 0 if integrity["essential_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
