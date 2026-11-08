#!/usr/bin/env python3
"""Stage 1 — verified environmental channel dictionary + leakage notes.

Writes only under outputs/env_ablation/. Does not train or mutate frozen artifacts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.data.channel_map import (  # noqa: E402
    ABLATION_GROUPS,
    FIREPRED_SOURCE_NAMES,
    FP_BAND_IDX,
    build_model_channel_table,
    validate_groups_disjoint,
)
from wtp.data.phase4b_bridge import EXPECTED_COUNTS, build_matched_ctx, verify_population_keys  # noqa: E402
from wtp.utils.paths import load_paths  # noqa: E402


def main() -> int:
    out = ROOT / "outputs" / "env_ablation" / "stage1_channel_audit"
    out.mkdir(parents=True, exist_ok=True)
    paths = load_paths()
    ctx = build_matched_ctx(paths)
    key_check = verify_population_keys(ctx, paths)
    counts = {
        "train": len(ctx["train_rows"]),
        "val": len(ctx["val_rows"]),
        "test24": len(ctx["test24_rows"]),
        "test18": len(ctx["test18_rows"]),
        "test16": len(ctx["test16_rows"]),
    }
    table = build_model_channel_table(ctx["fp_mean"], ctx["fp_std"])
    disjoint = validate_groups_disjoint()

    skipped = FIREPRED_SOURCE_NAMES[2]

    leakage = {
        "input_time_range": "days[L-TS : L] → L-TS … L-1 (end exclusive)",
        "target": "clip(ba[L] - ba[L-1], 0, 1)",
        "day_L_in_inputs": False,
        "forecast_policy_from_dataset_paper": (
            "GFS forecast bands are 24h means from forecasts issued at the end of the "
            "current day, without using future observations. For newest frame day L-1, "
            "forecast features describe anticipated conditions after L-1 (toward day L). "
            "That is intentional forecast input available at prediction time — not BA label leakage."
        ),
        "ba_channel": "cumulative BA at input day; newest = L-1 only",
        "fp_stats": "train-only matched fp_stats_matched.npz; not fit on val/test",
        "skipped_firepred_band": {
            "source_index_0based": 2,
            "publication_name": skipped["name"],
            "code_comment": "all-zeros probe band",
            "audit_finding": (
                "Publication = Total Precipitation (GRIDMET). Empirical probe: often zero "
                "but not universally — not a constant probe. Band is absent from all 27-ch models."
            ),
        },
        "land_cover_encoding": (
            "Stored as integer class ids and z-scored as a continuous channel; NOT one-hot."
        ),
        "ambiguities": [
            "DeepFire skip comment ('probe') conflicts with TS-SatFire Table 1 naming (Total Precipitation).",
            "GRIDMET wind direction uses ~[0,360]; GFS forecast wind direction rasters observed in ~[-90,90].",
            "GRIDMET temperatures appear Kelvin-like; GFS forecast temperature rasters appear °C-like.",
            "NDVI/EVI stored as large scaled integers (≈×10000), not [-1,1] floats.",
        ],
    }

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_model_channels": 27,
        "n_firepred_in_model": 18,
        "n_firepred_raw": 19,
        "fp_band_idx": FP_BAND_IDX,
        "population_counts": counts,
        "expected_counts": EXPECTED_COUNTS,
        "population_key_check": key_check,
        "channels": table,
        "ablation_groups": {
            k: {
                "title": v["title"],
                "model_channels": v["model_channels"],
                "n": len(v["model_channels"]),
                "description": v["description"],
            }
            for k, v in ABLATION_GROUPS.items()
        },
        "group_partition_check": disjoint,
        "leakage_alignment": leakage,
        "blocker": None
        if key_check["ok"] and counts == EXPECTED_COUNTS and disjoint["ok"]
        else "Population or group partition failed — stop before GPU training.",
    }

    (out / "channel_dictionary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown dictionary
    lines = [
        "# Environmental channel dictionary (verified)",
        "",
        f"**Generated (UTC):** {report['generated_utc']}",
        "",
        "## Model layout",
        "",
        "- Channels **0–5**: VIIRS Day (I1–I5, M11)",
        "- Channels **6–7**: VIIRS Night (I4, I5)",
        "- Channels **8–25**: FirePred 18-of-19 (skip source index 2)",
        "- Channel **26**: BA cumulative mask",
        "",
        f"Population keys OK: **{key_check['ok']}** | counts={counts}",
        f"Group partition OK: **{disjoint['ok']}**",
        "",
        "## All 27 model channels",
        "",
        "| Ch | Stream | Name | Group | Kind | Time-var | Norm |",
        "|---:|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(
            f"| {r['model_channel']} | {r['stream']} | {r['name']} | "
            f"{r.get('ablation_group') or '—'} | {r['kind']} | {r['time_varying']} | "
            f"{r['normalization'][:40]}… |"
        )
    lines += [
        "",
        "## Skipped FirePred source band",
        "",
        f"- Source index 2 / publication: **{skipped['name']}**",
        f"- In model: **False**",
        f"- Audit: {leakage['skipped_firepred_band']['audit_finding']}",
        "",
        "## Ablation groups (FirePred only; VIIRS+BA retained unless ENV-0 removes FP)",
        "",
    ]
    for k, v in ABLATION_GROUPS.items():
        lines.append(f"### {k} — {v['title']}")
        lines.append(f"- Channels: `{v['model_channels']}`")
        lines.append(f"- {v['description']}")
        lines.append("")
    lines += [
        "## Leakage / forecast timing",
        "",
        f"- {leakage['input_time_range']}",
        f"- {leakage['target']}",
        f"- Day L in inputs: **{leakage['day_L_in_inputs']}**",
        f"- Forecast policy: {leakage['forecast_policy_from_dataset_paper']}",
        "",
        "## Ambiguities (do not guess away)",
        "",
    ]
    for a in leakage["ambiguities"]:
        lines.append(f"- {a}")
    lines.append("")
    if report["blocker"]:
        lines += ["## BLOCKER", "", report["blocker"], ""]
    else:
        lines += ["## Gate", "", "Channel definitions verified sufficiently to proceed to CPU smoke + ENV-0.", ""]

    (out / "CHANNEL_DICTIONARY.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "LEAKAGE_AND_INPUT_AUDIT.md").write_text(
        "\n".join(
            [
                "# Input-variable and leakage audit",
                "",
                json.dumps(leakage, indent=2),
                "",
                "## Population",
                json.dumps({"counts": counts, "key_check": key_check}, indent=2),
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({"out": str(out), "blocker": report["blocker"], "disjoint": disjoint}, indent=2))
    return 0 if report["blocker"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
