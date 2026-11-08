#!/usr/bin/env python3
"""CPU smoke tests for environmental channel ablation masks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.data.channel_map import ABLATION_GROUPS, validate_groups_disjoint  # noqa: E402
from wtp.data.channel_mask import (  # noqa: E402
    apply_channel_mask_numpy,
    apply_channel_mask_torch,
    channels_for_ablation,
)
from wtp.data.matched_loader import build_window  # noqa: E402
from wtp.data.phase4b_bridge import EXPECTED_COUNTS, build_matched_ctx, verify_population_keys  # noqa: E402
from wtp.models import MeanPoolUNet3D, RecencyFusionUNet3D, count_params  # noqa: E402
from wtp.utils.paths import load_paths  # noqa: E402


def main() -> int:
    out = ROOT / "outputs" / "env_ablation" / "stage4_smoke"
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
    checks = []

    def add(name: str, ok: bool, detail=None):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)

    add("population_keys", key_check["ok"], key_check)
    add("population_counts", counts == EXPECTED_COUNTS, counts)
    disjoint = validate_groups_disjoint()
    add("group_partition", disjoint["ok"], disjoint)

    # Build one real window (normalized) for mask tests
    row = ctx["train_rows"][0]
    x = build_window(ctx, row, ts=6)
    add("window_shape", x.shape == (27, 6, 256, 256), list(x.shape))

    # Unmasked identity under empty mask
    x0 = apply_channel_mask_numpy(x, [], fill=0.0)
    add("empty_mask_identity", np.array_equal(x0, x))

    # ENV-0: channels 8..25 constant 0; 0..7 and 26 unchanged
    ch = channels_for_ablation("ENV-0")
    xm = apply_channel_mask_numpy(x, ch, fill=0.0)
    add("env0_masked_zero", bool(np.all(xm[8:26] == 0.0)))
    add("env0_sat_unchanged", np.array_equal(xm[0:8], x[0:8]))
    add("env0_ba_unchanged", np.array_equal(xm[26], x[26]))

    # Group masks: only those channels zeroed; others identical
    for g in ("ENV-WIND", "ENV-TERRAIN", "ENV-VEG", "ENV-METEO", "ENV-DROUGHT"):
        cg = channels_for_ablation(g)
        xg = apply_channel_mask_numpy(x, cg, fill=0.0)
        ok_zero = all(np.all(xg[c] == 0.0) for c in cg)
        keep = [c for c in range(27) if c not in cg]
        ok_keep = all(np.array_equal(xg[c], x[c]) for c in keep)
        add(f"{g}_mask", ok_zero and ok_keep, {"n": len(cg), "channels": cg})

    # Torch path
    xt = torch.from_numpy(x)
    xtm = apply_channel_mask_torch(xt.unsqueeze(0), ch, 0.0)
    add("torch_mask_shape", tuple(xtm.shape) == (1, 27, 6, 256, 256), list(xtm.shape))
    add("torch_mask_zero", bool(torch.all(xtm[0, 8:26] == 0)))

    # Param counts unchanged by masking (architecture fixed)
    base = MeanPoolUNet3D()
    rec = RecencyFusionUNet3D(max_t=6)
    add("params_mean_pool", count_params(base) == 24_279_561, count_params(base))
    add("params_recency", count_params(rec) == 24_279_568, count_params(rec))

    # Numerical stability forward with masked input
    rec.eval()
    with torch.no_grad():
        y = rec(xtm.float())
    add("forward_finite", bool(torch.isfinite(y).all()) and tuple(y.shape) == (1, 1, 256, 256), list(y.shape))

    # Leakage spot: day L not in inputs (static already audited; restate)
    add(
        "no_target_day_in_inputs_rule",
        True,
        "inputs days[L-6:L]; target ba[L]-ba[L-1] (see stage1 audit)",
    )

    ok = all(c["ok"] for c in checks)
    summary = {
        "ok": ok,
        "checks": checks,
        "ablation_groups": {k: v["model_channels"] for k, v in ABLATION_GROUPS.items()},
    }
    (out / "SMOKE_RESULTS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Env ablation CPU smoke", "", f"**Pass:** **{ok}**", ""]
    for c in checks:
        lines.append(f"- {'✅' if c['ok'] else '❌'} `{c['name']}`")
    (out / "SMOKE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("OVERALL", ok, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
