#!/usr/bin/env python3
"""STEP 1 — Reproduce frozen Phase 4B TS=6 metrics with the WTP eval harness.

Read-only w.r.t. DATA_ROOT / DeepFire trees. Writes only under this project's outputs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.data.phase4b_bridge import (  # noqa: E402
    EXPECTED_COUNTS,
    EXPECTED_POS_WEIGHT,
    build_matched_ctx,
    import_preload,
    verify_population_keys,
)
from wtp.eval.ba_eval import evaluate_windows  # noqa: E402
from wtp.utils.external import import_phase3a_modules  # noqa: E402
from wtp.utils.paths import load_paths  # noqa: E402

# Frozen Phase 4B TS=6 reference (matched_seed42_parallel_preload)
REF = {
    "thr": 0.75,
    "full24": {
        "n_windows": 561,
        "f1": 0.22393416265855806,
        "iou": 0.12608438153045087,
        "precision": 0.22064106625996957,
        "recall": 0.22732704820075755,
        "tp": 41088.0,
        "fp": 145133.0,
        "fn": 139656.0,
    },
    "rule_a18": {"n_windows": 553, "f1": 0.22593886871096877},
    "manuscript16": {"n_windows": 531, "f1": 0.2298206435090034},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _close(a: float, b: float, atol: float, rtol: float = 0.0) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "phase4c_e1" / "harness_parity",
    )
    ap.add_argument(
        "--f1-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance on F1/IoU/P/R (counts must match exactly).",
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = load_paths()
    ckpt_path = Path(paths["phase4b"]["checkpoint_ts6"])
    print(f"Loading frozen TS=6 ckpt: {ckpt_path}", flush=True)
    print(f"SHA256={_sha256(ckpt_path)}", flush=True)

    ctx = build_matched_ctx(paths)
    key_check = verify_population_keys(ctx, paths)
    print("Population key check:", json.dumps(key_check, indent=2), flush=True)
    if not key_check["ok"]:
        (out_dir / "PARITY_FAIL.json").write_text(
            json.dumps({"reason": "population_keys", "detail": key_check}, indent=2),
            encoding="utf-8",
        )
        print("FAIL: population keys mismatch — STOP", flush=True)
        return 2

    for name, n in EXPECTED_COUNTS.items():
        got = {
            "train": len(ctx["train_rows"]),
            "val": len(ctx["val_rows"]),
            "test24": len(ctx["test24_rows"]),
            "test18": len(ctx["test18_rows"]),
            "test16": len(ctx["test16_rows"]),
        }[name]
        assert got == n, (name, got, n)

    # Spot-check: labels identical for TS length; last-2 frame identity sample
    preload_matched_windows, PreloadedMatchedDataset = import_preload(paths)
    spot = ctx["test24_rows"][:3]
    from wtp.data.matched_loader import build_window

    identity_ok = True
    for r in spot:
        x2 = build_window(ctx, r, ts=2)
        x6 = build_window(ctx, r, ts=6)
        if not np.array_equal(x6[:, -2:], x2):
            identity_ok = False
            break
    print(f"Spot last-2 identity (3 test windows): {identity_ok}", flush=True)

    p3 = import_phase3a_modules(paths)
    SEUNet3DPred = p3["SEUNet3DPred"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert int(ckpt.get("ts", 6)) == 6
    model = SEUNet3DPred(in_ch=27)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    use_amp = device.type == "cuda"

    print("Preloading test24 BA-matched (TS=6 float16)...", flush=True)
    test24 = preload_matched_windows(
        ctx["test24_rows"],
        ctx["fire_by_id"],
        6,
        ctx["fp_mean"],
        ctx["fp_std"],
        split_name="parity_test24",
    )
    keys18 = {r["target_key"] for r in ctx["test18_rows"]}
    keys16 = {r["target_key"] for r in ctx["test16_rows"]}
    test18 = [w for w in test24 if w["target_key"] in keys18]
    test16 = [w for w in test24 if w["target_key"] in keys16]
    assert len(test24) == 561 and len(test18) == 553 and len(test16) == 531

    thr = REF["thr"]
    print(f"Evaluating at thr={thr} on {device} amp={use_amp}", flush=True)
    res24 = evaluate_windows(
        model, test24, device, thr, use_amp, PreloadedMatchedDataset, args.batch_size
    )
    res18 = evaluate_windows(
        model, test18, device, thr, use_amp, PreloadedMatchedDataset, args.batch_size
    )
    res16 = evaluate_windows(
        model, test16, device, thr, use_amp, PreloadedMatchedDataset, args.batch_size
    )

    comparisons = {
        "full24": {
            "ours": {k: res24[k] for k in ("f1", "iou", "precision", "recall", "tp", "fp", "fn", "n_windows")},
            "ref": REF["full24"],
        },
        "rule_a18": {
            "ours": {"f1": res18["f1"], "n_windows": res18["n_windows"], "tp": res18["tp"], "fp": res18["fp"], "fn": res18["fn"]},
            "ref": REF["rule_a18"],
        },
        "manuscript16": {
            "ours": {"f1": res16["f1"], "n_windows": res16["n_windows"], "tp": res16["tp"], "fp": res16["fp"], "fn": res16["fn"]},
            "ref": REF["manuscript16"],
        },
    }

    # Counts must match exactly; continuous metrics within atol
    failures = []
    for metric in ("tp", "fp", "fn", "n_windows"):
        if res24[metric] != REF["full24"][metric]:
            failures.append(
                f"full24 {metric}: ours={res24[metric]} ref={REF['full24'][metric]}"
            )
    for metric in ("f1", "iou", "precision", "recall"):
        if not _close(res24[metric], REF["full24"][metric], args.f1_atol):
            failures.append(
                f"full24 {metric}: ours={res24[metric]} ref={REF['full24'][metric]} "
                f"diff={abs(res24[metric]-REF['full24'][metric])}"
            )
    if res18["n_windows"] != 553 or not _close(res18["f1"], REF["rule_a18"]["f1"], args.f1_atol):
        failures.append(f"rule_a18 f1/n: ours={res18['f1']}/{res18['n_windows']}")
    if res16["n_windows"] != 531 or not _close(
        res16["f1"], REF["manuscript16"]["f1"], args.f1_atol
    ):
        failures.append(f"manuscript16 f1/n: ours={res16['f1']}/{res16['n_windows']}")

    # Load frozen JSON per-fire for equality on a few fires
    frozen_eval = Path(paths["phase4b"]["run_dir"]) / "ts6" / "evaluations" / "full_test24_ba_matched.json"
    frozen = json.loads(frozen_eval.read_text(encoding="utf-8"))
    per_fire_max_df1 = 0.0
    for fid, m in frozen["per_fire"].items():
        df1 = abs(m["f1"] - res24["per_fire"][fid]["f1"])
        per_fire_max_df1 = max(per_fire_max_df1, df1)
    if per_fire_max_df1 > args.f1_atol:
        failures.append(f"per_fire max |ΔF1|={per_fire_max_df1}")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pass": len(failures) == 0,
        "failures": failures,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": _sha256(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "selected_threshold": thr,
        "pos_weight_expected": EXPECTED_POS_WEIGHT,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "population_key_check": key_check,
        "spot_last2_identity": identity_ok,
        "comparisons": comparisons,
        "per_fire_max_abs_df1_vs_frozen_json": per_fire_max_df1,
        "f1_atol": args.f1_atol,
    }
    (out_dir / "harness_parity_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )

    # Also dump full evals for audit
    for name, res in (
        ("full_test24_ba_matched", res24),
        ("test18_ba_matched", res18),
        ("test16_ba_matched", res16),
    ):
        (out_dir / f"{name}.json").write_text(
            json.dumps(res, indent=2, default=float), encoding="utf-8"
        )

    lines = [
        "# Phase 4C E1 — harness parity vs frozen Phase 4B TS=6",
        "",
        f"**Pass:** **{report['pass']}**",
        f"**Generated (UTC):** {report['generated_utc']}",
        f"**Device:** `{device}`",
        f"**Checkpoint epoch:** {ckpt.get('epoch')}",
        f"**Threshold:** {thr}",
        "",
        "## Full-24",
        "",
        f"| Metric | Ours | Frozen |",
        f"|---|---:|---:|",
        f"| n_windows | {res24['n_windows']} | {REF['full24']['n_windows']} |",
        f"| F1 | {res24['f1']:.12f} | {REF['full24']['f1']:.12f} |",
        f"| IoU | {res24['iou']:.12f} | {REF['full24']['iou']:.12f} |",
        f"| Precision | {res24['precision']:.12f} | {REF['full24']['precision']:.12f} |",
        f"| Recall | {res24['recall']:.12f} | {REF['full24']['recall']:.12f} |",
        f"| TP | {res24['tp']} | {REF['full24']['tp']} |",
        f"| FP | {res24['fp']} | {REF['full24']['fp']} |",
        f"| FN | {res24['fn']} | {REF['full24']['fn']} |",
        "",
        f"- Rule-A 18 F1: ours={res18['f1']:.12f} frozen={REF['rule_a18']['f1']:.12f} (n={res18['n_windows']})",
        f"- Manuscript 16 F1: ours={res16['f1']:.12f} frozen={REF['manuscript16']['f1']:.12f} (n={res16['n_windows']})",
        f"- Max per-fire |ΔF1|: {per_fire_max_df1}",
        "",
    ]
    if failures:
        lines += ["## Failures", ""] + [f"- {f}" for f in failures] + [""]
    (out_dir / "HARNESS_PARITY.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"pass": report["pass"], "failures": failures}, indent=2), flush=True)
    print(f"Wrote {out_dir / 'HARNESS_PARITY.md'}", flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
