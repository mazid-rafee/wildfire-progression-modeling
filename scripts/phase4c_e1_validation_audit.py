#!/usr/bin/env python3
"""Phase 4C E1 — read-only result-validation audit.

Independently re-evaluates E1 from its saved best checkpoint, confirms matched
membership (561/553/531), re-runs frozen Phase 4B TS=2 and TS=6 through the
same WTP eval harness, extracts learned temporal weights, runs inference-only
distribution-shift diagnostics, and writes a report under outputs/.

Does NOT:
  - modify DeepFire / DATA_ROOT / frozen Phase 4B artifacts
  - overwrite the original E1 run directory
  - train a new model
  - commit or push
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.data.matched_loader import build_window  # noqa: E402
from wtp.data.phase4b_bridge import (  # noqa: E402
    EXPECTED_COUNTS,
    EXPECTED_POS_WEIGHT,
    build_matched_ctx,
    import_preload,
    load_phase4b_population_keys,
    verify_population_keys,
)
from wtp.eval.ba_eval import MetaCollate, evaluate_windows, threshold_sweep  # noqa: E402
from wtp.models import RecencyFusionUNet3D, count_params  # noqa: E402
from wtp.train.losses import PredLoss  # noqa: E402
from wtp.utils.external import import_phase3a_modules, import_phase4b_modules  # noqa: E402
from wtp.utils.paths import load_paths  # noqa: E402

E1_RUN = (
    ROOT
    / "runs"
    / "phase4c_e1_spase_recency_matched_ts6_seed42_20261102_185755"
)
DEFAULT_OUT = ROOT / "outputs" / "phase4c_e1_validation_audit"

# Headline metrics claimed by the completed E1 run (for reproduction checks).
E1_CLAIMED_FULL24 = {
    "f1": 0.562282836565145,
    "iou": 0.3910941949262073,
    "precision": 0.5925456094227448,
    "recall": 0.5349610498826775,
    "tp": 96691.0,
    "fp": 66488.0,
    "fn": 84053.0,
    "n_windows": 561,
}

# Temporal position labels for TS=6 inputs days[L-6:L] → indices L-6 … L-1.
TEMPORAL_POSITIONS = [
    {"slot": 0, "relative_day": "L-6", "role": "oldest"},
    {"slot": 1, "relative_day": "L-5", "role": "older"},
    {"slot": 2, "relative_day": "L-4", "role": "mid"},
    {"slot": 3, "relative_day": "L-3", "role": "mid"},
    {"slot": 4, "relative_day": "L-2", "role": "newer"},
    {"slot": 5, "relative_day": "L-1", "role": "newest"},
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def _close(a: float, b: float, atol: float = 1e-6, rtol: float = 0.0) -> bool:
    return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


def _metric_parity(ours: dict, frozen: dict, atol: float = 1e-6) -> dict:
    checks = {}
    ok = True
    for k in ("n_windows", "tp", "fp", "fn"):
        match = float(ours[k]) == float(frozen[k])
        checks[k] = {"ours": ours[k], "frozen": frozen[k], "exact_match": match}
        ok = ok and match
    for k in ("f1", "iou", "precision", "recall"):
        match = _close(ours[k], frozen[k], atol=atol)
        checks[k] = {
            "ours": ours[k],
            "frozen": frozen[k],
            "abs_delta": abs(float(ours[k]) - float(frozen[k])),
            "match_atol": match,
        }
        ok = ok and match
    return {"ok": ok, "checks": checks}


def load_frozen_phase4b_evals(paths: dict) -> dict:
    run = Path(paths["phase4b"]["run_dir"])
    out = {}
    for arm in ("ts2", "ts6"):
        out[arm] = {}
        for name in (
            "full_test24_ba_matched",
            "test18_ba_matched",
            "test16_ba_matched",
        ):
            p = run / arm / "evaluations" / f"{name}.json"
            out[arm][name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def audit_leakage_and_alignment(ctx: dict, paths: dict, n_spot: int = 8) -> dict:
    """Static + spot-check audit of labels, inputs, and future-target leakage."""
    p4 = import_phase4b_modules(paths)
    label_crop_for_row = p4["label_crop_for_row"]
    build_x_for_row = p4["build_x_for_row"]

    static = {
        "input_slice_rule": "days[L-ts : L]  → frames L-ts … L-1 (end-exclusive)",
        "target_rule": "y = clip(ba[L] - ba[L-1], 0, 1)  # next-day BA progression",
        "ba_channel_newest_frame": "uses ba_last at L-1 (not ba[L])",
        "future_day_L_in_inputs": False,
        "fp_stats_source": "train-only shared_populations/fp_stats_matched.npz",
        "threshold_source": "validation-only sweep (0.05…0.75); never tuned on test",
        "notes": [
            "Label uses ba_next (day L) minus ba_last (day L-1).",
            "Inputs never include day L satellite/FP/BA stacks.",
            "Newest BA channel is ba at L-1; progression target is L-1→L.",
        ],
    }

    spot_rows = list(ctx["test24_rows"][:n_spot]) + list(ctx["val_rows"][: max(2, n_spot // 4)])
    identity_ok = True
    label_checks = []
    leakage_checks = []
    for r in spot_rows:
        info = ctx["fire_by_id"][r["fire_id"]]
        days = info["day_files"]
        L = int(r["label_idx"])
        input_days_6 = days[L - 6 : L]
        input_days_2 = days[L - 2 : L]
        assert len(input_days_6) == 6 and len(input_days_2) == 2
        assert days[L] not in input_days_6
        assert days[L] not in input_days_2
        assert input_days_6[-1] == days[L - 1]
        assert input_days_6[-2:] == input_days_2

        x2 = build_window(ctx, r, ts=2)
        x6 = build_window(ctx, r, ts=6)
        if not np.array_equal(x6[:, -2:], x2):
            identity_ok = False

        y = label_crop_for_row(info, r)
        x = build_x_for_row(info, r, 6, ctx["fp_mean"], ctx["fp_std"])
        # BA channel is last channel index (sat + fp + ba); confirm newest BA != target.
        # Channel layout: sat bands + FP bands + 1 BA → BA is channel -1.
        ba_newest = x[-1, -1]  # (H,W) at newest time
        # Target should not equal newest BA mask (progression vs cumulative).
        # Soft check: if y has positives, they should be subset-ish of growth, not identical to ba_newest.
        same_as_ba = bool(np.array_equal(y.astype(np.float32), ba_newest.astype(np.float32)))
        label_checks.append(
            {
                "target_key": r["target_key"],
                "label_idx": L,
                "input_days": list(input_days_6),
                "label_day": days[L],
                "y_pos_px": float(y.sum()),
                "y_identical_to_newest_ba_channel": same_as_ba,
                "x_shape": list(x.shape),
            }
        )
        leakage_checks.append(
            {
                "target_key": r["target_key"],
                "day_L_in_inputs": False,
                "input_end_is_L_minus_1": input_days_6[-1] == days[L - 1],
            }
        )

    return {
        "static": static,
        "spot_n": len(spot_rows),
        "last2_identity_ts6_vs_ts2": identity_ok,
        "all_spot_exclude_day_L": all(c["day_L_in_inputs"] is False for c in leakage_checks),
        "any_label_identical_to_newest_ba": any(
            c["y_identical_to_newest_ba_channel"] for c in label_checks
        ),
        "label_checks": label_checks,
        "leakage_checks": leakage_checks,
        "pos_weight_locked": EXPECTED_POS_WEIGHT,
        "ctx_pos_weight": float(ctx["pos_weight"]),
        "build_x_helper": "phase4b_matched_data.build_x_for_row",
        "label_helper": "phase4b_matched_data.label_crop_for_row",
    }


def clone_windows_transform(windows: list[dict], fn) -> list[dict]:
    out = []
    for w in windows:
        nw = {
            "x": fn(np.array(w["x"], copy=True)),
            "y": w["y"],  # labels unchanged; share reference (read-only in eval)
            "fire_id": w["fire_id"],
            "target_key": w["target_key"],
            "n_pos": w.get("n_pos"),
        }
        out.append(nw)
    return out


def diag_reverse_time(x: np.ndarray) -> np.ndarray:
    # x: (C, T, H, W)
    return np.ascontiguousarray(x[:, ::-1, :, :])


def diag_shuffle_older(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = x.shape[1]
    order = np.arange(t)
    older = order[:-1].copy()
    rng.shuffle(older)
    order[:-1] = older
    return np.ascontiguousarray(x[:, order, :, :])


def diag_ablate_older(x: np.ndarray) -> np.ndarray:
    out = np.array(x, copy=True)
    out[:, :-1, :, :] = 0
    return out


def analyze_learning_curves(hist_csv: Path) -> dict:
    rows = list(csv.DictReader(hist_csv.open(encoding="utf-8")))
    for r in rows:
        for k in (
            "epoch",
            "train_loss",
            "train_f1",
            "val_loss",
            "val_f1",
            "val_iou",
            "lr",
            "epoch_sec",
        ):
            r[k] = float(r[k]) if k != "epoch" else int(float(r[k]))
        r["is_best"] = str(r["is_best"]).lower() in ("true", "1", "yes")

    best = max(rows, key=lambda r: r["val_f1"])
    last = rows[-1]
    # Late-phase: epochs 41–60
    late = [r for r in rows if r["epoch"] >= 41]
    early_peak = max((r for r in rows if r["epoch"] <= 41), key=lambda r: r["val_f1"])

    # Overfitting signals
    train_val_f1_gap_best = best["train_f1"] - best["val_f1"]
    train_val_f1_gap_last = last["train_f1"] - last["val_f1"]
    val_loss_at_best = best["val_loss"]
    min_val_loss = min(rows, key=lambda r: r["val_loss"])

    best_marks = [r["epoch"] for r in rows if r["is_best"]]

    return {
        "n_epochs": len(rows),
        "best_epoch_by_val_f1_at_0_5": best["epoch"],
        "best_val_f1": best["val_f1"],
        "best_val_iou": best["val_iou"],
        "best_val_loss": best["val_loss"],
        "best_train_f1": best["train_f1"],
        "best_train_loss": best["train_loss"],
        "best_lr": best["lr"],
        "final_epoch": last["epoch"],
        "final_val_f1": last["val_f1"],
        "final_train_f1": last["train_f1"],
        "final_val_loss": last["val_loss"],
        "checkpoint_mark_epochs": best_marks,
        "train_val_f1_gap_at_best": train_val_f1_gap_best,
        "train_val_f1_gap_at_final": train_val_f1_gap_last,
        "min_val_loss_epoch": min_val_loss["epoch"],
        "min_val_loss": min_val_loss["val_loss"],
        "val_loss_at_best_vs_min": val_loss_at_best - min_val_loss["val_loss"],
        "early_peak_epoch_le_41": early_peak["epoch"],
        "early_peak_val_f1": early_peak["val_f1"],
        "late_phase_val_f1_mean": float(np.mean([r["val_f1"] for r in late])),
        "late_phase_val_f1_std": float(np.std([r["val_f1"] for r in late])),
        "late_phase_val_loss_mean": float(np.mean([r["val_loss"] for r in late])),
        "why_best_is_58": [
            "Checkpoint selection uses validation micro-F1 at fixed thr=0.5 (not selected thr).",
            f"Epoch 58 achieved the global max val_f1={best['val_f1']:.6f} among 60 epochs.",
            f"Epoch 60 final val_f1={last['val_f1']:.6f} is lower despite higher train_f1 "
            f"({last['train_f1']:.4f} vs {best['train_f1']:.4f}), consistent with mild late overfitting.",
            f"Val loss at best ({best['val_loss']:.4f}) is above the min val loss "
            f"({min_val_loss['val_loss']:.4f} at ep {min_val_loss['epoch']}); selection is F1-driven, not loss-driven.",
            f"OneCycleLR lr at ep58 was ~{best['lr']:.3e} (near end of cosine anneal); "
            "small late improvements are typical when LR is tiny.",
            f"Train−val F1 gap at best={train_val_f1_gap_best:.4f}; at final={train_val_f1_gap_last:.4f}.",
        ],
        "per_epoch": [
            {
                "epoch": r["epoch"],
                "train_loss": r["train_loss"],
                "train_f1": r["train_f1"],
                "val_loss": r["val_loss"],
                "val_f1": r["val_f1"],
                "val_iou": r["val_iou"],
                "lr": r["lr"],
                "is_best": r["is_best"],
                "w0": float(r["w0"]),
                "w1": float(r["w1"]),
                "w2": float(r["w2"]),
                "w3": float(r["w3"]),
                "w4": float(r["w4"]),
                "w5": float(r["w5"]),
            }
            for r in rows
        ],
    }


def recommend_ablations(weights: list[float], curves: dict) -> list[dict]:
    return [
        {
            "name": "last_frame_only",
            "type": "controlled_retraining_ablation",
            "description": (
                "Train SpaSE with TS=1 (or TS=6 inputs with older frames zeroed/masked "
                "and fusion collapsed to the newest frame) on the same matched population, "
                "seed 42, identical loss/opt/threshold protocol."
            ),
            "rationale": (
                f"E1 puts ~{weights[5]*100:.1f}% mass on L-1. A last-frame-only control "
                "tests whether older frames contribute causal signal beyond the newest observation."
            ),
            "success_criterion": (
                "If last-frame-only matches E1 within sampling noise on BA-24/16, "
                "recency fusion's gain is largely newest-frame emphasis rather than multi-day history use."
            ),
        },
        {
            "name": "fixed_recency_weighting",
            "type": "controlled_retraining_ablation",
            "description": (
                "Replace learnable logits with a frozen increasing prior "
                "(e.g. softmax(c * linspace(-1,1,6)) with c fixed, or hand-set weights "
                "matching E1's final weights), train remaining SpaSE params only."
            ),
            "rationale": (
                "Separates 'recency inductive bias' from 'learned weight discovery'. "
                "E1's learned profile is strongly newest-heavy."
            ),
            "success_criterion": (
                "If fixed recency ≈ learned E1, learning the weights is unnecessary; "
                "if fixed << learned, data-driven weights matter."
            ),
        },
        {
            "name": "learned_weighting_multiseed",
            "type": "controlled_retraining_ablation",
            "description": (
                "Replicate E1 RecencyFusionUNet3D across ≥3 seeds (e.g. 42/43/44) with "
                "identical matched membership and report mean±std F1 plus weight trajectories."
            ),
            "rationale": (
                f"Current E1 is single-seed; best epoch {curves['best_epoch_by_val_f1_at_0_5']}/60 "
                "and large Δ vs Phase 4B need multi-seed confirmation before architecture expansion (E2)."
            ),
            "success_criterion": (
                "Pre-register: claim replication only if mean BA-24 F1 remains clearly above "
                "matched Phase 4B TS=2/TS=6 with non-overlapping uncertainty."
            ),
        },
        {
            "name": "uniform_mean_pool_matched_control",
            "type": "controlled_retraining_ablation",
            "description": (
                "Re-train Phase 4B-style mean-pool SpaSE TS=6 in this harness (sanity twin) "
                "to ensure E1's lift is not an accidental harness/optimization difference."
            ),
            "rationale": (
                "Phase 4B TS=6 is frozen; a same-codebase mean-pool retrain controls for "
                "any subtle train-loop drift vs DeepFire's original trainer."
            ),
            "success_criterion": (
                "Mean-pool twin should land near frozen Phase 4B TS=6; large deviation implies "
                "harness confound rather than fusion benefit."
            ),
        },
    ]


def slim_eval(res: dict) -> dict:
    return {
        k: res[k]
        for k in (
            "f1",
            "iou",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "n_windows",
            "n_fires",
        )
    }


def write_markdown_report(out_dir: Path, report: dict) -> Path:
    e1 = report["e1_reproduction"]["evaluations"]["full_test24_ba_matched"]
    claimed = report["e1_reproduction"]["claimed_full24"]
    repro_ok = report["e1_reproduction"]["full24_matches_claimed"]
    memb = report["membership"]
    leak = report["leakage_alignment"]
    w = report["learned_temporal_weights"]
    curves = report["learning_curves"]
    diag = report["inference_diagnostics"]
    p4b = report["phase4b_reeval"]

    lines = [
        "# Phase 4C E1 — Result Validation Audit",
        "",
        f"**Generated (UTC):** {report['generated_utc']}",
        f"**Output dir:** `{out_dir}`",
        f"**E1 run (read-only):** `{report['e1_run_dir']}`",
        f"**Device:** `{report['device']}`",
        f"**Physical GPU (CUDA_VISIBLE_DEVICES):** `{report.get('cuda_visible_devices')}`",
        "",
        "## Verdict",
        "",
        f"- **E1 full-24 F1 ≈ 0.562 independently reproduced:** "
        f"**{'YES' if repro_ok else 'NO'}** "
        f"(audit F1={e1['f1']:.12f}, claimed={claimed['f1']:.12f})",
        f"- **Evaluation inconsistencies found:** "
        f"**{'YES — see findings' if report['inconsistencies'] else 'NONE material'}**",
        f"- Membership 561/553/531: **{'PASS' if memb['counts_ok'] else 'FAIL'}**",
        f"- Population key byte-order match vs frozen Phase 4B: "
        f"**{'PASS' if memb['key_check']['ok'] else 'FAIL'}**",
        f"- Future-target leakage (spot audit): "
        f"**{'PASS (no day-L in inputs)' if leak['all_spot_exclude_day_L'] else 'FAIL'}**",
        f"- Phase 4B TS=6 same-harness parity: "
        f"**{'PASS' if p4b['ts6']['parity_full24']['ok'] else 'FAIL'}**",
        f"- Phase 4B TS=2 same-harness parity: "
        f"**{'PASS' if p4b['ts2']['parity_full24']['ok'] else 'FAIL'}**",
        "",
    ]
    if report["inconsistencies"]:
        lines += ["### Findings / inconsistencies", ""]
        for item in report["inconsistencies"]:
            lines.append(f"- {item}")
        lines.append("")

    lines += [
        "## 1. Independent E1 reproduction",
        "",
        f"- Checkpoint: `{report['e1_reproduction']['checkpoint']}`",
        f"- SHA256: `{report['e1_reproduction']['checkpoint_sha256']}`",
        f"- Checkpoint epoch: **{report['e1_reproduction']['checkpoint_epoch']}**",
        f"- Eval threshold used: **{report['e1_reproduction']['threshold']}** "
        f"(source: {report['e1_reproduction']['threshold_source']})",
        f"- Val re-sweep best thr: **{report['e1_reproduction']['val_resweep']['best_thr']}** "
        f"(val F1={report['e1_reproduction']['val_resweep']['best_f1']:.6f})",
        "",
        "### Full-24 metrics",
        "",
        "| Metric | Audit re-eval | Claimed E1 run | Match |",
        "|---|---:|---:|:---:|",
    ]
    for k in ("f1", "iou", "precision", "recall", "tp", "fp", "fn", "n_windows"):
        a, b = e1[k], claimed[k]
        if k in ("tp", "fp", "fn", "n_windows"):
            match = float(a) == float(b)
        else:
            match = _close(a, b)
        lines.append(f"| {k} | {a} | {b} | {'✅' if match else '❌'} |")

    lines += [
        "",
        "### All populations (audit)",
        "",
        "| Population | n | F1 | IoU | P | R | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (
        ("full_test24_ba_matched", "Full-24"),
        ("test18_ba_matched", "Rule-A 18"),
        ("test16_ba_matched", "Manuscript 16"),
    ):
        r = report["e1_reproduction"]["evaluations"][name]
        lines.append(
            f"| {label} | {r['n_windows']} | {r['f1']:.6f} | {r['iou']:.6f} | "
            f"{r['precision']:.6f} | {r['recall']:.6f} | {r['tp']} | {r['fp']} | {r['fn']} |"
        )

    lines += [
        "",
        "### Paired per-fire ΔF1 (manuscript-16)",
        "",
        "| Fire | E1 F1 | Phase4B TS6 | Phase4B TS2 | E1−TS6 | E1−TS2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["paired_per_fire_manuscript16"]:
        lines.append(
            f"| {row['fire']} | {row['e1_f1']:.4f} | {row['ts6_f1']:.4f} | "
            f"{row['ts2_f1']:.4f} | {row['e1_minus_ts6']:+.4f} | {row['e1_minus_ts2']:+.4f} |"
        )

    lines += [
        "",
        "## 2. Matched membership",
        "",
        f"- Counts: `{memb['counts']}` (expected `{EXPECTED_COUNTS}`)",
        f"- Counts OK: **{memb['counts_ok']}**",
        f"- Key check OK: **{memb['key_check']['ok']}**",
        f"- Filtered from preloaded test24: "
        f"18→{memb['filtered_from_test24']['test18']}, "
        f"16→{memb['filtered_from_test24']['test16']}",
        "",
        "## 3. Preprocessing / target alignment / leakage",
        "",
        f"- Input rule: `{leak['static']['input_slice_rule']}`",
        f"- Target rule: `{leak['static']['target_rule']}`",
        f"- Day L in inputs (static): **{leak['static']['future_day_L_in_inputs']}**",
        f"- Spot last-2 identity TS6 vs TS2: **{leak['last2_identity_ts6_vs_ts2']}**",
        f"- All spot rows exclude day L: **{leak['all_spot_exclude_day_L']}**",
        f"- Any spot label identical to newest BA channel: "
        f"**{leak['any_label_identical_to_newest_ba']}** "
        "(expected False — target is progression, not cumulative BA)",
        f"- Threshold source: `{leak['static']['threshold_source']}`",
        f"- Locked pos_weight: **{leak['pos_weight_locked']}**",
        "",
        "## 4. Phase 4B same-harness re-evaluation",
        "",
        "### TS=6",
        "",
        f"- Full-24 parity OK: **{p4b['ts6']['parity_full24']['ok']}**",
        f"- Audit F1={p4b['ts6']['evaluations']['full_test24_ba_matched']['f1']:.12f}",
        f"- Frozen F1={p4b['ts6']['frozen']['full_test24_ba_matched']['f1']:.12f}",
        "",
        "### TS=2",
        "",
        f"- Full-24 parity OK: **{p4b['ts2']['parity_full24']['ok']}**",
        f"- Audit F1={p4b['ts2']['evaluations']['full_test24_ba_matched']['f1']:.12f}",
        f"- Frozen F1={p4b['ts2']['frozen']['full_test24_ba_matched']['f1']:.12f}",
        "",
        "## 5. Learned temporal weights",
        "",
        "| Slot | Relative day | Role | Weight |",
        "|---:|---|---|---:|",
    ]
    for pos, wi in zip(TEMPORAL_POSITIONS, w["weights_oldest_to_newest"]):
        lines.append(
            f"| {pos['slot']} | {pos['relative_day']} | {pos['role']} | {wi:.8f} |"
        )
    lines += [
        "",
        f"- Sum: **{sum(w['weights_oldest_to_newest']):.8f}**",
        f"- Newest > oldest: **{w['newest_gt_oldest']}**",
        f"- Spread (max−min): **{w['spread']:.6f}**",
        f"- `recency_scale` (ckpt): **{w['recency_scale']}**",
        f"- logits (ckpt): `{w['logits']}`",
        "",
        "## 6. Inference-only diagnostics (distribution shift — NOT training ablations)",
        "",
        "> These corrupt or reorder test inputs at inference time only. "
        "They measure sensitivity of the frozen E1 checkpoint under shift; "
        "they are **not** controlled retraining ablations.",
        "",
        f"| Diagnostic | Full-24 F1 | Δ vs clean | IoU | Notes |",
        f"|---|---:|---:|---:|---|",
        f"| clean (reference) | {diag['clean']['f1']:.6f} | 0.000000 | "
        f"{diag['clean']['iou']:.6f} | unmodified test24 |",
        f"| reverse_time | {diag['reverse_time']['f1']:.6f} | "
        f"{diag['reverse_time']['f1']-diag['clean']['f1']:+.6f} | "
        f"{diag['reverse_time']['iou']:.6f} | flip T axis |",
        f"| shuffle_older_keep_newest | {diag['shuffle_older_keep_newest']['f1']:.6f} | "
        f"{diag['shuffle_older_keep_newest']['f1']-diag['clean']['f1']:+.6f} | "
        f"{diag['shuffle_older_keep_newest']['iou']:.6f} | "
        f"seed={diag['shuffle_seed']} |",
        f"| ablate_older_zero | {diag['ablate_older_zero']['f1']:.6f} | "
        f"{diag['ablate_older_zero']['f1']-diag['clean']['f1']:+.6f} | "
        f"{diag['ablate_older_zero']['iou']:.6f} | frames 0–4 → 0 |",
        "",
        "## 7. Learning curves / why best epoch = 58/60",
        "",
    ]
    for bullet in curves["why_best_is_58"]:
        lines.append(f"- {bullet}")
    lines += [
        "",
        f"- Checkpoint mark epochs: `{curves['checkpoint_mark_epochs']}`",
        f"- Min val-loss epoch: **{curves['min_val_loss_epoch']}** "
        f"(loss={curves['min_val_loss']:.4f})",
        f"- Late-phase (ep≥41) val F1 mean±std: "
        f"{curves['late_phase_val_f1_mean']:.4f}±{curves['late_phase_val_f1_std']:.4f}",
        "",
        "## 8. Recommended controlled retraining ablations (do not run yet)",
        "",
    ]
    for ab in report["recommended_retraining_ablations"]:
        lines += [
            f"### `{ab['name']}`",
            "",
            f"- **Type:** {ab['type']}",
            f"- **Description:** {ab['description']}",
            f"- **Rationale:** {ab['rationale']}",
            f"- **Success criterion:** {ab['success_criterion']}",
            "",
        ]

    lines += [
        "## 9. Scope / non-interference",
        "",
        "- No writes under DeepFire-Forecaster, TS-SatFire, or DATA_ROOT.",
        "- E1 run directory was not modified.",
        "- No new model was trained.",
        "- No git commit/push performed.",
        "",
    ]
    path = out_dir / "PHASE4C_E1_VALIDATION_AUDIT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--e1-run", type=Path, default=E1_RUN)
    ap.add_argument("--f1-atol", type=float, default=1e-6)
    ap.add_argument("--shuffle-seed", type=int, default=12345)
    ap.add_argument(
        "--skip-val-preload",
        action="store_true",
        help="Skip val preload/re-sweep; use saved selected_threshold.json only",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not str(out_dir).startswith(str(ROOT.resolve())):
        raise SystemExit(f"out-dir must be inside project: {ROOT}")

    e1_run = args.e1_run.resolve()
    ckpt_path = e1_run / "checkpoints" / "best_e1.pth"
    if not ckpt_path.is_file():
        raise SystemExit(f"Missing E1 checkpoint: {ckpt_path}")

    paths = load_paths()
    t_start = time.time()
    print("=== E1 VALIDATION AUDIT (read-only) ===", flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"ckpt={ckpt_path} sha256={_sha256(ckpt_path)}", flush=True)

    # ---- membership ----
    ctx = build_matched_ctx(paths)
    key_check = verify_population_keys(ctx, paths)
    counts = {
        "train": len(ctx["train_rows"]),
        "val": len(ctx["val_rows"]),
        "test24": len(ctx["test24_rows"]),
        "test18": len(ctx["test18_rows"]),
        "test16": len(ctx["test16_rows"]),
    }
    counts_ok = counts == EXPECTED_COUNTS and key_check["ok"]
    print(f"Membership counts={counts} key_ok={key_check['ok']}", flush=True)
    _write_json(out_dir / "membership_check.json", {"counts": counts, "key_check": key_check})

    # ---- leakage / alignment ----
    print("=== Leakage / alignment spot audit ===", flush=True)
    leak = audit_leakage_and_alignment(ctx, paths)
    _write_json(out_dir / "leakage_alignment_audit.json", leak)
    print(
        f"last2_identity={leak['last2_identity_ts6_vs_ts2']} "
        f"exclude_day_L={leak['all_spot_exclude_day_L']}",
        flush=True,
    )

    # ---- learning curves (CPU) ----
    curves = analyze_learning_curves(e1_run / "training_history.csv")
    _write_json(out_dir / "learning_curves.json", curves)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    use_amp = device.type == "cuda"
    bs = args.batch_size

    preload_matched_windows, PreloadedMatchedDataset = import_preload(paths)
    p3 = import_phase3a_modules(paths)
    SEUNet3DPred = p3["SEUNet3DPred"]

    # ---- load E1 ----
    print("=== Load E1 checkpoint ===", flush=True)
    e1_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    e1_model = RecencyFusionUNet3D(
        in_ch=27, enc=(64, 128, 256, 512), bneck=1024, max_t=6, init_recency_scale=0.0
    )
    e1_model.load_state_dict(e1_obj["model_state_dict"])
    e1_model.to(device)
    e1_model.eval()
    n_params = count_params(e1_model)
    assert n_params == 24_279_568, n_params

    weights = e1_model.fusion_weights(6).detach().cpu().tolist()
    logits = e1_model.temporal_fusion.logits.detach().cpu().tolist()
    recency_scale = float(e1_model.temporal_fusion.recency_scale.detach().cpu())
    weight_report = {
        "weights_oldest_to_newest": weights,
        "temporal_positions": TEMPORAL_POSITIONS,
        "weights_with_positions": [
            {**pos, "weight": wi} for pos, wi in zip(TEMPORAL_POSITIONS, weights)
        ],
        "newest_gt_oldest": bool(weights[5] > weights[0]),
        "spread": float(max(weights) - min(weights)),
        "sum": float(sum(weights)),
        "logits": logits,
        "recency_scale": recency_scale,
        "checkpoint_fusion_weights_field": e1_obj.get("fusion_weights"),
    }
    _write_json(out_dir / "learned_temporal_weights.json", weight_report)
    print(f"weights={weights}", flush=True)

    selected = json.loads((e1_run / "selected_threshold.json").read_text(encoding="utf-8"))
    thr_saved = float(selected["selected_threshold"])

    # ---- val preload + threshold re-sweep ----
    criterion = PredLoss(pos_weight=EXPECTED_POS_WEIGHT).to(device)
    if args.skip_val_preload:
        thr = thr_saved
        val_resweep = {
            "skipped": True,
            "best_thr": thr_saved,
            "best_f1": selected.get("val_f1_at_selected"),
        }
    else:
        print("=== Preload val (threshold re-sweep) ===", flush=True)
        val_wins = preload_matched_windows(
            ctx["val_rows"],
            ctx["fire_by_id"],
            6,
            ctx["fp_mean"],
            ctx["fp_std"],
            split_name="audit_val",
        )
        assert len(val_wins) == 135
        val_dl = DataLoader(
            PreloadedMatchedDataset(val_wins, augment=False),
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=MetaCollate(False),
        )
        print("=== Val threshold re-sweep ===", flush=True)
        sweep = threshold_sweep(e1_model, val_dl, criterion, device, use_amp)
        val_resweep = {
            "skipped": False,
            "best_thr": float(sweep["best_thr"]),
            "best_f1": float(sweep["best_f1"]),
            "rows": sweep["rows"],
        }
        thr = thr_saved  # primary reproduction uses the run's saved thr
        del val_wins, val_dl
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write_json(out_dir / "threshold_resweep.json", val_resweep)
    print(f"Using thr={thr} (saved); resweep best={val_resweep['best_thr']}", flush=True)

    # ---- preload test24 TS=6 ----
    print("=== Preload test24 TS=6 ===", flush=True)
    test24 = preload_matched_windows(
        ctx["test24_rows"],
        ctx["fire_by_id"],
        6,
        ctx["fp_mean"],
        ctx["fp_std"],
        split_name="audit_test24_ts6",
    )
    keys18 = {r["target_key"] for r in ctx["test18_rows"]}
    keys16 = {r["target_key"] for r in ctx["test16_rows"]}
    test_sets = {
        "full_test24_ba_matched": test24,
        "test18_ba_matched": [w for w in test24 if w["target_key"] in keys18],
        "test16_ba_matched": [w for w in test24 if w["target_key"] in keys16],
    }
    assert len(test_sets["full_test24_ba_matched"]) == 561
    assert len(test_sets["test18_ba_matched"]) == 553
    assert len(test_sets["test16_ba_matched"]) == 531

    membership = {
        "counts": counts,
        "counts_ok": counts_ok,
        "key_check": key_check,
        "filtered_from_test24": {
            "test18": len(test_sets["test18_ba_matched"]),
            "test16": len(test_sets["test16_ba_matched"]),
        },
        "frozen_keys_n": {k: len(v) for k, v in load_phase4b_population_keys(paths).items()},
    }

    # ---- E1 eval ----
    print("=== Re-evaluate E1 ===", flush=True)
    e1_evals = {}
    eval_dir = out_dir / "evaluations_e1"
    eval_dir.mkdir(exist_ok=True)
    for name, wins in test_sets.items():
        print(f"  E1 {name} n={len(wins)} thr={thr}", flush=True)
        res = evaluate_windows(
            e1_model, wins, device, thr, use_amp, PreloadedMatchedDataset, bs
        )
        e1_evals[name] = res
        _write_json(eval_dir / f"{name}.json", res)

    full24_match = _metric_parity(slim_eval(e1_evals["full_test24_ba_matched"]), E1_CLAIMED_FULL24, args.f1_atol)
    print(
        f"E1 full24 F1={e1_evals['full_test24_ba_matched']['f1']:.12f} "
        f"match_claimed={full24_match['ok']}",
        flush=True,
    )

    # ---- Phase 4B TS=6 re-eval on same windows ----
    print("=== Re-evaluate Phase 4B TS=6 (same harness) ===", flush=True)
    frozen = load_frozen_phase4b_evals(paths)
    ckpt6 = Path(paths["phase4b"]["checkpoint_ts6"])
    obj6 = torch.load(ckpt6, map_location="cpu", weights_only=False)
    m6 = SEUNet3DPred(in_ch=27)
    m6.load_state_dict(obj6["model_state_dict"])
    m6.to(device)
    m6.eval()
    # Phase 4B used thr=0.75 (same as E1 selected)
    thr_p4b = 0.75
    p4b6_evals = {}
    eval6_dir = out_dir / "evaluations_phase4b_ts6"
    eval6_dir.mkdir(exist_ok=True)
    for name, wins in test_sets.items():
        print(f"  P4B-TS6 {name} n={len(wins)} thr={thr_p4b}", flush=True)
        res = evaluate_windows(m6, wins, device, thr_p4b, use_amp, PreloadedMatchedDataset, bs)
        p4b6_evals[name] = res
        _write_json(eval6_dir / f"{name}.json", res)
    parity6 = {
        name: _metric_parity(slim_eval(p4b6_evals[name]), frozen["ts6"][name], args.f1_atol)
        for name in p4b6_evals
    }
    print(f"P4B TS6 full24 parity ok={parity6['full_test24_ba_matched']['ok']}", flush=True)
    del m6
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Phase 4B TS=2 re-eval ----
    print("=== Preload test24 TS=2 + re-eval Phase 4B TS=2 ===", flush=True)
    test24_ts2 = preload_matched_windows(
        ctx["test24_rows"],
        ctx["fire_by_id"],
        2,
        ctx["fp_mean"],
        ctx["fp_std"],
        split_name="audit_test24_ts2",
    )
    test_sets_ts2 = {
        "full_test24_ba_matched": test24_ts2,
        "test18_ba_matched": [w for w in test24_ts2 if w["target_key"] in keys18],
        "test16_ba_matched": [w for w in test24_ts2 if w["target_key"] in keys16],
    }
    assert len(test_sets_ts2["full_test24_ba_matched"]) == 561
    assert len(test_sets_ts2["test18_ba_matched"]) == 553
    assert len(test_sets_ts2["test16_ba_matched"]) == 531

    ckpt2 = Path(paths["phase4b"]["checkpoint_ts2"])
    obj2 = torch.load(ckpt2, map_location="cpu", weights_only=False)
    m2 = SEUNet3DPred(in_ch=27)
    m2.load_state_dict(obj2["model_state_dict"])
    m2.to(device)
    m2.eval()
    p4b2_evals = {}
    eval2_dir = out_dir / "evaluations_phase4b_ts2"
    eval2_dir.mkdir(exist_ok=True)
    for name, wins in test_sets_ts2.items():
        print(f"  P4B-TS2 {name} n={len(wins)} thr={thr_p4b}", flush=True)
        res = evaluate_windows(m2, wins, device, thr_p4b, use_amp, PreloadedMatchedDataset, bs)
        p4b2_evals[name] = res
        _write_json(eval2_dir / f"{name}.json", res)
    parity2 = {
        name: _metric_parity(slim_eval(p4b2_evals[name]), frozen["ts2"][name], args.f1_atol)
        for name in p4b2_evals
    }
    print(f"P4B TS2 full24 parity ok={parity2['full_test24_ba_matched']['ok']}", flush=True)
    del m2, test24_ts2, test_sets_ts2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- inference diagnostics on E1 ----
    print("=== Inference-only diagnostics (distribution shift) ===", flush=True)
    rng = np.random.default_rng(args.shuffle_seed)
    # Independent RNG draws per-window would differ; use one permutation for all windows
    # for a cleaner "global older-shuffle" diagnostic.
    older_perm = np.arange(5)
    rng.shuffle(older_perm)

    def shuffle_with_fixed_perm(x: np.ndarray) -> np.ndarray:
        order = np.arange(x.shape[1])
        order[:-1] = older_perm
        return np.ascontiguousarray(x[:, order, :, :])

    clean = slim_eval(e1_evals["full_test24_ba_matched"])
    rev_wins = clone_windows_transform(test24, diag_reverse_time)
    shuf_wins = clone_windows_transform(test24, shuffle_with_fixed_perm)
    abl_wins = clone_windows_transform(test24, diag_ablate_older)

    rev_res = evaluate_windows(
        e1_model, rev_wins, device, thr, use_amp, PreloadedMatchedDataset, bs
    )
    shuf_res = evaluate_windows(
        e1_model, shuf_wins, device, thr, use_amp, PreloadedMatchedDataset, bs
    )
    abl_res = evaluate_windows(
        e1_model, abl_wins, device, thr, use_amp, PreloadedMatchedDataset, bs
    )
    diagnostics = {
        "disclaimer": (
            "Inference-only distribution-shift diagnostics on a frozen checkpoint. "
            "Not controlled training ablations."
        ),
        "threshold": thr,
        "shuffle_seed": args.shuffle_seed,
        "older_permutation_slots_0_to_4": older_perm.tolist(),
        "clean": clean,
        "reverse_time": slim_eval(rev_res),
        "shuffle_older_keep_newest": slim_eval(shuf_res),
        "ablate_older_zero": slim_eval(abl_res),
    }
    _write_json(out_dir / "inference_diagnostics.json", diagnostics)
    print(
        f"diag F1 clean={clean['f1']:.4f} rev={rev_res['f1']:.4f} "
        f"shuf={shuf_res['f1']:.4f} abl={abl_res['f1']:.4f}",
        flush=True,
    )
    del rev_wins, shuf_wins, abl_wins

    # ---- paired per-fire (use audit E1 + frozen P4B tables for fair Δ vs published baselines;
    # also include audit-recomputed P4B per-fire when available) ----
    pf_e1 = e1_evals["test16_ba_matched"]["per_fire"]
    pf6 = frozen["ts6"]["test16_ba_matched"]["per_fire"]
    pf2 = frozen["ts2"]["test16_ba_matched"]["per_fire"]
    paired = []
    for fid in sorted(set(pf_e1) & set(pf6) & set(pf2)):
        a, b, c = pf_e1[fid]["f1"], pf6[fid]["f1"], pf2[fid]["f1"]
        paired.append(
            {
                "fire": fid,
                "e1_f1": a,
                "ts6_f1": b,
                "ts2_f1": c,
                "e1_minus_ts6": a - b,
                "e1_minus_ts2": a - c,
            }
        )
    _write_json(out_dir / "paired_per_fire_manuscript16.json", paired)

    inconsistencies: list[str] = []
    if not counts_ok:
        inconsistencies.append("Matched membership counts or key order failed.")
    if not leak["all_spot_exclude_day_L"]:
        inconsistencies.append("Spot audit found day-L present in inputs.")
    if not leak["last2_identity_ts6_vs_ts2"]:
        inconsistencies.append("TS=6 last-2 frames not identical to TS=2 inputs.")
    if leak["any_label_identical_to_newest_ba"]:
        inconsistencies.append(
            "Spot label identical to newest BA channel (possible progression mis-definition)."
        )
    if not full24_match["ok"]:
        inconsistencies.append(
            "Independent E1 full-24 metrics do not match the claimed run artifacts."
        )
    if not parity6["full_test24_ba_matched"]["ok"]:
        inconsistencies.append("Phase 4B TS=6 same-harness re-eval disagrees with frozen JSON.")
    if not parity2["full_test24_ba_matched"]["ok"]:
        inconsistencies.append("Phase 4B TS=2 same-harness re-eval disagrees with frozen JSON.")
    if not _close(val_resweep["best_thr"], thr_saved, atol=1e-9) and not val_resweep.get(
        "skipped"
    ):
        inconsistencies.append(
            f"Val re-sweep selected thr={val_resweep['best_thr']} "
            f"!= saved thr={thr_saved} (reported for transparency; primary eval used saved thr)."
        )

    recommendations = recommend_ablations(weights, curves)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_min": (time.time() - t_start) / 60.0,
        "e1_run_dir": str(e1_run),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "membership": membership,
        "leakage_alignment": leak,
        "learned_temporal_weights": weight_report,
        "learning_curves": {
            k: v for k, v in curves.items() if k != "per_epoch"
        },  # keep MD lean; full in learning_curves.json
        "e1_reproduction": {
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": _sha256(ckpt_path),
            "checkpoint_epoch": int(e1_obj.get("epoch", -1)),
            "n_params": n_params,
            "threshold": thr,
            "threshold_source": "selected_threshold.json from E1 run (validation-only)",
            "val_resweep": {
                "skipped": val_resweep.get("skipped", False),
                "best_thr": val_resweep["best_thr"],
                "best_f1": val_resweep["best_f1"],
            },
            "claimed_full24": E1_CLAIMED_FULL24,
            "full24_matches_claimed": full24_match["ok"],
            "full24_parity_detail": full24_match,
            "evaluations": {k: slim_eval(v) for k, v in e1_evals.items()},
        },
        "phase4b_reeval": {
            "threshold": thr_p4b,
            "ts6": {
                "checkpoint": str(ckpt6),
                "checkpoint_sha256": _sha256(ckpt6),
                "evaluations": {k: slim_eval(v) for k, v in p4b6_evals.items()},
                "frozen": {
                    k: {
                        m: frozen["ts6"][k][m]
                        for m in (
                            "f1",
                            "iou",
                            "precision",
                            "recall",
                            "tp",
                            "fp",
                            "fn",
                            "n_windows",
                        )
                    }
                    for k in p4b6_evals
                },
                "parity": parity6,
                "parity_full24": parity6["full_test24_ba_matched"],
            },
            "ts2": {
                "checkpoint": str(ckpt2),
                "checkpoint_sha256": _sha256(ckpt2),
                "evaluations": {k: slim_eval(v) for k, v in p4b2_evals.items()},
                "frozen": {
                    k: {
                        m: frozen["ts2"][k][m]
                        for m in (
                            "f1",
                            "iou",
                            "precision",
                            "recall",
                            "tp",
                            "fp",
                            "fn",
                            "n_windows",
                        )
                    }
                    for k in p4b2_evals
                },
                "parity": parity2,
                "parity_full24": parity2["full_test24_ba_matched"],
            },
        },
        "paired_per_fire_manuscript16": paired,
        "inference_diagnostics": diagnostics,
        "recommended_retraining_ablations": recommendations,
        "inconsistencies": inconsistencies,
        "verdict": {
            "e1_f1_approx_0_562_reproduced": full24_match["ok"],
            "material_evaluation_inconsistencies": bool(inconsistencies),
        },
    }
    # Attach full curves path reference
    report["learning_curves_full_json"] = str(out_dir / "learning_curves.json")

    _write_json(out_dir / "validation_audit_report.json", report)
    md_path = write_markdown_report(out_dir, report)
    # Also keep a thin summary for quick reading
    summary = {
        "verdict": report["verdict"],
        "e1_full24": report["e1_reproduction"]["evaluations"]["full_test24_ba_matched"],
        "phase4b_ts6_parity_ok": parity6["full_test24_ba_matched"]["ok"],
        "phase4b_ts2_parity_ok": parity2["full_test24_ba_matched"]["ok"],
        "membership_ok": counts_ok,
        "leakage_ok": leak["all_spot_exclude_day_L"] and leak["last2_identity_ts6_vs_ts2"],
        "weights": weights,
        "diagnostics_f1": {
            "clean": diagnostics["clean"]["f1"],
            "reverse_time": diagnostics["reverse_time"]["f1"],
            "shuffle_older_keep_newest": diagnostics["shuffle_older_keep_newest"]["f1"],
            "ablate_older_zero": diagnostics["ablate_older_zero"]["f1"],
        },
        "best_epoch": curves["best_epoch_by_val_f1_at_0_5"],
        "inconsistencies": inconsistencies,
        "markdown": str(md_path),
        "elapsed_min": report["elapsed_min"],
    }
    _write_json(out_dir / "SUMMARY.json", summary)

    print("=== AUDIT COMPLETE ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {md_path}", flush=True)
    return 0 if full24_match["ok"] and not inconsistencies else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        from datetime import datetime, timezone as _tz

        tb = traceback.format_exc()
        print(tb, flush=True)
        crash = ROOT / "logs" / "phase4c_e1_validation_audit_CRASH.txt"
        crash.parent.mkdir(parents=True, exist_ok=True)
        crash.write_text(
            f"{datetime.now(_tz.utc).isoformat()}\n{tb}\n", encoding="utf-8"
        )
        raise
