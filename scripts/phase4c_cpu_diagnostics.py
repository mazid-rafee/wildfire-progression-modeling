#!/usr/bin/env python3
"""Phase 4C CPU-only diagnostics (no GPU train/eval, no writes to external trees).

1. TS=2 vs TS=6 last-two-frame identity after deterministic preprocessing
2. Frozen TS=6 temporal permutation invariance (small CPU inference)
3. Missing / zero-filled observation audit across six temporal positions
4. Writes reports under this project's outputs/ only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.data.matched_loader import build_window, load_matched_context  # noqa: E402
from wtp.utils.external import import_phase3a_modules  # noqa: E402
from wtp.utils.paths import load_paths  # noqa: E402


def _sha16(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sample_rows(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    if n >= len(rows):
        return list(rows)
    idx = rng.choice(len(rows), size=n, replace=False)
    return [rows[int(i)] for i in sorted(idx.tolist())]


def diag_last_two_frames(ctx: dict, rows: list[dict], out_dir: Path) -> dict:
    """Verify TS=6[..., -2:] == TS=2 after preprocessing."""
    mismatches = []
    max_abs = 0.0
    records = []
    for r in rows:
        x2 = build_window(ctx, r, ts=2)
        x6 = build_window(ctx, r, ts=6)
        last2 = x6[:, -2:, :, :]
        diff = np.max(np.abs(last2.astype(np.float64) - x2.astype(np.float64)))
        max_abs = max(max_abs, float(diff))
        ok = bool(np.array_equal(last2, x2))
        rec = {
            "target_key": r["target_key"],
            "identical": ok,
            "max_abs_diff": float(diff),
            "x2_shape": list(x2.shape),
            "x6_shape": list(x6.shape),
            "x2_sha16": _sha16(x2),
            "x6_last2_sha16": _sha16(last2),
        }
        records.append(rec)
        if not ok:
            mismatches.append(rec)

    report = {
        "n_checked": len(rows),
        "n_identical": len(rows) - len(mismatches),
        "n_mismatch": len(mismatches),
        "max_abs_diff_all": max_abs,
        "pass": len(mismatches) == 0,
        "mismatches": mismatches[:20],
        "records": records,
    }
    _write_json(out_dir / "last_two_frame_identity.json", report)
    return report


def diag_permutation_invariance(
    paths: dict, rows: list[dict], ctx: dict, out_dir: Path, n_infer: int = 4
) -> dict:
    """CPU eval of frozen TS=6 SpaSE under temporal permutations."""
    p3 = import_phase3a_modules(paths)
    SEUNet3DPred = p3["SEUNet3DPred"]
    ckpt_path = Path(paths["phase4b"]["checkpoint_ts6"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SEUNet3DPred(in_ch=27)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample = sample_rows(rows, n_infer, seed=1)
    perms = {
        "identity": list(range(6)),
        "reverse": list(range(5, -1, -1)),
        "rotate_left_1": [1, 2, 3, 4, 5, 0],
        "swap_oldest_newest": [5, 1, 2, 3, 4, 0],
        "swap_last_two": [0, 1, 2, 3, 5, 4],
    }

    cases = []
    max_abs_all = 0.0
    all_invariant = True
    with torch.no_grad():
        for r in sample:
            x6 = build_window(ctx, r, ts=6)
            xt = torch.from_numpy(x6).unsqueeze(0)  # 1,C,T,H,W
            base = model(xt).numpy()
            case = {"target_key": r["target_key"], "perms": {}}
            for name, order in perms.items():
                xp = xt[:, :, order, :, :]
                yp = model(xp).numpy()
                diff = float(np.max(np.abs(yp - base)))
                max_abs_all = max(max_abs_all, diff)
                identical = bool(np.allclose(yp, base, rtol=0, atol=0))
                close_1e5 = bool(np.allclose(yp, base, rtol=1e-5, atol=1e-5))
                case["perms"][name] = {
                    "order": order,
                    "max_abs_diff_vs_identity": diff,
                    "bitwise_identical": identical,
                    "allclose_1e5": close_1e5,
                }
                if name != "identity" and not close_1e5:
                    all_invariant = False
            cases.append(case)

    # Analytic note from architecture: time-separable kernels + final mean.
    report = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_ts": ckpt.get("ts"),
        "n_params_ckpt": ckpt.get("n_params"),
        "device": "cpu",
        "n_windows": len(sample),
        "permutations": perms,
        "max_abs_diff_all": max_abs_all,
        "invariant_atol_1e5": all_invariant,
        "cases": cases,
        "architecture_expectation": (
            "SEUNet3DPred uses temporal kernel size 1 throughout, SE global pool is "
            "T-permutation invariant, and the final collapse is mean(dim=2). "
            "In eval mode, outputs should be invariant to temporal permutations."
        ),
    }
    _write_json(out_dir / "permutation_invariance.json", report)
    return report


def _modality_zero_flags(
    fire_dir: str, day_path: str, p3: dict, cs: int, *, probe_sizes: bool = False
) -> dict[str, bool]:
    """Return which modalities are missing / effectively zero-filled for one day.

    Default path uses filesystem existence only (fast, full-split friendly).
    Optional ``probe_sizes`` opens FirePred/Day headers to detect undersized
    rasters or missing BA band-8 (slower; use on a sample).
    """
    night_dir = os.path.join(fire_dir, "VIIRS_Night")
    night_fname = os.path.basename(day_path).replace("_VIIRS_Day", "_VIIRS_Night")
    night_path = os.path.join(night_dir, night_fname)
    night_missing = not os.path.exists(night_path)

    fp_dir = os.path.join(fire_dir, "FirePred")
    fp_fname = os.path.basename(day_path).replace("_VIIRS_Day", "_FirePred")
    fp_path = os.path.join(fp_dir, fp_fname)
    fp_missing = not os.path.exists(fp_path)
    fp_too_small = False
    ba_missing = False

    if probe_sizes:
        try:
            import rasterio
        except ImportError:  # pragma: no cover
            rasterio = None
        if rasterio is not None:
            if not fp_missing:
                with rasterio.open(fp_path) as src:
                    if src.count < 19 or src.height < cs or src.width < cs:
                        fp_too_small = True
            with rasterio.open(day_path) as src:
                ba_missing = src.count < 8
        else:
            # Fallback: full loaders (slow)
            fp = p3["load_firepred_raw"](fire_dir, day_path)
            fp_missing = fp is None
            if fp is not None:
                fp_too_small = fp.shape[1] < cs or fp.shape[2] < cs
            ba_missing = p3["load_ba_mask"](day_path) is None

    return {
        "night_missing": night_missing,
        "firepred_missing": fp_missing,
        "firepred_too_small": fp_too_small,
        "ba_mask_missing": ba_missing,
        "firepred_zero_filled": fp_missing or fp_too_small,
        "ba_zero_filled": ba_missing,
    }


def diag_missing_zero_fill(
    ctx: dict,
    rows: list[dict],
    out_dir: Path,
    max_rows: int | None = None,
    size_probe_rows: int = 48,
) -> dict:
    """Audit missing Night / FirePred / BA across the six TS=6 temporal slots."""
    p3 = ctx["p3"]
    CFG = ctx["CFG"]
    cs = CFG.CROP_SIZE
    use_rows = rows if max_rows is None else rows[:max_rows]

    # position 0 = oldest, 5 = newest (t)
    pos_counts = {
        k: [0] * 6
        for k in (
            "night_missing",
            "firepred_missing",
            "firepred_too_small",
            "firepred_zero_filled",
            "ba_mask_missing",
            "ba_zero_filled",
        )
    }
    calendar_gap_hist = defaultdict(int)
    n_windows = 0
    n_with_any_gap_gt1 = 0

    for r in use_rows:
        info = ctx["fire_by_id"][r["fire_id"]]
        days = info["day_files"]
        L = int(r["label_idx"])
        input_days = days[L - 6 : L]
        assert len(input_days) == 6
        n_windows += 1

        # calendar gaps between consecutive input days
        dates = []
        for d in input_days:
            # basename like 2017-07-21_VIIRS_Day.tif
            dates.append(os.path.basename(d)[:10])
        gaps = []
        for a, b in zip(dates[:-1], dates[1:]):
            da = datetime.strptime(a, "%Y-%m-%d")
            db = datetime.strptime(b, "%Y-%m-%d")
            gaps.append((db - da).days)
        max_gap = max(gaps) if gaps else 1
        calendar_gap_hist[int(max_gap)] += 1
        if max_gap > 1:
            n_with_any_gap_gt1 += 1

        for t_idx, day_path in enumerate(input_days):
            flags = _modality_zero_flags(
                info["fire_dir"], day_path, p3, cs, probe_sizes=False
            )
            for k, v in flags.items():
                if v:
                    pos_counts[k][t_idx] += 1

    # Header probe on a sample for undersized FirePred / missing BA band
    probe_rows = sample_rows(use_rows, min(size_probe_rows, len(use_rows)), seed=7)
    probe_counts = {
        "firepred_too_small": [0] * 6,
        "ba_mask_missing": [0] * 6,
        "n_probe_windows": 0,
    }
    for r in probe_rows:
        info = ctx["fire_by_id"][r["fire_id"]]
        days = info["day_files"]
        L = int(r["label_idx"])
        input_days = days[L - 6 : L]
        probe_counts["n_probe_windows"] += 1
        for t_idx, day_path in enumerate(input_days):
            flags = _modality_zero_flags(
                info["fire_dir"], day_path, p3, cs, probe_sizes=True
            )
            if flags["firepred_too_small"]:
                probe_counts["firepred_too_small"][t_idx] += 1
            if flags["ba_mask_missing"]:
                probe_counts["ba_mask_missing"][t_idx] += 1

    def rates(counts: list[int], denom: int) -> list[float]:
        return [c / max(denom, 1) for c in counts]

    report = {
        "n_windows": n_windows,
        "method": "existence_full_split + rasterio_header_probe_sample",
        "position_index_meaning": {
            "0": "oldest (t-5)",
            "1": "t-4",
            "2": "t-3",
            "3": "t-2",
            "4": "t-1",
            "5": "newest / t (same as TS=2 last frame)",
        },
        "counts_by_position": pos_counts,
        "rates_by_position": {
            k: rates(v, n_windows) for k, v in pos_counts.items()
        },
        "size_probe": {
            **probe_counts,
            "rates_firepred_too_small": rates(
                probe_counts["firepred_too_small"], probe_counts["n_probe_windows"]
            ),
            "rates_ba_mask_missing": rates(
                probe_counts["ba_mask_missing"], probe_counts["n_probe_windows"]
            ),
        },
        "n_windows_max_calendar_gap_gt1": n_with_any_gap_gt1,
        "fraction_max_calendar_gap_gt1": n_with_any_gap_gt1 / max(n_windows, 1),
        "max_calendar_gap_hist": dict(sorted(calendar_gap_hist.items())),
        "notes": [
            "Day VIIRS files always exist for selected indices (windows are built from day_files).",
            "Missing Night → 2 night channels zero-filled after sat normalize.",
            "Missing / undersized FirePred → 18 FP channels zero-filled.",
            "Missing BA band-8 → BA channel zero-filled (last frame uses ba_last).",
            "Calendar gaps >1 day mean non-consecutive observations in the day_files list order; slots are not zero-filled for missing calendar days.",
            "Full-split counts use path existence. Undersized FirePred / BA band-8 absence are probed on a sample via rasterio headers.",
        ],
    }
    _write_json(out_dir / "missing_zero_fill_audit.json", report)
    return report


def write_summary_md(
    out_dir: Path,
    pop_counts: dict,
    last2: dict,
    perm: dict,
    miss: dict,
) -> None:
    lines = [
        "# Phase 4C CPU diagnostics summary",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        "",
        "Constraints honored: CPU-only; no writes to TS-SatFire / DeepFire trees; "
        "no GPU train/eval; no dataset duplication.",
        "",
        "## Populations (matched Phase 4B)",
        "",
        f"- train seed-42: {pop_counts['train']} (expected {pop_counts['expected_train']})",
        f"- val: {pop_counts['val']} (expected {pop_counts['expected_val']})",
        f"- test24 BA: {pop_counts['test24']} (expected {pop_counts['expected_test24']})",
        "",
        "## 1. Last-two-frame identity (TS=6 vs TS=2)",
        "",
        f"- Checked: **{last2['n_checked']}** windows",
        f"- Identical: **{last2['n_identical']}**",
        f"- Mismatches: **{last2['n_mismatch']}**",
        f"- Max |diff|: `{last2['max_abs_diff_all']}`",
        f"- Pass: **{last2['pass']}**",
        "",
        "## 2. Frozen TS=6 permutation invariance",
        "",
        f"- Checkpoint epoch={perm.get('checkpoint_epoch')} ts={perm.get('checkpoint_ts')}",
        f"- Windows inferred: {perm['n_windows']}",
        f"- Max |Δlogit| vs identity over tested perms: `{perm['max_abs_diff_all']}`",
        f"- Invariant @ atol=1e-5: **{perm['invariant_atol_1e5']}**",
        "",
        f"> {perm['architecture_expectation']}",
        "",
        "## 3. Missing / zero-filled observations (TS=6 positions 0..5)",
        "",
        f"- Windows audited: **{miss['n_windows']}**",
        f"- Fraction with max calendar gap > 1 day: "
        f"**{miss['fraction_max_calendar_gap_gt1']:.4f}** "
        f"({miss['n_windows_max_calendar_gap_gt1']})",
        "",
        "### FirePred zero-fill rate by position",
        "",
        "| pos | rate |",
        "|----:|-----:|",
    ]
    for i, rate in enumerate(miss["rates_by_position"]["firepred_zero_filled"]):
        lines.append(f"| {i} | {rate:.4f} |")
    lines += [
        "",
        "### Night missing rate by position",
        "",
        "| pos | rate |",
        "|----:|-----:|",
    ]
    for i, rate in enumerate(miss["rates_by_position"]["night_missing"]):
        lines.append(f"| {i} | {rate:.4f} |")
    lines += [
        "",
        "### BA mask missing rate by position",
        "",
        "| pos | rate |",
        "|----:|-----:|",
    ]
    for i, rate in enumerate(miss["rates_by_position"]["ba_mask_missing"]):
        lines.append(f"| {i} | {rate:.4f} |")
    lines += [
        "",
        "JSON artifacts: `last_two_frame_identity.json`, "
        "`permutation_invariance.json`, `missing_zero_fill_audit.json`.",
        "",
    ]
    _write_md(out_dir / "PHASE4C_CPU_DIAGNOSTICS.md", "\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "phase4c_diagnostics",
    )
    ap.add_argument("--n-identity", type=int, default=24, help="Windows for last-2 check")
    ap.add_argument("--n-perm", type=int, default=4, help="Windows for permutation CPU infer")
    ap.add_argument(
        "--missing-split",
        choices=("train", "val", "test24", "all"),
        default="train",
        help="Which matched population to audit for missing/zero-fill",
    )
    ap.add_argument(
        "--missing-max-rows",
        type=int,
        default=None,
        help="Optional cap for missing audit (default: full selected split)",
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = load_paths()
    print("Loading matched context (read-only)...", flush=True)
    ctx = load_matched_context(paths)
    exp = ctx["expected"]
    pop_counts = {
        "train": len(ctx["train_rows"]),
        "val": len(ctx["val_rows"]),
        "test24": len(ctx["test24_rows"]),
        "expected_train": exp["train_matched_seed42"],
        "expected_val": exp["val_matched"],
        "expected_test24": exp["test24_ba_matched"],
    }
    print("Population counts:", pop_counts, flush=True)
    if (
        pop_counts["train"] != pop_counts["expected_train"]
        or pop_counts["val"] != pop_counts["expected_val"]
        or pop_counts["test24"] != pop_counts["expected_test24"]
    ):
        print("ERROR: population count mismatch vs Phase 4B EXPECTED", flush=True)
        return 1

    id_rows = sample_rows(ctx["train_rows"], args.n_identity, seed=0)
    # also include a few val/test for coverage
    id_rows += sample_rows(ctx["val_rows"], min(4, len(ctx["val_rows"])), seed=2)
    id_rows += sample_rows(ctx["test24_rows"], min(4, len(ctx["test24_rows"])), seed=3)

    print(f"[1/3] Last-two-frame identity on {len(id_rows)} windows...", flush=True)
    last2 = diag_last_two_frames(ctx, id_rows, out_dir)
    print(
        f"  pass={last2['pass']} identical={last2['n_identical']}/{last2['n_checked']} "
        f"max_abs={last2['max_abs_diff_all']}",
        flush=True,
    )

    print(f"[2/3] Permutation invariance (CPU) on {args.n_perm} windows...", flush=True)
    perm = diag_permutation_invariance(
        paths, ctx["val_rows"], ctx, out_dir, n_infer=args.n_perm
    )
    print(
        f"  invariant@1e-5={perm['invariant_atol_1e5']} "
        f"max_abs={perm['max_abs_diff_all']}",
        flush=True,
    )

    split_map = {
        "train": ctx["train_rows"],
        "val": ctx["val_rows"],
        "test24": ctx["test24_rows"],
        "all": ctx["train_rows"] + ctx["val_rows"] + ctx["test24_rows"],
    }
    miss_rows = split_map[args.missing_split]
    print(
        f"[3/3] Missing/zero-fill audit on {args.missing_split} "
        f"(n={len(miss_rows)}, cap={args.missing_max_rows})...",
        flush=True,
    )
    miss = diag_missing_zero_fill(
        ctx, miss_rows, out_dir, max_rows=args.missing_max_rows
    )
    print(
        f"  firepred_zero_fill rates={miss['rates_by_position']['firepred_zero_filled']}",
        flush=True,
    )

    write_summary_md(out_dir, pop_counts, last2, perm, miss)
    print(f"Wrote summary → {out_dir / 'PHASE4C_CPU_DIAGNOSTICS.md'}", flush=True)
    return 0 if last2["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
