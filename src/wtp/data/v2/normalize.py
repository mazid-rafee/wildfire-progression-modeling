"""Train-matched-only FirePred normalization statistics (V2).

V1 limitation (STATS-SUBSAMPLE): frozen ``fp_stats_matched.npz`` came from
``compute_firepred_stats`` on ≤20 fires × ≤10 days, not the full matched
train population.

V2 recomputes mean/std independently over matched train windows after
geographic alignment (raw FP on VIIRS grid, center-cropped), excluding
val/test events entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from wtp.data.v2.constants import CROP_SIZE, FP_BAND_IDX
from wtp.data.v2.geo_align import align_firepred_to_viirs, center_crop_hw
from wtp.data.v2.io import load_firepred_raw_with_georef, load_frame_sat_v2


def normalize_firepred(
    fp: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    m = mean.reshape(-1, 1, 1)
    s = np.maximum(std, 0.1).reshape(-1, 1, 1)
    out = (fp - m) / s
    out = np.clip(out, -5.0, 5.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _accumulate_crop(
    sums: np.ndarray,
    sqsums: np.ndarray,
    counts: np.ndarray,
    fire_dir: str,
    day_path: str,
    cs: int = CROP_SIZE,
) -> bool:
    fp_raw, fp_ref, _ = load_firepred_raw_with_georef(fire_dir, day_path)
    if fp_raw is None or fp_ref is None:
        return False
    _, viirs_ref = load_frame_sat_v2(fire_dir, day_path)
    if viirs_ref.height < cs or viirs_ref.width < cs:
        return False
    fp_raw = np.nan_to_num(fp_raw, nan=0.0, posinf=0.0, neginf=0.0)
    fp_aln = align_firepred_to_viirs(fp_raw, fp_ref, viirs_ref)
    fp_crop = center_crop_hw(fp_aln, cs, viirs_ref.height, viirs_ref.width)
    for c in range(len(FP_BAND_IDX)):
        ch = fp_crop[c]
        sums[c] += float(ch.sum())
        sqsums[c] += float((ch.astype(np.float64) ** 2).sum())
        counts[c] += ch.size
    return True


def _unique_train_day_keys(
    train_rows: Sequence[dict], fire_by_id: dict, ts: int
) -> list[tuple[str, str, str]]:
    """Return unique (fire_id, fire_dir, day_path) from matched train windows."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for row in train_rows:
        info = fire_by_id[row["fire_id"]]
        days = info["day_files"]
        L = int(row["label_idx"])
        if L < ts:
            continue
        for day_path in days[L - ts : L]:
            key = (row["fire_id"], day_path)
            if key in seen:
                continue
            seen.add(key)
            out.append((row["fire_id"], info["fire_dir"], day_path))
    return out


def compute_matched_train_fp_stats(
    train_rows: Sequence[dict],
    fire_by_id: dict,
    ts: int = 6,
    max_windows: Optional[int] = None,
    verbose: bool = False,
    unique_days: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit FP mean/std on matched train population only.

    Default ``unique_days=True`` accumulates each (fire, day) crop once among
    days that appear in matched train windows (TS history). Val/test fires never
    enter. Set ``unique_days=False`` to weight by window multiplicity.
    """
    n_ch = len(FP_BAND_IDX)
    sums = np.zeros(n_ch, dtype=np.float64)
    sqsums = np.zeros(n_ch, dtype=np.float64)
    counts = np.zeros(n_ch, dtype=np.int64)
    n_days = 0
    n_missing_fp = 0
    train_fire_ids = sorted({r["fire_id"] for r in train_rows})
    rows = list(train_rows)
    if max_windows is not None:
        rows = rows[:max_windows]

    if unique_days:
        day_keys = _unique_train_day_keys(rows, fire_by_id, ts)
        for i, (fid, fire_dir, day_path) in enumerate(day_keys):
            ok = _accumulate_crop(sums, sqsums, counts, fire_dir, day_path)
            if ok:
                n_days += 1
            else:
                n_missing_fp += 1
            if verbose and ((i + 1) % 25 == 0 or i + 1 == len(day_keys)):
                print(
                    f"\r  V2 FP stats (unique days): {i+1}/{len(day_keys)} | ok={n_days}",
                    end="",
                    flush=True,
                )
        n_windows = len(rows)
    else:
        n_windows = 0
        for i, row in enumerate(rows):
            info = fire_by_id[row["fire_id"]]
            days = info["day_files"]
            L = int(row["label_idx"])
            if L < ts:
                continue
            used = False
            for day_path in days[L - ts : L]:
                ok = _accumulate_crop(sums, sqsums, counts, info["fire_dir"], day_path)
                if ok:
                    n_days += 1
                    used = True
                else:
                    n_missing_fp += 1
            if used:
                n_windows += 1
            if verbose and ((i + 1) % 50 == 0 or i + 1 == len(rows)):
                print(
                    f"\r  V2 FP stats: {i+1}/{len(rows)} rows | "
                    f"{n_windows} windows | {n_days} day-crops",
                    end="",
                    flush=True,
                )
    if verbose:
        print()

    mean = (sums / np.maximum(counts, 1)).astype(np.float32)
    var = (sqsums / np.maximum(counts, 1)) - mean.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)
    meta = {
        "protocol": "V2.0-matched-train-only",
        "ts_for_day_union": ts,
        "unique_days": unique_days,
        "n_train_rows_used": len(rows),
        "n_train_rows_total": len(train_rows),
        "n_day_crops": int(n_days),
        "n_missing_fp_days": int(n_missing_fp),
        "n_train_fires": len(train_fire_ids),
        "train_fire_ids": train_fire_ids,
        "counts_per_channel": counts.tolist(),
        "geo_aligned": True,
        "val_test_excluded": True,
        "n_windows_ref": int(n_windows),
    }
    return mean, std, meta


def save_fp_stats(path: Path, mean: np.ndarray, std: np.ndarray, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean.astype(np.float32), std=std.astype(np.float32), meta=json_bytes(meta))
    # also write sidecar json for humans
    import json

    side = path.with_suffix(".json")
    with side.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": mean.astype(float).tolist(),
                "std": std.astype(float).tolist(),
                "meta": meta,
            },
            f,
            indent=2,
        )
    return path


def json_bytes(meta: dict) -> np.ndarray:
    import json

    return np.asarray(json.dumps(meta, default=str))


def load_fp_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    st = np.load(path, allow_pickle=True)
    return st["mean"].astype(np.float32), st["std"].astype(np.float32)


def compare_to_v1(
    v2_mean: np.ndarray,
    v2_std: np.ndarray,
    v1_mean: np.ndarray,
    v1_std: np.ndarray,
) -> dict[str, Any]:
    return {
        "l2_mean": float(np.linalg.norm(v2_mean.astype(np.float64) - v1_mean.astype(np.float64))),
        "l2_std": float(np.linalg.norm(v2_std.astype(np.float64) - v1_std.astype(np.float64))),
        "max_abs_mean": float(np.max(np.abs(v2_mean - v1_mean))),
        "max_abs_std": float(np.max(np.abs(v2_std - v1_std))),
        "per_slot_mean_delta": (v2_mean - v1_mean).astype(float).tolist(),
        "per_slot_std_delta": (v2_std - v1_std).astype(float).tolist(),
    }
