"""Explicit V1 / V2 data loading. Never silently swaps protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from wtp.data.phase4b_bridge import (
    EXPECTED_COUNTS,
    EXPECTED_POS_WEIGHT,
    build_matched_ctx,
    import_preload,
    load_fp_stats,
)
from wtp.data.v2.constants import PROTOCOL_VERSION as V2_PROTOCOL_VERSION
from wtp.data.v2.dataset import PreloadedMatchedDatasetV2, preload_matched_windows_v2
from wtp.data.v2.normalize import load_fp_stats as load_v2_fp_stats
from wtp.utils.paths import PROJECT_ROOT, load_paths

FORECAST_CHANNELS = (22, 23, 24, 25)


class ForecastMaskedDataset(Dataset):
    """Wrap a preloaded dataset; zero forecast channels after load (conservative)."""

    def __init__(
        self,
        base: Dataset,
        channels: Sequence[int] = FORECAST_CHANNELS,
        fill: float = 0.0,
    ):
        self.base = base
        self.channels = list(channels)
        self.fill = float(fill)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if len(item) == 3:
            x, y, meta = item
        else:
            x, y = item
            meta = None
        x = x.clone()
        for ch in self.channels:
            x[ch] = self.fill
        if meta is None:
            return x, y
        return x, y, meta


def resolve_v2_stats_path(paths: dict | None = None) -> Path:
    cfg = paths or load_paths()
    # Prefer project-local V2 stats
    local = PROJECT_ROOT / "outputs" / "v2_protocol" / "fp_stats_matched_v2.npz"
    if local.exists():
        return local
    # Allow absolute override from paths if added later
    alt = cfg.get("v2", {}).get("fp_stats") if isinstance(cfg.get("v2"), dict) else None
    if alt and Path(alt).exists():
        return Path(alt)
    raise FileNotFoundError(
        f"V2 FP stats not found at {local}. Run scripts/prepare_v2_protocol.py first."
    )


def build_ctx_for_protocol(
    protocol_key: str,
    paths: dict | None = None,
) -> dict[str, Any]:
    """Build matched context with protocol-specific FP stats.

    V1 uses frozen Phase4B stats. V2 replaces mean/std with matched-train V2 stats
    without mutating the V1 npz on disk.
    """
    cfg = paths or load_paths()
    key = protocol_key.lower().replace("-", "_")
    ctx = build_matched_ctx(cfg)
    ctx["protocol_key"] = protocol_key
    ctx["pos_weight"] = EXPECTED_POS_WEIGHT
    ctx["expected_counts"] = dict(EXPECTED_COUNTS)

    if key in ("v1", "v1_frozen"):
        ctx["preprocessing_version"] = "1.0"
        ctx["loader"] = "v1_phase4b"
        # ctx already has V1 fp_mean/std from phase4b shared
        return ctx

    if key.startswith("v2"):
        stats_path = resolve_v2_stats_path(cfg)
        mean, std = load_v2_fp_stats(stats_path)
        ctx["fp_mean"] = mean
        ctx["fp_std"] = std
        ctx["stats_path"] = stats_path
        ctx["v1_stats_path"] = load_fp_stats(cfg)[2]  # preserve pointer to frozen V1
        ctx["preprocessing_version"] = V2_PROTOCOL_VERSION
        ctx["loader"] = "v2_corrected"
        return ctx

    raise KeyError(f"Unsupported protocol_key={protocol_key}")


def get_preload_fns(
    protocol_key: str,
) -> tuple[Callable, type]:
    """Return (preload_fn, DatasetCls) for the selected protocol."""
    key = protocol_key.lower().replace("-", "_")
    if key in ("v1", "v1_frozen"):
        preload, Ds = import_preload()
        return preload, Ds
    if key.startswith("v2"):
        return preload_matched_windows_v2, PreloadedMatchedDatasetV2
    raise KeyError(protocol_key)


def make_datasets(
    ctx: dict[str, Any],
    *,
    ts: int,
    protocol_key: str,
    forecast_policy: str = "include_as_forecast",
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    augment_train: bool = True,
    verbose: bool = True,
    mask_channels: Optional[Sequence[int]] = None,
    aug_seed: int = 42,
) -> dict[str, Any]:
    """Preload train/val windows under the selected protocol.

    For dry-run / smoke, pass small max_train/max_val.
    ``mask_channels`` zeros listed model channels after aug (e.g. ENV-0: 8..25).
    ``forecast_policy=mask_zero`` zeros GFS channels 22–25 (no-op if already in mask).
    ``aug_seed`` seeds directional geometric augmentation (defaults 42 for V1/V2 seed-42 parity).
    """
    from wtp.data.channel_mask import MaskedPreloadedDataset

    preload_fn, Ds = get_preload_fns(protocol_key)
    train_rows = ctx["train_rows"][: max_train or None]
    val_rows = ctx["val_rows"][: max_val or None]

    common_kw = dict(
        fire_by_id=ctx["fire_by_id"],
        ts=ts,
        fp_mean=ctx["fp_mean"],
        fp_std=ctx["fp_std"],
        verbose=verbose,
    )
    # V1 preload signature uses split_name; V2 uses split_name too + max_windows
    key = protocol_key.lower().replace("-", "_")
    if key.startswith("v2"):
        train_wins = preload_fn(
            train_rows,
            split_name="train",
            max_windows=max_train,
            **common_kw,
        )
        val_wins = preload_fn(
            val_rows,
            split_name="val",
            max_windows=max_val,
            **common_kw,
        )
        train_ds = Ds(
            train_wins,
            augment=augment_train,
            fp_mean=ctx["fp_mean"],
            fp_std=ctx["fp_std"],
            correct_directional=True,
            seed=int(aug_seed),
        )
        val_ds = Ds(
            val_wins,
            augment=False,
            fp_mean=ctx["fp_mean"],
            fp_std=ctx["fp_std"],
            correct_directional=True,
            seed=int(aug_seed),
        )
    else:
        train_wins = preload_fn(
            train_rows, ctx["fire_by_id"], ts, ctx["fp_mean"], ctx["fp_std"], "train"
        )
        val_wins = preload_fn(
            val_rows, ctx["fire_by_id"], ts, ctx["fp_mean"], ctx["fp_std"], "val"
        )
        train_ds = Ds(train_wins, augment=augment_train)
        val_ds = Ds(val_wins, augment=False)

    applied_masks: list[str] = []
    if mask_channels:
        ch = list(mask_channels)
        train_ds = MaskedPreloadedDataset(train_ds, ch, 0.0)
        val_ds = MaskedPreloadedDataset(val_ds, ch, 0.0)
        applied_masks.append(f"channels:{ch}")

    if forecast_policy == "mask_zero":
        # Skip redundant wrap if ENV-0 already covers 22–25
        need = [c for c in FORECAST_CHANNELS if not mask_channels or c not in set(mask_channels)]
        if need:
            train_ds = ForecastMaskedDataset(train_ds, need, 0.0)
            val_ds = ForecastMaskedDataset(val_ds, need, 0.0)
            applied_masks.append(f"gfs:{need}")
        else:
            applied_masks.append("gfs:covered_by_channel_mask")
    elif forecast_policy not in ("include_as_forecast", "include"):
        raise ValueError(
            f"Unsupported forecast_policy={forecast_policy}. "
            "Use include_as_forecast or mask_zero (drop_channels not wired for 27-ch parity)."
        )

    return {
        "train_windows": train_wins,
        "val_windows": val_wins,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "n_train": len(train_wins),
        "n_val": len(val_wins),
        "protocol_key": protocol_key,
        "forecast_policy": forecast_policy,
        "mask_channels": list(mask_channels) if mask_channels else [],
        "applied_masks": applied_masks,
        "ts": ts,
        "dataset_cls": Ds,
        "fp_mean": ctx["fp_mean"],
        "fp_std": ctx["fp_std"],
    }


def build_model_from_config(model_cfg: dict[str, Any], ts: int):
    """Instantiate MeanPool or RecencyFusion from resolved model block.

    If ``freeze_temporal_fusion`` is true on a RecencyFusion arm, fusion
    parameters are zeroed and frozen to uniform 1/T (mean-pool-equivalent).
    """
    from wtp.models import MeanPoolUNet3D, RecencyFusionUNet3D
    from wtp.models.recency_fusion import freeze_temporal_fusion_uniform

    name = model_cfg["name"]
    enc = tuple(model_cfg.get("enc") or (64, 128, 256, 512))
    bneck = int(model_cfg.get("bneck") or 1024)
    in_ch = int(model_cfg.get("in_channels") or 27)
    if name in ("mean_pool", "meanpool", "spase"):
        return MeanPoolUNet3D(in_ch=in_ch, enc=enc, bneck=bneck)
    if name in ("recency_fusion", "e1", "recency"):
        model = RecencyFusionUNet3D(
            in_ch=in_ch,
            enc=enc,
            bneck=bneck,
            max_t=int(model_cfg.get("max_t") or ts),
            init_recency_scale=float(model_cfg.get("init_recency_scale") or 0.0),
        )
        if model_cfg.get("freeze_temporal_fusion"):
            mode = str(model_cfg.get("freeze_temporal_mode") or "uniform")
            if mode != "uniform":
                raise ValueError(f"Unsupported freeze_temporal_mode={mode}")
            meta = freeze_temporal_fusion_uniform(model)
            model._freeze_temporal_meta = meta  # type: ignore[attr-defined]
        return model
    raise KeyError(f"Unknown model name={name}")
