"""RecencyFusionUNet3D: learnable, recency-aware temporal fusion.

Preserves the shared spatial encoder / bottleneck / decoder / 2D head from
``MeanPoolUNet3D``. Only the final equal ``mean(dim=2)`` is replaced.

Design (minimal):
  w = softmax(logits[:T] + recency_scale * prior[:T])
  pooled = sum_t w_t * feats[:, :, t]

where ``prior`` is a fixed increasing sequence favoring newer frames
(normalized to mean 0). ``logits`` and ``recency_scale`` are learnable.

At initialization with zero logits and recency_scale=0 the fusion reduces
to uniform 1/T averaging (mean-pool-equivalent start).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from wtp.models.registry import register
from wtp.models.spase_backbone import MeanPoolUNet3D, count_params


class RecencyAwareTemporalFusion(nn.Module):
    """Learnable temporal weights with an additive recency prior."""

    def __init__(self, max_t: int = 6, init_recency_scale: float = 0.0):
        super().__init__()
        if max_t < 2:
            raise ValueError("max_t must be >= 2")
        self.max_t = max_t
        # Content-agnostic logits (shared across batch/channel/space).
        self.logits = nn.Parameter(torch.zeros(max_t))
        # Strength of the fixed recency prior; 0 => uniform if logits~0.
        self.recency_scale = nn.Parameter(torch.tensor(float(init_recency_scale)))
        prior = torch.linspace(-1.0, 1.0, steps=max_t)
        self.register_buffer("prior", prior, persistent=False)

    def weights(self, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if t > self.max_t:
            raise ValueError(f"T={t} exceeds max_t={self.max_t}")
        logits = self.logits[:t] + self.recency_scale * self.prior[:t]
        return torch.softmax(logits, dim=0).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> (B, C, H, W)"""
        if x.ndim != 5:
            raise ValueError(f"expected 5D (B,C,T,H,W), got {tuple(x.shape)}")
        t = x.shape[2]
        w = self.weights(t, x.device, x.dtype).view(1, 1, t, 1, 1)
        return (x * w).sum(dim=2)


class RecencyFusionUNet3D(MeanPoolUNet3D):
    """Mean-pool backbone + recency-aware temporal fusion instead of mean pool."""

    def __init__(
        self,
        in_ch: int = 27,
        enc: tuple[int, ...] = (64, 128, 256, 512),
        bneck: int = 1024,
        se_reduction: int = 8,
        max_t: int = 6,
        init_recency_scale: float = 0.0,
    ):
        super().__init__(
            in_ch=in_ch, enc=enc, bneck=bneck, se_reduction=se_reduction
        )
        self.temporal_fusion = RecencyAwareTemporalFusion(
            max_t=max_t, init_recency_scale=init_recency_scale
        )

    def temporal_pool(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal_fusion(x)

    def fusion_weights(self, t: Optional[int] = None) -> torch.Tensor:
        t = self.temporal_fusion.max_t if t is None else t
        return self.temporal_fusion.weights(
            t, self.temporal_fusion.logits.device, self.temporal_fusion.logits.dtype
        )


def freeze_temporal_fusion_uniform(model: RecencyFusionUNet3D) -> dict:
    """Force uniform 1/T fusion and disable gradients on fusion parameters.

    With logits=0 and recency_scale=0, softmax yields exactly 1/T — mathematically
    equivalent to ``MeanPoolUNet3D.temporal_pool`` (verified in V2 audit).
    """
    if not isinstance(model, RecencyFusionUNet3D):
        raise TypeError(f"expected RecencyFusionUNet3D, got {type(model)}")
    with torch.no_grad():
        model.temporal_fusion.logits.zero_()
        model.temporal_fusion.recency_scale.zero_()
    model.temporal_fusion.logits.requires_grad_(False)
    model.temporal_fusion.recency_scale.requires_grad_(False)
    t = int(model.temporal_fusion.max_t)
    w = model.fusion_weights(t).detach().cpu().tolist()
    n_frozen = sum(
        p.numel() for p in model.temporal_fusion.parameters() if not p.requires_grad
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "mode": "uniform",
        "frozen": True,
        "weights": w,
        "n_fusion_params_frozen": n_frozen,
        "n_trainable_params": n_train,
        "expected_trainable": 24_279_561,  # MeanPool count
    }


def backbone_state_dict(model: nn.Module) -> dict:
    """State dict excluding temporal_fusion.* (shared spatial backbone + head)."""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith("temporal_fusion.")
    }


def load_backbone_state_dict(model: nn.Module, state: dict) -> list[str]:
    """Load backbone/head weights; leave fusion params untouched if present."""
    cur = model.state_dict()
    filtered = {k: v for k, v in state.items() if not k.startswith("temporal_fusion.")}
    cur.update(filtered)
    missing, unexpected = model.load_state_dict(cur, strict=False)
    return list(missing)


@register("mean_pool")
def build_mean_pool(**kwargs):
    # Registry builders may receive arm extras; strip unknown keys.
    allowed = {"in_ch", "enc", "bneck", "se_reduction", "in_channels"}
    kw = {}
    if "in_channels" in kwargs:
        kw["in_ch"] = kwargs["in_channels"]
    if "in_ch" in kwargs:
        kw["in_ch"] = kwargs["in_ch"]
    for k in ("enc", "bneck", "se_reduction"):
        if k in kwargs:
            kw[k] = kwargs[k]
    return MeanPoolUNet3D(**kw)


@register("recency_fusion")
def build_recency_fusion(**kwargs):
    kw = {}
    if "in_channels" in kwargs:
        kw["in_ch"] = kwargs["in_channels"]
    if "in_ch" in kwargs:
        kw["in_ch"] = kwargs["in_ch"]
    for k in ("enc", "bneck", "se_reduction", "max_t", "init_recency_scale"):
        if k in kwargs:
            kw[k] = kwargs[k]
    return RecencyFusionUNet3D(**kw)


def describe_param_delta(baseline: nn.Module, variant: nn.Module) -> dict:
    n_base = count_params(baseline)
    n_var = count_params(variant)
    return {
        "baseline_params": n_base,
        "variant_params": n_var,
        "delta_params": n_var - n_base,
        "delta_pct": 100.0 * (n_var - n_base) / max(n_base, 1),
        "expected_extra": variant.temporal_fusion.max_t + 1  # type: ignore[attr-defined]
        if hasattr(variant, "temporal_fusion")
        else None,
        "uniform_init_entropy": math.log(variant.temporal_fusion.max_t)  # type: ignore[attr-defined]
        if hasattr(variant, "temporal_fusion")
        else None,
    }
