#!/usr/bin/env python3
"""STEP 2 — Validate RecencyFusionUNet3D for Experiment E1 (CPU/GPU smoke, no training)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.models import RecencyFusionUNet3D, MeanPoolUNet3D, count_params  # noqa: E402


def main() -> int:
    out_dir = ROOT / "outputs" / "phase4c_e1" / "model_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(42)

    model = RecencyFusionUNet3D(max_t=6, init_recency_scale=0.0).to(device)
    baseline = MeanPoolUNet3D().to(device)
    n = count_params(model)
    checks = {}

    checks["param_count"] = {
        "n_params": n,
        "expected": 24_279_568,
        "pass": n == 24_279_568,
        "baseline": count_params(baseline),
        "delta": n - count_params(baseline),
    }

    w = model.fusion_weights(6).detach()
    checks["uniform_init"] = {
        "weights": w.tolist(),
        "pass": bool(torch.allclose(w, torch.full((6,), 1.0 / 6.0))),
        "indexing": {
            "0": "oldest (t-5)",
            "5": "newest (t)",
            "prior": model.temporal_fusion.prior.tolist(),
        },
    }

    # Gradients to fusion
    x = torch.randn(1, 27, 6, 64, 64)
    y = torch.zeros(1, 1, 64, 64)
    y[:, :, 16:32, 16:32] = 1.0
    loss = F.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    checks["grads"] = {
        "logits_grad_norm": float(model.temporal_fusion.logits.grad.norm().item()),
        "recency_scale_grad": float(model.temporal_fusion.recency_scale.grad.item()),
        "pass": bool(
            torch.isfinite(model.temporal_fusion.logits.grad).all()
            and model.temporal_fusion.logits.grad.abs().sum() > 0
        ),
    }

    # Force nonuniform weights → output must change under temporal permutation
    with torch.no_grad():
        model.temporal_fusion.logits.copy_(
            torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        )
        model.temporal_fusion.recency_scale.copy_(torch.tensor(0.0))
        w2 = model.fusion_weights(6)
        x = torch.randn(1, 27, 6, 64, 64)
        model.eval()
        y_id = model(x)
        y_rev = model(x[:, :, torch.arange(5, -1, -1), :, :])
        diff = float((y_id - y_rev).abs().max().item())
    checks["nonuniform_permutation_sensitivity"] = {
        "weights": w2.tolist(),
        "max_abs_output_diff_vs_reverse": diff,
        "pass": diff > 1e-4,
        "note": "With unequal positional weights, reversing time changes the fused map.",
    }

    # Redundancy note
    checks["recency_scale_redundancy"] = {
        "statement": (
            "For fixed TS=6, unrestricted positional logits already span all "
            "softmax weight vectors on the 6-simplex. An additive "
            "recency_scale * linspace(-1,1) prior is an affine reparameterization "
            "of the same logit family and is therefore redundant for expressivity; "
            "it only changes initialization geometry / optimization path. "
            "This model is positional temporal weighting, NOT spatiotemporal attention."
        )
    }

    checks["architecture_scope"] = {
        "changed": "final temporal collapse only (mean -> learned positional weights)",
        "unchanged": [
            "inp Conv3d",
            "ResBlock3D encoder",
            "spatial MaxPool3d",
            "ASPP3D bottleneck",
            "ConvTranspose3d decoder",
            "2D head",
        ],
        "not_implemented": "spatial-temporal attention / content-aware temporal attention",
    }

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pass": all(
            checks[k]["pass"]
            for k in (
                "param_count",
                "uniform_init",
                "grads",
                "nonuniform_permutation_sensitivity",
            )
        ),
        "checks": checks,
    }
    (out_dir / "e1_model_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    md = [
        "# E1 model validation — RecencyFusionUNet3D",
        "",
        f"**Pass:** **{report['pass']}**",
        "",
        f"- Params: `{n}` (expected 24279568)",
        f"- Uniform init weights: {checks['uniform_init']['weights']}",
        f"- Fusion grads OK: {checks['grads']['pass']}",
        f"- Nonuniform → permutation-sensitive: Δ={diff:.6g}",
        "",
        "## Recency-scale redundancy",
        "",
        checks["recency_scale_redundancy"]["statement"],
        "",
        "## Not spatiotemporal attention",
        "",
        "Weights are content-agnostic and shared across space/channels; only the "
        "final temporal axis is reweighted.",
        "",
    ]
    (out_dir / "E1_MODEL_VALIDATION.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "n_params": n, "perm_diff": diff}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
