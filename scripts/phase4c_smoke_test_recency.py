#!/usr/bin/env python3
"""CPU forward/backward smoke tests for RecencyFusionUNet3D (no GPU training)."""

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
from wtp.models.recency_fusion import describe_param_delta  # noqa: E402


def main() -> int:
    out_dir = ROOT / "outputs" / "phase4c_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(0)

    baseline = MeanPoolUNet3D().to(device)
    model = RecencyFusionUNet3D(max_t=6, init_recency_scale=0.0).to(device)
    model.train()

    delta = describe_param_delta(baseline, model)
    # max_t logits + 1 recency_scale
    expected_extra = 6 + 1

    results = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "param_counts": delta,
        "expected_extra_params": expected_extra,
        "checks": {},
    }

    # CPU smoke uses 64² (divisible by 16) for speed; shapes match production layout.
    spatial = 64
    results["spatial_size_smoke"] = spatial

    # Shape checks for TS=2 and TS=6
    for ts in (2, 6):
        x = torch.randn(2, 27, ts, spatial, spatial, device=device)
        y = torch.zeros(2, 1, spatial, spatial, device=device)
        # sprinkle a few positives for non-trivial loss
        y[:, :, 20:40, 20:40] = 1.0
        logits = model(x)
        assert logits.shape == (2, 1, spatial, spatial), logits.shape
        loss = F.binary_cross_entropy_with_logits(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()

        grad_ok = True
        nan_params = []
        missing_grad = []
        for name, p in model.named_parameters():
            if p.grad is None:
                missing_grad.append(name)
                grad_ok = False
                continue
            if not torch.isfinite(p.grad).all():
                nan_params.append(name)
                grad_ok = False
        finite_logits = bool(torch.isfinite(logits).all().item())
        finite_loss = bool(torch.isfinite(loss).item())
        results["checks"][f"ts{ts}"] = {
            "input_shape": list(x.shape),
            "output_shape": list(logits.shape),
            "loss": float(loss.item()),
            "finite_logits": finite_logits,
            "finite_loss": finite_loss,
            "grad_ok": grad_ok,
            "missing_grad": missing_grad,
            "nonfinite_grad_params": nan_params,
            "fusion_weights": model.fusion_weights(ts).detach().cpu().tolist(),
        }

    # Uniform init: with zero logits + scale 0, weights == 1/T
    w6 = model.fusion_weights(6).detach()
    uniform = torch.full_like(w6, 1.0 / 6.0)
    results["checks"]["uniform_init"] = {
        "weights": w6.tolist(),
        "max_abs_diff_vs_uniform": float((w6 - uniform).abs().max().item()),
        "pass": bool(torch.allclose(w6, uniform, atol=1e-6)),
    }

    # Gradients reach fusion params
    x = torch.randn(1, 27, 6, spatial, spatial, device=device)
    y = torch.zeros(1, 1, spatial, spatial, device=device)
    y[:, :, 16:32, 16:32] = 1.0
    model.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    results["checks"]["fusion_grads"] = {
        "logits_grad_norm": float(model.temporal_fusion.logits.grad.norm().item()),
        "recency_scale_grad": float(model.temporal_fusion.recency_scale.grad.item()),
        "logits_grad_finite": bool(
            torch.isfinite(model.temporal_fusion.logits.grad).all().item()
        ),
    }

    # Numerical stability under large input
    x_big = torch.randn(1, 27, 6, spatial, spatial, device=device) * 5.0
    with torch.no_grad():
        out_big = model(x_big)
    results["checks"]["large_input"] = {
        "finite": bool(torch.isfinite(out_big).all().item()),
        "abs_max": float(out_big.abs().max().item()),
    }

    # Confirm both produce finite same-shape outs
    with torch.no_grad():
        b_out = baseline(torch.randn(1, 27, 6, spatial, spatial))
        m_out = model(torch.randn(1, 27, 6, spatial, spatial))
    results["checks"]["baseline_vs_variant_shapes"] = {
        "baseline": list(b_out.shape),
        "variant": list(m_out.shape),
        "baseline_params": count_params(baseline),
        "variant_params": count_params(model),
    }

    results["pass"] = all(
        [
            results["param_counts"]["delta_params"] == expected_extra,
            results["checks"]["uniform_init"]["pass"],
            results["checks"]["ts2"]["grad_ok"],
            results["checks"]["ts6"]["grad_ok"],
            results["checks"]["ts2"]["finite_loss"],
            results["checks"]["ts6"]["finite_loss"],
            results["checks"]["large_input"]["finite"],
            results["checks"]["fusion_grads"]["logits_grad_finite"],
        ]
    )

    out_json = out_dir / "smoke_test_recency_fusion.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md = [
        "# RecencyFusionUNet3D CPU smoke test",
        "",
        f"**Generated (UTC):** {results['generated_utc']}",
        f"**Pass:** **{results['pass']}**",
        "",
        "## Parameters",
        "",
        f"- Baseline MeanPoolUNet3D: `{delta['baseline_params']}`",
        f"- Recency variant: `{delta['variant_params']}`",
        f"- Delta: `{delta['delta_params']}` (expected `{expected_extra}`)",
        "",
        "## TS=6 forward/backward",
        "",
        f"- Output shape: `{results['checks']['ts6']['output_shape']}`",
        f"- Loss finite: {results['checks']['ts6']['finite_loss']} "
        f"(loss={results['checks']['ts6']['loss']:.6f})",
        f"- Grad OK: {results['checks']['ts6']['grad_ok']}",
        f"- Init fusion weights: {results['checks']['uniform_init']['weights']}",
        "",
        f"Artifact: `{out_json}`",
        "",
    ]
    (out_dir / "SMOKE_TEST_RECENCY_FUSION.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"pass": results["pass"], "delta_params": delta["delta_params"]}, indent=2))
    print(f"Wrote {out_json}")
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
