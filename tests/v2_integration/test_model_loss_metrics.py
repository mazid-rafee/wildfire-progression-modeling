"""CPU integration: model parity, loss finite, metrics formulas."""

from __future__ import annotations

import torch

from wtp.data.protocol_loader import build_model_from_config
from wtp.eval.ba_eval import threshold_sweep_from_logits
from wtp.models import MeanPoolUNet3D, RecencyFusionUNet3D, count_params
from wtp.train.losses import PredLoss


def test_meanpool_and_recency_param_delta():
    m = MeanPoolUNet3D(in_ch=27)
    r = RecencyFusionUNet3D(in_ch=27, max_t=6, init_recency_scale=0.0)
    # Recency adds fusion logits + scale
    assert count_params(r) - count_params(m) == 7
    assert count_params(r) == 24_279_568


def test_zero_init_recency_matches_meanpool_forward():
    torch.manual_seed(0)
    x = torch.randn(1, 27, 6, 32, 32)
    mean = MeanPoolUNet3D(in_ch=27)
    rec = RecencyFusionUNet3D(in_ch=27, max_t=6, init_recency_scale=0.0)
    # Copy overlapping weights
    mean_sd = mean.state_dict()
    rec_sd = rec.state_dict()
    for k, v in mean_sd.items():
        if k in rec_sd and rec_sd[k].shape == v.shape:
            rec_sd[k] = v
    rec.load_state_dict(rec_sd)
    mean.eval()
    rec.eval()
    with torch.no_grad():
        a = mean(x)
        b = rec(x)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_build_model_from_config_both_arms():
    m = build_model_from_config(
        {"name": "mean_pool", "in_channels": 27, "enc": [64, 128, 256, 512], "bneck": 1024},
        ts=2,
    )
    r = build_model_from_config(
        {
            "name": "recency_fusion",
            "in_channels": 27,
            "enc": [64, 128, 256, 512],
            "bneck": 1024,
            "max_t": 6,
            "init_recency_scale": 0.0,
        },
        ts=6,
    )
    x2 = torch.randn(1, 27, 2, 32, 32)
    x6 = torch.randn(1, 27, 6, 32, 32)
    y2 = m(x2)
    y6 = r(x6)
    assert y2.shape == (1, 1, 32, 32)
    assert y6.shape == (1, 1, 32, 32)


def test_loss_backward_finite():
    model = RecencyFusionUNet3D(in_ch=27, max_t=6)
    crit = PredLoss(pos_weight=112.87821151853076)
    x = torch.randn(2, 27, 6, 32, 32)
    y = torch.zeros(2, 1, 32, 32)
    y[:, :, 4:8, 4:8] = 1.0
    logits = model(x)
    loss = crit(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_extended_threshold_contract():
    logits = torch.full((1, 1, 8, 8), -5.0)
    labels = torch.zeros_like(logits)
    logits[0, 0, 0, 0] = 8.0
    labels[0, 0, 0, 0] = 1.0
    sweep = threshold_sweep_from_logits(logits, labels, 0.01, 0.99, 0.01)
    assert sweep["n_thresholds"] == 99
    assert sweep["rows"][0]["thr"] == 0.01
    assert abs(sweep["rows"][-1]["thr"] - 0.99) < 1e-9
