"""Frozen-uniform RecencyFusion ≡ MeanPool (prep for multi-seed control arm)."""

from __future__ import annotations

import torch

from wtp.data.protocol_loader import build_model_from_config
from wtp.models import MeanPoolUNet3D, RecencyFusionUNet3D
from wtp.models.recency_fusion import (
    backbone_state_dict,
    freeze_temporal_fusion_uniform,
    load_backbone_state_dict,
)


def test_freeze_uniform_matches_meanpool():
    torch.manual_seed(0)
    e1 = RecencyFusionUNet3D(in_ch=27, enc=(64, 128, 256, 512), bneck=1024, max_t=6)
    mp = MeanPoolUNet3D(in_ch=27, enc=(64, 128, 256, 512), bneck=1024)
    load_backbone_state_dict(mp, backbone_state_dict(e1))
    meta = freeze_temporal_fusion_uniform(e1)
    assert meta["frozen"] is True
    assert abs(sum(meta["weights"]) - 1.0) < 1e-6
    assert max(abs(w - 1.0 / 6) for w in meta["weights"]) < 1e-6
    assert meta["n_trainable_params"] == 24_279_561
    x = torch.randn(1, 27, 6, 32, 32)
    e1.eval()
    mp.eval()
    with torch.no_grad():
        a = e1(x)
        b = mp(x)
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-5)


def test_build_model_from_config_freeze_flag():
    cfg = {
        "name": "recency_fusion",
        "in_channels": 27,
        "enc": [64, 128, 256, 512],
        "bneck": 1024,
        "max_t": 6,
        "init_recency_scale": 0.0,
        "freeze_temporal_fusion": True,
        "freeze_temporal_mode": "uniform",
    }
    model = build_model_from_config(cfg, ts=6)
    assert not model.temporal_fusion.logits.requires_grad
    assert not model.temporal_fusion.recency_scale.requires_grad
    w = model.fusion_weights(6).tolist()
    assert max(abs(x - 1.0 / 6) for x in w) < 1e-6
