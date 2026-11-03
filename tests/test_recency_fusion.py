"""CPU unit tests for RecencyFusionUNet3D (no GPU, no dataset I/O)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from wtp.models import RecencyFusionUNet3D, MeanPoolUNet3D, available, build, count_params
from wtp.models.recency_fusion import RecencyAwareTemporalFusion


def test_registry_contains_recency():
    assert "recency_fusion" in available()
    assert "mean_pool" in available()
    m = build("recency_fusion", max_t=6)
    assert isinstance(m, RecencyFusionUNet3D)


def test_fusion_uniform_at_init():
    fuse = RecencyAwareTemporalFusion(max_t=6, init_recency_scale=0.0)
    w = fuse.weights(6, torch.device("cpu"), torch.float32)
    assert torch.allclose(w, torch.full((6,), 1.0 / 6.0), atol=1e-6)


def test_fusion_recency_bias_increases_newest():
    fuse = RecencyAwareTemporalFusion(max_t=6, init_recency_scale=2.0)
    w = fuse.weights(6, torch.device("cpu"), torch.float32)
    assert w[-1] > w[0]
    assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-6)


def test_forward_shapes_ts2_ts6():
    model = RecencyFusionUNet3D(max_t=6)
    model.eval()
    with torch.no_grad():
        for ts in (2, 6):
            x = torch.randn(1, 27, ts, 64, 64)
            y = model(x)
            assert y.shape == (1, 1, 64, 64)


def test_backward_reaches_fusion_params():
    model = RecencyFusionUNet3D(max_t=6)
    x = torch.randn(1, 27, 6, 64, 64)
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 10:20, 10:20] = 1.0
    loss = F.binary_cross_entropy_with_logits(model(x), target)
    loss.backward()
    assert model.temporal_fusion.logits.grad is not None
    assert torch.isfinite(model.temporal_fusion.logits.grad).all()
    assert model.temporal_fusion.recency_scale.grad is not None


def test_param_delta_is_seven():
    base = MeanPoolUNet3D()
    var = RecencyFusionUNet3D(max_t=6)
    assert count_params(var) - count_params(base) == 7


def test_mean_pool_baseline_param_count_matches_frozen_baseline():
    # Frozen matched mean-pool baseline EXPECTED["params"] == 24279561
    n = count_params(MeanPoolUNet3D())
    assert n == 24_279_561
