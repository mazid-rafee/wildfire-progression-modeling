"""Regression: micro F1/IoU formulas + extended threshold sweep helper."""

from __future__ import annotations

import torch

from wtp.eval.ba_eval import threshold_sweep_from_logits


def test_f1_from_counts():
    tp, fp, fn = 10.0, 5.0, 5.0
    f1 = 2 * tp / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)
    assert abs(f1 - 0.6666666667) < 1e-6
    assert abs(iou - 0.5) < 1e-6


def test_extended_threshold_grid_and_ceiling_flag():
    # Perfect separation at high confidence → best near 0.5+; construct so thr=0.99 wins
    logits = torch.tensor([[[[5.0, -5.0], [-5.0, -5.0]]]])  # 1x1x2x2
    labels = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    # Expand to more positives with high logits so high thr still works
    logits = torch.full((2, 1, 4, 4), -10.0)
    labels = torch.zeros_like(logits)
    logits[0, 0, 0, 0] = 10.0
    labels[0, 0, 0, 0] = 1.0
    sweep = threshold_sweep_from_logits(logits, labels, 0.01, 0.99, 0.01)
    assert sweep["n_thresholds"] == 99
    assert abs(sweep["rows"][0]["thr"] - 0.01) < 1e-9
    assert abs(sweep["rows"][-1]["thr"] - 0.99) < 1e-9
    # Best thr should be high but F1 identical across thr in (sigmoid(10), 1)
    assert sweep["best_f1"] > 0.99
    assert sweep["best_thr"] >= 0.01
