"""Phase 3A/4B-compatible progression losses (Dice + Focal + weighted BCE)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, p, t):
        p = torch.sigmoid(p).view(-1)
        t = t.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 3.0, alpha: float = 0.85):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, p, t):
        bce = F.binary_cross_entropy_with_logits(p, t, reduction="none")
        prob = torch.sigmoid(p)
        pt = prob * t + (1 - prob) * (1 - t)
        at = self.alpha * t + (1 - self.alpha) * (1 - t)
        return (at * (1 - pt) ** self.gamma * bce).mean()


class PredLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        bce_weight: float = 0.2,
        focal_gamma: float = 3.0,
        focal_alpha: float = 0.85,
    ):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(focal_gamma, focal_alpha)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.register_buffer("pw", torch.tensor([float(pos_weight)]))

    def forward(self, p, t):
        return (
            self.dice_weight * self.dice(p, t)
            + self.focal_weight * self.focal(p, t)
            + self.bce_weight
            * F.binary_cross_entropy_with_logits(p, t, pos_weight=self.pw)
        )
