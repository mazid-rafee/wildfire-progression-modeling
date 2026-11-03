"""3D U-Net spatial backbone with equal temporal mean pooling.

Temporal operators that are *spatially* 3D but temporally separable:
- Conv3d / ConvTranspose3d kernels use temporal size 1
- MaxPool3d is (1, 2, 2) — spatial only
- SEBlock3D AdaptiveAvgPool3d(1) pools equally over T×H×W for channel squeeze
- ASPP global branch AdaptiveAvgPool3d((None, 1, 1)) pools H×W, keeps T

Equal temporal averaging that collapses history into a 2D map occurs at:
    x = x.mean(dim=2)   # after decoder, before 2D head

Upstream reference (read-only MIT-attributed reimplementation basis):
  DeepFire-Forecaster/phase3a_spase_fp_reproduction/phase3a_fp_model.py
  Copyright (c) 2026 Nikoloas Mavros — MIT License (see docs/ATTRIBUTION.md)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        mid = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(True),
            nn.Linear(mid, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1, 1)


class ResBlock3D(nn.Module):
    def __init__(self, ic: int, oc: int, se_reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv3d(ic, oc, (1, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(oc)
        self.conv2 = nn.Conv3d(oc, oc, (1, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(oc)
        self.se = SEBlock3D(oc, se_reduction)
        self.skip = nn.Conv3d(ic, oc, 1, bias=False) if ic != oc else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.skip(x)
        o = self.act(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        return self.act(self.se(o) + r)


class ASPP3D(nn.Module):
    def __init__(self, ic: int, oc: int, dils: tuple[int, ...] = (1, 6, 12)):
        super().__init__()
        bc = oc // len(dils)
        self.br = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        ic,
                        bc,
                        (1, 3, 3),
                        padding=(0, d, d),
                        dilation=(1, d, d),
                        bias=False,
                    ),
                    nn.BatchNorm3d(bc),
                    nn.GELU(),
                )
                for d in dils
            ]
        )
        self.gp = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(ic, bc, 1, bias=False),
            nn.BatchNorm3d(bc),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(bc * (len(dils) + 1), oc, 1, bias=False),
            nn.BatchNorm3d(oc),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [b(x) for b in self.br]
        g = self.gp(x).expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        parts.append(g)
        return self.fuse(torch.cat(parts, 1))


class MeanPoolUNet3D(nn.Module):
    """Spatial 3D U-Net baseline with equal temporal mean pooling."""

    def __init__(
        self,
        in_ch: int = 27,
        enc: tuple[int, ...] = (64, 128, 256, 512),
        bneck: int = 1024,
        se_reduction: int = 8,
    ):
        super().__init__()
        self.enc_channels = tuple(enc)
        self.inp = nn.Sequential(
            nn.Conv3d(in_ch, enc[0], (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(enc[0]),
            nn.GELU(),
        )
        self.encs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = enc[0]
        for c in enc:
            self.encs.append(ResBlock3D(prev, c, se_reduction))
            self.pools.append(nn.MaxPool3d((1, 2, 2)))
            prev = c
        self.bneck = ASPP3D(enc[-1], bneck)
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        prev = bneck
        for c in reversed(enc):
            self.ups.append(nn.ConvTranspose3d(prev, c, (1, 2, 2), stride=(1, 2, 2)))
            self.decs.append(ResBlock3D(c * 2, c, se_reduction))
            prev = c
        self.head = nn.Sequential(
            nn.Conv2d(enc[0], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode_decode_5d(self, x: torch.Tensor) -> torch.Tensor:
        """Return decoder features shaped (B, C, T, H, W) before temporal collapse."""
        x = self.inp(x)
        skips = []
        for enc, pool in zip(self.encs, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bneck(x)
        for up, dec, sk in zip(self.ups, self.decs, reversed(skips)):
            x = up(x)
            d = [sk.shape[i] - x.shape[i] for i in range(2, 5)]
            if any(di != 0 for di in d):
                x = F.pad(x, [0, d[2], 0, d[1], 0, d[0]])
            x = dec(torch.cat([x, sk], 1))
        return x

    def temporal_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Equal average over the temporal axis (dim=2)."""
        return x.mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encode_decode_5d(x)
        pooled = self.temporal_pool(feats)
        return self.head(pooled)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
