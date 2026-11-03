"""Train / validate epochs matching Phase 3A/4B mechanics."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dl, optimizer, scheduler, scaler, criterion, device, use_amp):
    model.train()
    tloss, tp, fp, fn = 0.0, 0.0, 0.0, 0.0
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            pred = model(xb)
            loss = criterion(pred, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        tloss += loss.item()
        with torch.no_grad():
            pb = (torch.sigmoid(pred) > 0.5).float()
            tp += (pb * yb).sum().item()
            fp += (pb * (1 - yb)).sum().item()
            fn += ((1 - pb) * yb).sum().item()
    n = max(len(dl), 1)
    return tloss / n, 2 * tp / (2 * tp + fp + fn + 1e-8)
