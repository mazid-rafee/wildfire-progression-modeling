"""BA-labelled micro metrics (Phase 4B-compatible aggregation)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader


class MetaCollate:
    def __init__(self, keep_meta: bool = False):
        self.keep_meta = keep_meta

    def __call__(self, batch):
        xs, ys, metas = zip(*batch)
        xb = torch.stack(xs, 0)
        yb = torch.stack(ys, 0)
        if self.keep_meta:
            return xb, yb, list(metas)
        return xb, yb


@torch.no_grad()
def evaluate_windows(
    model: torch.nn.Module,
    windows: Sequence[dict],
    device: torch.device,
    thr: float,
    use_amp: bool,
    dataset_cls,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Micro-aggregate TP/FP/FN over windows; also report per-fire F1/IoU."""
    ds = dataset_cls(windows, augment=False)
    dl = DataLoader(
        ds,
        batch_size=min(batch_size, max(len(ds), 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=MetaCollate(keep_meta=True),
    )
    fire_met: dict[str, dict[str, float]] = defaultdict(
        lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "n": 0}
    )
    model.eval()
    for xb, yb, metas in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        with autocast(enabled=use_amp and device.type == "cuda"):
            pred = model(xb)
        pb = (torch.sigmoid(pred) > thr).float()
        for j, key in enumerate(metas):
            fid = key.split("|")[1] if "|" in key else key
            fire_met[fid]["tp"] += (pb[j] * yb[j]).sum().item()
            fire_met[fid]["fp"] += (pb[j] * (1 - yb[j])).sum().item()
            fire_met[fid]["fn"] += ((1 - pb[j]) * yb[j]).sum().item()
            fire_met[fid]["n"] += 1

    per_fire = {}
    atp = afp = afn = 0.0
    for fid, m in sorted(fire_met.items()):
        f1 = 2 * m["tp"] / (2 * m["tp"] + m["fp"] + m["fn"] + 1e-8)
        iou = m["tp"] / (m["tp"] + m["fp"] + m["fn"] + 1e-8)
        per_fire[fid] = {"f1": f1, "iou": iou, **m}
        atp += m["tp"]
        afp += m["fp"]
        afn += m["fn"]
    return {
        "f1": 2 * atp / (2 * atp + afp + afn + 1e-8),
        "iou": atp / (atp + afp + afn + 1e-8),
        "precision": atp / (atp + afp + 1e-8),
        "recall": atp / (atp + afn + 1e-8),
        "tp": atp,
        "fp": afp,
        "fn": afn,
        "n_windows": len(windows),
        "n_fires": len(per_fire),
        "per_fire": per_fire,
    }


@torch.no_grad()
def validate_f1(
    model, dl, criterion, device, use_amp: bool, thr: float = 0.5
) -> tuple[float, float, float]:
    model.eval()
    tloss = tp = fp = fn = 0.0
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast(enabled=use_amp and device.type == "cuda"):
            pred = model(xb)
            loss = criterion(pred, yb)
        tloss += loss.item()
        pb = (torch.sigmoid(pred) > thr).float()
        tp += (pb * yb).sum().item()
        fp += (pb * (1 - yb)).sum().item()
        fn += ((1 - pb) * yb).sum().item()
    n = max(len(dl), 1)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return tloss / n, f1, iou


def threshold_sweep(model, val_dl, criterion, device, use_amp: bool) -> dict:
    import numpy as np

    best_thr, best_f1 = 0.5, 0.0
    rows = []
    for thr in np.arange(0.05, 0.80, 0.05):
        _, f1, iou = validate_f1(
            model, val_dl, criterion, device, use_amp, thr=float(thr)
        )
        rows.append({"thr": float(thr), "f1": float(f1), "iou": float(iou)})
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return {"best_thr": best_thr, "best_f1": best_f1, "rows": rows}


@torch.no_grad()
def collect_val_logits(
    model, dl, device, use_amp: bool
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Concatenate validation logits and labels for offline threshold analysis."""
    model.eval()
    logits_list = []
    labels_list = []
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        with autocast(enabled=use_amp and device.type == "cuda"):
            pred = model(xb)
        logits_list.append(pred.detach().float().cpu())
        labels_list.append(yb.float().cpu())
    return torch.cat(logits_list, 0), torch.cat(labels_list, 0)


def threshold_sweep_from_logits(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    thr_min: float = 0.01,
    thr_max: float = 0.99,
    thr_step: float = 0.01,
) -> dict:
    """Val-only extended sweep on cached logits. Does not alter historical headlines."""
    import numpy as np

    probs = torch.sigmoid(logits)
    y = labels
    best_thr, best_f1 = 0.5, -1.0
    rows = []
    thr_values = np.arange(thr_min, thr_max + 1e-12, thr_step)
    for thr in thr_values:
        pb = (probs > float(thr)).float()
        tp = (pb * y).sum().item()
        fp = (pb * (1 - y)).sum().item()
        fn = ((1 - pb) * y).sum().item()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        rows.append(
            {
                "thr": float(thr),
                "f1": float(f1),
                "iou": float(iou),
                "precision": float(prec),
                "recall": float(rec),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
            }
        )
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    selected_is_ceiling = abs(best_thr - thr_max) < 1e-9
    return {
        "best_thr": best_thr,
        "best_f1": best_f1,
        "thr_min": thr_min,
        "thr_max": thr_max,
        "thr_step": thr_step,
        "selected_is_grid_ceiling": selected_is_ceiling,
        "n_thresholds": len(rows),
        "rows": rows,
    }
