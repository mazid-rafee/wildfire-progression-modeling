#!/usr/bin/env python3
"""Unified V2 (or explicit V1) matched train/eval entrypoint.

Dry-run (default): CPU smoke only.
Full train: requires --no-dry-run --i-understand-full-train after explicit approval.

Does not mutate DeepFire / DATA_ROOT / V1 artifacts. Writes only under --run-dir
inside this project.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.data.channel_mask import apply_channel_mask_numpy  # noqa: E402
from wtp.data.protocol_loader import (  # noqa: E402
    FORECAST_CHANNELS,
    build_ctx_for_protocol,
    build_model_from_config,
    make_datasets,
)
from wtp.eval.ba_eval import (  # noqa: E402
    MetaCollate,
    collect_val_logits,
    evaluate_windows,
    threshold_sweep_from_logits,
    validate_f1,
)
from wtp.models import count_params  # noqa: E402
from wtp.protocol.config import ARMS, build_resolved_config, compare_train_locks  # noqa: E402
from wtp.train.epoch import set_seed, train_epoch  # noqa: E402
from wtp.train.gate import require_train_approval  # noqa: E402
from wtp.train.losses import PredLoss  # noqa: E402
from wtp.tracking.run_card import write_run_card  # noqa: E402
from wtp.utils.paths import load_paths  # noqa: E402


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_fingerprint() -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=str(ROOT), text=True).strip()
        except Exception as e:  # noqa: BLE001
            return f"<error:{e}>"

    return {
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "git_status_short": _run(["git", "status", "--short"]),
        "git_diff_stat": _run(["git", "diff", "--stat"]),
    }


def _resolve_protocol_loader_key(resolved: dict, forecast_policy: str) -> tuple[str, str]:
    protocol_key = resolved["protocol_key"]
    if protocol_key in ("v2_corrected",):
        protocol_key = "v2"
    elif protocol_key in ("v2_corrected_mask_gfs_forecast",):
        protocol_key = "v2_mask_gfs"

    if str(protocol_key).startswith("v2_mask") or forecast_policy == "mask_zero":
        forecast_policy = "mask_zero"
        protocol_loader_key = "v2"
    elif str(protocol_key).startswith("v1"):
        protocol_loader_key = "v1"
    else:
        protocol_loader_key = "v2"
    return protocol_loader_key, forecast_policy


def _arm_mask_channels(arm_cfg: dict) -> Optional[list[int]]:
    if arm_cfg.get("mask_channels"):
        return list(arm_cfg["mask_channels"])
    if arm_cfg.get("channel_ablation") == "ENV-0":
        return list(range(8, 26))
    return None


def cpu_smoke(
    model: torch.nn.Module,
    ds,
    *,
    pos_weight: float,
    batch_size: int = 2,
    do_backward: bool = True,
) -> dict:
    device = torch.device("cpu")
    model = model.to(device)
    model.train(do_backward)
    criterion = PredLoss(pos_weight=pos_weight)
    dl = DataLoader(
        ds,
        batch_size=min(batch_size, max(len(ds), 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=MetaCollate(False),
    )
    xb, yb = next(iter(dl))
    xb = xb.to(device)
    yb = yb.to(device)
    t0 = time.time()
    logits = model(xb)
    loss = criterion(logits, yb)
    grad_norm = None
    if do_backward:
        loss.backward()
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().norm().item())
        grad_norm = total
        model.zero_grad(set_to_none=True)
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        pb = (probs > 0.5).float()
        tp = (pb * yb).sum().item()
        fp = (pb * (1 - yb)).sum().item()
        fn = ((1 - pb) * yb).sum().item()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        # Masked-channel assertion if present
        mask_ok = True
        if hasattr(ds, "channels"):
            for c in ds.channels:
                if float(xb[0, c].abs().max()) > 0:
                    mask_ok = False
                    break
    return {
        "device": "cpu",
        "batch_x_shape": list(xb.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "finite_inputs": bool(torch.isfinite(xb).all().item()),
        "f1_at_0.5": float(f1),
        "backward": do_backward,
        "grad_norm_l2_sum": grad_norm,
        "mask_channels_zero_ok": mask_ok,
        "elapsed_sec": time.time() - t0,
        "optimizer_step_executed": False,
    }


def dry_run_arm(args) -> dict:
    paths = load_paths()
    resolved = build_resolved_config(
        args.arm,
        protocol_name=args.protocol,
        overrides={"forecast_policy": args.forecast_policy} if args.forecast_policy else None,
    )
    if args.forecast_policy:
        resolved["forecast_policy"] = args.forecast_policy

    forecast_policy = args.forecast_policy or resolved["forecast_policy"]
    protocol_loader_key, forecast_policy = _resolve_protocol_loader_key(
        resolved, forecast_policy
    )
    mask_channels = _arm_mask_channels(resolved["arm"])

    policy_tag = (forecast_policy or "default").replace("include_as_forecast", "forecast_incl")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else (
        ROOT
        / "outputs"
        / "v2_integration"
        / "dry_runs"
        / f"{args.arm}_{protocol_loader_key}_{policy_tag}"
    )
    if not str(run_dir).startswith(str(ROOT.resolve())):
        raise SystemExit(f"run_dir must be under project root: {ROOT}")
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(resolved["train"]["seed"]))
    ctx = build_ctx_for_protocol(protocol_loader_key, paths)
    counts = {
        "train": len(ctx["train_rows"]),
        "val": len(ctx["val_rows"]),
        "test24": len(ctx["test24_rows"]),
        "test18": len(ctx["test18_rows"]),
        "test16": len(ctx["test16_rows"]),
    }
    for k, exp in resolved["shared_train"]["expected_counts"].items():
        if counts[k] != exp:
            raise SystemExit(f"Count mismatch {k}={counts[k]} != {exp}")

    ts = int(resolved["ts_length"])
    print(
        f"[dry-run] arm={args.arm} protocol={protocol_loader_key} "
        f"forecast_policy={forecast_policy} ts={ts} mask={mask_channels}",
        flush=True,
    )

    n_train = int(args.smoke_windows)
    n_val = min(2, n_train)
    packs = make_datasets(
        ctx,
        ts=ts,
        protocol_key=protocol_loader_key,
        forecast_policy=forecast_policy,
        max_train=n_train,
        max_val=n_val,
        augment_train=False,
        verbose=True,
        mask_channels=mask_channels,
    )
    model = build_model_from_config(resolved["model"], ts)
    n_params = count_params(model)
    smoke = cpu_smoke(
        model,
        packs["train_ds"],
        pos_weight=float(resolved["train"]["pos_weight"]),
        batch_size=min(2, packs["n_train"]),
        do_backward=not args.forward_only,
    )

    thr_cfg = resolved["eval"]
    fake_logits = torch.zeros(1, 1, 4, 4)
    fake_labels = torch.zeros(1, 1, 4, 4)
    fake_labels[0, 0, 0, 0] = 1.0
    fake_logits[0, 0, 0, 0] = 5.0
    thr_probe = threshold_sweep_from_logits(
        fake_logits,
        fake_labels,
        thr_min=float(thr_cfg["threshold_min"]),
        thr_max=float(thr_cfg["threshold_max"]),
        thr_step=float(thr_cfg["threshold_step"]),
    )

    locks = compare_train_locks()
    meta = {
        "mode": "dry_run",
        "arm_id": args.arm,
        "protocol_key": protocol_loader_key,
        "protocol_id": resolved["protocol_id"],
        "preprocessing_version": resolved["preprocessing_version"],
        "config_sha256": resolved["config_sha256"],
        "shared_train_lock_id": resolved["shared_train_lock_id"],
        "forecast_policy": forecast_policy,
        "mask_channels": mask_channels,
        "applied_masks": packs.get("applied_masks"),
        "ts_length": ts,
        "model": resolved["model"],
        "n_params": n_params,
        "population_counts": counts,
        "smoke": smoke,
        "threshold_grid": {
            "min": thr_cfg["threshold_min"],
            "max": thr_cfg["threshold_max"],
            "step": thr_cfg["threshold_step"],
            "n_thresholds": thr_probe["n_thresholds"],
            "tune_on_test": thr_cfg.get("tune_on_test", False),
        },
        "train_locks_ok": locks["ok"],
        "optimizer_updates_executed": False,
        "gpu_used": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_fingerprint(),
    }
    _write_json(run_dir / "dry_run_meta.json", meta)
    write_run_card(
        run_dir,
        {
            "experiment_id": args.arm,
            "mode": "dry_run",
            "config_sha256": resolved["config_sha256"],
            "preprocessing_version": resolved["preprocessing_version"],
            "smoke_ok": smoke["finite_loss"],
        },
    )
    print(
        json.dumps(
            {"arm": args.arm, "smoke_ok": smoke["finite_loss"], "run_dir": str(run_dir)},
            indent=2,
        ),
        flush=True,
    )
    return meta


class _EvalDatasetAdapter(Dataset):
    """Adapter so evaluate_windows can construct V2 datasets with kwargs."""

    def __init__(self, windows, augment: bool = False, **kwargs):
        from wtp.data.v2.dataset import PreloadedMatchedDatasetV2

        self.inner = PreloadedMatchedDatasetV2(
            windows,
            augment=False,
            fp_mean=kwargs.get("fp_mean"),
            fp_std=kwargs.get("fp_std"),
            correct_directional=True,
            seed=42,
        )
        self.mask_channels = list(kwargs.get("mask_channels") or [])
        self.forecast_policy = kwargs.get("forecast_policy")

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        x, y, meta = self.inner[idx]
        if self.mask_channels:
            x = x.clone()
            for c in self.mask_channels:
                x[c] = 0.0
        if self.forecast_policy == "mask_zero":
            x = x.clone()
            for c in FORECAST_CHANNELS:
                x[c] = 0.0
        return x, y, meta


def full_train_arm(args) -> int:
    require_train_approval(enabled=True, i_understand_full_train=True)

    paths = load_paths()
    resolved = build_resolved_config(
        args.arm,
        protocol_name=args.protocol,
        overrides={
            "forecast_policy": args.forecast_policy or "mask_zero",
            "train_enabled": True,
        },
    )
    forecast_policy = args.forecast_policy or "mask_zero"
    protocol_loader_key, forecast_policy = _resolve_protocol_loader_key(
        resolved, forecast_policy
    )
    mask_channels = _arm_mask_channels(resolved["arm"])
    # Primary experiments: force mask_zero unless ENV-0 already covers GFS
    if forecast_policy != "mask_zero" and not mask_channels:
        raise SystemExit("Primary V2 launch requires --forecast-policy mask_zero")

    ts = int(resolved["ts_length"])
    train_cfg = resolved["train"]
    # Seed: CLI override for multi-seed; shared_train.yaml remains the lock template.
    seed = int(args.seed if args.seed is not None else train_cfg["seed"])
    epochs = int(args.epochs or train_cfg["epochs"])
    bs = int(args.batch_size or train_cfg["batch_size"])
    pos_weight = float(train_cfg["pos_weight"])
    lr = float(train_cfg["lr"])
    wd = float(train_cfg["weight_decay"])
    use_amp = bool(train_cfg["amp"])
    drop_last = bool(train_cfg["drop_last"])
    grad_clip = float(train_cfg["grad_clip"])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    default_name = {
        "V2-P4B-TS2": f"v2_matched_spase_ts2_seed{seed}_{stamp}",
        "V2-P4B-TS6": f"v2_matched_spase_ts6_seed{seed}_{stamp}",
        "V2-E1": f"v2_matched_e1_ts6_gfs_mask_seed{seed}_{stamp}",
        "V2-E1-TS6": f"v2_matched_e1_ts6_gfs_mask_seed{seed}_{stamp}",
        "V2-ENV0-TS6": f"v2_matched_env0_ts6_seed{seed}_{stamp}",
        "V2-E1-UNIFORM-TS6": f"v2_matched_e1_uniform_ts6_seed{seed}_{stamp}",
        "V2-MEANPOOL-TS6-CTRL": f"v2_matched_meanpool_ts6_ctrl_seed{seed}_{stamp}",
    }.get(args.arm, f"v2_{args.arm}_seed{seed}_{stamp}")

    run_dir = Path(args.run_dir).resolve() if args.run_dir else (ROOT / "runs" / default_name)
    if not str(run_dir).startswith(str(ROOT.resolve())):
        raise SystemExit(f"run_dir must be under project: {ROOT}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for full V2 training")
    device = torch.device("cuda:0")
    set_seed(seed)

    ctx = build_ctx_for_protocol(protocol_loader_key, paths)
    counts = {
        "train": len(ctx["train_rows"]),
        "val": len(ctx["val_rows"]),
        "test24": len(ctx["test24_rows"]),
        "test18": len(ctx["test18_rows"]),
        "test16": len(ctx["test16_rows"]),
    }
    for k, exp in resolved["shared_train"]["expected_counts"].items():
        if counts[k] != exp:
            raise SystemExit(f"Count mismatch {k}={counts[k]} != {exp}")

    # Disjointness checks
    def fire_ids(rows):
        return {r["fire_id"] for r in rows}

    def keys(rows):
        return {r["target_key"] for r in rows}

    tr, va, te = fire_ids(ctx["train_rows"]), fire_ids(ctx["val_rows"]), fire_ids(ctx["test24_rows"])
    if tr & va or tr & te or va & te:
        raise SystemExit(f"Fire-id overlap: {tr&va} {tr&te} {va&te}")
    tk, vk, ek = keys(ctx["train_rows"]), keys(ctx["val_rows"]), keys(ctx["test24_rows"])
    if tk & vk or tk & ek or vk & ek:
        raise SystemExit("Target-key overlap across splits")

    stats_path = Path(ctx["stats_path"])
    stats_sha = _sha256_file(stats_path)

    print(
        f"=== V2 FULL TRAIN arm={args.arm} ts={ts} forecast={forecast_policy} "
        f"mask={mask_channels} gpu_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} ===",
        flush=True,
    )
    print(f"run_dir={run_dir}", flush=True)
    print(f"population={counts}", flush=True)
    print(f"fp_stats={stats_path} sha256={stats_sha[:16]}…", flush=True)

    # Optional preload stagger
    if args.preload_delay_sec > 0:
        print(f"Stagger sleep {args.preload_delay_sec}s before preload…", flush=True)
        time.sleep(args.preload_delay_sec)

    packs = make_datasets(
        ctx,
        ts=ts,
        protocol_key=protocol_loader_key,
        forecast_policy=forecast_policy,
        max_train=None,
        max_val=None,
        augment_train=True,
        verbose=True,
        mask_channels=mask_channels,
        aug_seed=seed,
    )
    assert packs["n_train"] == 983 and packs["n_val"] == 135

    # Assert masked channels zero on a train batch
    x0, y0, _ = packs["train_ds"][0]
    assert torch.isfinite(x0).all(), "non-finite inputs"
    check_ch = list(mask_channels or [])
    if forecast_policy == "mask_zero":
        check_ch = sorted(set(check_ch) | set(FORECAST_CHANNELS))
    for c in check_ch:
        if float(x0[c].abs().max()) > 0:
            raise RuntimeError(f"Masked channel {c} not zero after preprocessing/aug")

    train_dl = DataLoader(
        packs["train_ds"],
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=MetaCollate(False),
    )
    val_dl = DataLoader(
        packs["val_ds"],
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=MetaCollate(False),
    )

    set_seed(seed)
    model = build_model_from_config(resolved["model"], ts).to(device)

    # Optional paired-init: export/load backbone (+head) state for E1↔uniform pairs.
    from wtp.models.recency_fusion import backbone_state_dict, load_backbone_state_dict

    freeze_meta = getattr(model, "_freeze_temporal_meta", None)
    if args.load_init_backbone:
        init_path = Path(args.load_init_backbone).resolve()
        obj_init = torch.load(init_path, map_location="cpu", weights_only=False)
        bb = obj_init["backbone_state_dict"] if isinstance(obj_init, dict) and "backbone_state_dict" in obj_init else obj_init
        missing = load_backbone_state_dict(model, bb)
        if missing:
            raise RuntimeError(f"load_init_backbone missing keys: {missing[:8]}…")
        # Re-assert freeze after load (state may have overwritten fusion if full sd)
        if resolved["model"].get("freeze_temporal_fusion"):
            from wtp.models.recency_fusion import freeze_temporal_fusion_uniform

            freeze_meta = freeze_temporal_fusion_uniform(model)
            model._freeze_temporal_meta = freeze_meta  # type: ignore[attr-defined]
        print(f"Loaded paired init backbone from {init_path}", flush=True)
    if args.export_init_backbone:
        exp_path = Path(args.export_init_backbone).resolve()
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone_state_dict": backbone_state_dict(model),
                "seed": seed,
                "arm": args.arm,
                "ts": ts,
                "model_name": resolved["model"]["name"],
                "paired_init_group": resolved["arm"].get("paired_init_group"),
            },
            exp_path,
        )
        print(f"Exported init backbone → {exp_path}", flush=True)

    n_params = count_params(model)
    print(f"n_params={n_params}", flush=True)
    if resolved["model"]["name"] in ("recency_fusion", "e1", "recency"):
        expected = 24_279_561 if freeze_meta else 24_279_568
        if n_params != expected:
            raise RuntimeError(
                f"RecencyFusion trainable params {n_params} != {expected} "
                f"(freeze={bool(freeze_meta)})"
            )
        init_w = model.fusion_weights(ts).detach().cpu().tolist()
        print(f"Init fusion weights: {init_w}", flush=True)
        if freeze_meta:
            print(f"Frozen temporal fusion: {freeze_meta}", flush=True)
    else:
        init_w = None

    criterion = PredLoss(pos_weight=pos_weight).to(device)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "best.pth"

    launch_meta = {
        "experiment": args.arm,
        "mode": "full_train",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": args.physical_gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "protocol_key": protocol_loader_key,
        "protocol_id": resolved["protocol_id"],
        "preprocessing_version": resolved["preprocessing_version"],
        "forecast_policy": forecast_policy,
        "mask_channels": mask_channels,
        "applied_masks": packs.get("applied_masks"),
        "ts": ts,
        "seed": seed,
        "epochs": epochs,
        "batch_size": bs,
        "lr": lr,
        "weight_decay": wd,
        "pos_weight": pos_weight,
        "amp": use_amp,
        "drop_last": drop_last,
        "grad_clip": grad_clip,
        "n_params": n_params,
        "population_counts": counts,
        "fp_stats_path": str(stats_path),
        "fp_stats_sha256": stats_sha,
        "config_sha256": resolved["config_sha256"],
        "shared_train_lock_id": resolved["shared_train_lock_id"],
        "init_fusion_weights": init_w,
        "freeze_temporal_meta": freeze_meta,
        "paired_init_group": resolved["arm"].get("paired_init_group"),
        "export_init_backbone": args.export_init_backbone,
        "load_init_backbone": args.load_init_backbone,
        "git": _git_fingerprint(),
        "run_dir": str(run_dir),
        "resolved_train": {**train_cfg, "seed": seed},
        "model": resolved["model"],
    }
    _write_json(run_dir / "launch_meta.json", launch_meta)
    _write_json(run_dir / "resolved_config.json", {
        "arm": resolved["arm"],
        "protocol_id": resolved["protocol_id"],
        "train": resolved["train"],
        "eval": resolved["eval"],
        "data": resolved["data"],
        "forecast_policy": forecast_policy,
        "mask_channels": mask_channels,
        "config_sha256": resolved["config_sha256"],
    })
    write_run_card(
        run_dir,
        {
            "experiment_id": args.arm,
            "mode": "full_train",
            "config_sha256": resolved["config_sha256"],
            "preprocessing_version": resolved["preprocessing_version"],
            "forecast_policy": forecast_policy,
        },
    )

    best_f1, best_ep = 0.0, 0
    hist = []
    t0 = time.time()

    if not args.skip_train:
        # Only trainable params (frozen fusion excluded from AdamW)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=wd,
        )
        shared = resolved["shared_train"]
        oc = shared.get("one_cycle") or {}
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=max(len(train_dl), 1),
            pct_start=float(oc.get("pct_start", 0.1)),
            anneal_strategy=oc.get("anneal_strategy", "cos"),
        )
        scaler = GradScaler(enabled=use_amp)

        print("=== TRAINING ===", flush=True)
        for ep in range(1, epochs + 1):
            te = time.time()
            tl, tf = train_epoch(
                model, train_dl, optimizer, scheduler, scaler, criterion, device, use_amp
            )
            # enforce grad clip already inside train_epoch? check
            vl, vf, vi = validate_f1(model, val_dl, criterion, device, use_amp, thr=0.5)
            cur_lr = optimizer.param_groups[0]["lr"]
            mark = ""
            fw = None
            if hasattr(model, "fusion_weights"):
                fw = model.fusion_weights(ts).detach().cpu().tolist()
            if vf > best_f1:
                best_f1, best_ep = vf, ep
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": ep,
                        "best_val_f1": best_f1,
                        "pos_weight": pos_weight,
                        "seed": seed,
                        "ts": ts,
                        "n_params": n_params,
                        "model_name": resolved["model"]["name"],
                        "arm": args.arm,
                        "forecast_policy": forecast_policy,
                        "mask_channels": mask_channels,
                        "fusion_weights": fw,
                        "protocol_id": resolved["protocol_id"],
                        "config_sha256": resolved["config_sha256"],
                    },
                    ckpt_path,
                )
                mark = " *"
            row = {
                "epoch": ep,
                "train_loss": tl,
                "train_f1": tf,
                "val_loss": vl,
                "val_f1": vf,
                "val_iou": vi,
                "lr": cur_lr,
                "epoch_sec": time.time() - te,
                "is_best": bool(mark),
                "fusion_weights": fw,
            }
            hist.append(row)
            print(
                f"{ep:3d} tl={tl:.4f} tf={tf:.4f} vl={vl:.4f} "
                f"vf={vf:.4f} vi={vi:.4f} lr={cur_lr:.6f} t={row['epoch_sec']:.0f}s"
                f"{' w='+str(fw) if fw else ''}{mark}",
                flush=True,
            )

        if not ckpt_path.exists() or best_ep == 0:
            raise RuntimeError("Best checkpoint never written")

        flat = []
        for r in hist:
            rr = {k: v for k, v in r.items() if k != "fusion_weights"}
            if r["fusion_weights"]:
                for i, wi in enumerate(r["fusion_weights"]):
                    rr[f"w{i}"] = wi
            flat.append(rr)
        with open(run_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)

        train_result = {
            "status": "TRAIN_OK",
            "best_epoch": best_ep,
            "best_val_f1_at_0.5": best_f1,
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": _sha256_file(ckpt_path),
            "n_params": n_params,
            "pos_weight": pos_weight,
            "train_elapsed_min": (time.time() - t0) / 60.0,
            "n_train": 983,
            "n_val": 135,
            "batch_size": bs,
        }
        _write_json(run_dir / "train_result.json", train_result)
    else:
        if not ckpt_path.exists():
            raise SystemExit(f"Missing checkpoint for --skip-train: {ckpt_path}")
        train_result = json.loads((run_dir / "train_result.json").read_text())
        best_ep = train_result["best_epoch"]
        best_f1 = train_result["best_val_f1_at_0.5"]

    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(obj["model_state_dict"])
    learned_w = None
    if hasattr(model, "fusion_weights"):
        learned_w = model.fusion_weights(ts).detach().cpu().tolist()

    print("=== VAL THRESHOLD SWEEP 0.01–0.99 ===", flush=True)
    logits, labels = collect_val_logits(model, val_dl, device, use_amp)
    sweep = threshold_sweep_from_logits(
        logits,
        labels,
        thr_min=float(resolved["eval"]["threshold_min"]),
        thr_max=float(resolved["eval"]["threshold_max"]),
        thr_step=float(resolved["eval"]["threshold_step"]),
    )
    thr = float(sweep["best_thr"])
    _write_json(
        run_dir / "selected_threshold.json",
        {
            "selected_threshold": thr,
            "val_f1_at_selected": sweep["best_f1"],
            "checkpoint_epoch": best_ep,
            "selected_is_grid_ceiling": sweep["selected_is_grid_ceiling"],
            "n_thresholds": sweep["n_thresholds"],
            "sweep": sweep["rows"],
            "source": "validation_only",
            "tune_on_test": False,
        },
    )

    print("=== PRELOAD + EVAL test sets ===", flush=True)
    from wtp.data.v2.dataset import preload_matched_windows_v2

    test24 = preload_matched_windows_v2(
        ctx["test24_rows"],
        ctx["fire_by_id"],
        ts,
        ctx["fp_mean"],
        ctx["fp_std"],
        split_name="eval_test24",
        verbose=True,
    )
    keys18 = {r["target_key"] for r in ctx["test18_rows"]}
    keys16 = {r["target_key"] for r in ctx["test16_rows"]}

    def _apply_masks_numpy(wins):
        out = []
        for w in wins:
            nw = dict(w)
            x = w["x"].astype(np.float32)
            if mask_channels:
                x = apply_channel_mask_numpy(x, mask_channels, 0.0)
            if forecast_policy == "mask_zero":
                for c in FORECAST_CHANNELS:
                    x[c] = 0.0
            nw["x"] = x.astype(np.float16)
            out.append(nw)
        return out

    test24_m = _apply_masks_numpy(test24)
    test_sets = {
        "full_test24_ba_matched": test24_m,
        "test18_ba_matched": [w for w in test24_m if w["target_key"] in keys18],
        "test16_ba_matched": [w for w in test24_m if w["target_key"] in keys16],
    }
    assert len(test_sets["full_test24_ba_matched"]) == 561
    assert len(test_sets["test18_ba_matched"]) == 553
    assert len(test_sets["test16_ba_matched"]) == 531

    # Factory for evaluate_windows
    def _ds_factory(windows, augment=False):
        return _EvalDatasetAdapter(
            windows,
            augment=False,
            fp_mean=ctx["fp_mean"],
            fp_std=ctx["fp_std"],
            mask_channels=mask_channels,
            # already applied in numpy for test windows; keep adapter clean
            forecast_policy=None,
        )

    eval_dir = run_dir / "evaluations"
    eval_dir.mkdir(exist_ok=True)
    evals = {}
    for name, wins in test_sets.items():
        print(f"Eval {name} n={len(wins)} thr={thr}", flush=True)
        res = evaluate_windows(model, wins, device, thr, use_amp, _ds_factory, bs)
        res.update(
            {
                "protocol": name,
                "ts": ts,
                "selected_threshold": thr,
                "checkpoint_epoch": best_ep,
                "arm": args.arm,
                "forecast_policy": forecast_policy,
                "mask_channels": mask_channels,
            }
        )
        _write_json(eval_dir / f"{name}.json", res)
        evals[name] = res

    peak_mem = torch.cuda.max_memory_allocated(0) / (1024**3)
    summary = {
        "experiment": args.arm,
        "best_epoch": best_ep,
        "best_val_f1_at_0.5": best_f1,
        "selected_threshold": thr,
        "n_params": n_params,
        "train_elapsed_min": (time.time() - t0) / 60.0
        if not args.skip_train
        else train_result.get("train_elapsed_min"),
        "peak_gpu_mem_gib": peak_mem,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": _sha256_file(ckpt_path),
        "forecast_policy": forecast_policy,
        "mask_channels": mask_channels,
        "learned_temporal_weights": learned_w,
        "physical_gpu": args.physical_gpu or os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol_id": resolved["protocol_id"],
        "fp_stats_sha256": stats_sha,
        "config_sha256": resolved["config_sha256"],
        "population_counts": counts,
        "evaluations": {
            k: {
                m: evals[k][m]
                for m in (
                    "f1",
                    "iou",
                    "precision",
                    "recall",
                    "tp",
                    "fp",
                    "fn",
                    "n_windows",
                    "n_fires",
                    "per_fire",
                )
            }
            for k in evals
        },
    }
    _write_json(run_dir / "run_summary.json", summary)
    print(f"COMPLETE {args.arm} → {run_dir}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="V2 matched train/eval")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS.keys()))
    ap.add_argument("--protocol", default=None, help="v1|v2|v2_mask_gfs")
    ap.add_argument(
        "--forecast-policy",
        default=None,
        choices=["include_as_forecast", "mask_zero"],
    )
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--no-dry-run", action="store_true")
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--smoke-windows", type=int, default=2)
    ap.add_argument("--i-understand-full-train", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--physical-gpu", type=str, default=None)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override shared_train seed for multi-seed runs (locks otherwise unchanged)",
    )
    ap.add_argument(
        "--export-init-backbone",
        type=str,
        default=None,
        help="Write backbone+head init state after construction (for E1↔uniform pairing)",
    )
    ap.add_argument(
        "--load-init-backbone",
        type=str,
        default=None,
        help="Load backbone+head init state before training (paired-init consumer)",
    )
    ap.add_argument(
        "--preload-delay-sec",
        type=int,
        default=0,
        help="Stagger expensive float16 preload across concurrent jobs",
    )
    ap.add_argument("--min-free-vram-gib", type=float, default=14.0)
    args = ap.parse_args()

    if args.no_dry_run:
        if not args.i_understand_full_train and not args.skip_train:
            print("Refusing full train without --i-understand-full-train", flush=True)
            return 2
        # Optional VRAM gate
        if torch.cuda.is_available() and args.min_free_vram_gib > 0:
            free, total = torch.cuda.mem_get_info(0)
            free_gib = free / (1024**3)
            print(
                f"[vram] free={free_gib:.2f} GiB total={total/(1024**3):.1f} "
                f"(need>={args.min_free_vram_gib})",
                flush=True,
            )
            if free_gib < args.min_free_vram_gib:
                print("Insufficient free VRAM; refusing launch.", flush=True)
                return 3
        return full_train_arm(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    meta = dry_run_arm(args)
    return 0 if meta["smoke"]["finite_loss"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
