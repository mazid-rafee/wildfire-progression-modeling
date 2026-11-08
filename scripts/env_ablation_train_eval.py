#!/usr/bin/env python3
"""Phase 4C Experiment E1 — matched RecencyFusionUNet3D TS=6 train + BA eval.

Writes only under this project's run directory. Does not mutate DeepFire / DATA_ROOT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.data.channel_map import ABLATION_GROUPS  # noqa: E402
from wtp.data.channel_mask import (  # noqa: E402
    MaskedPreloadedDataset,
    apply_channel_mask_numpy,
    channels_for_ablation,
)
from wtp.data.phase4b_bridge import (  # noqa: E402
    EXPECTED_COUNTS,
    EXPECTED_POS_WEIGHT,
    build_matched_ctx,
    import_preload,
    verify_population_keys,
)
from wtp.eval.ba_eval import MetaCollate, evaluate_windows, threshold_sweep, validate_f1  # noqa: E402
from wtp.models import RecencyFusionUNet3D, count_params  # noqa: E402
from wtp.train.epoch import set_seed, train_epoch  # noqa: E402
from wtp.train.gate import require_train_approval  # noqa: E402
from wtp.train.losses import PredLoss  # noqa: E402
from wtp.utils.paths import load_paths  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def wait_for_cuda_free_gib(min_free_gib: float, poll_sec: int = 60) -> None:
    if not torch.cuda.is_available():
        return
    min_free = int(min_free_gib * (1024**3))
    while True:
        free, total = torch.cuda.mem_get_info(0)
        free_gib = free / (1024**3)
        print(
            f"[vram-gate] visible cuda:0 free={free_gib:.2f} GiB "
            f"(need>={min_free_gib:.1f}); total={total/(1024**3):.1f} GiB",
            flush=True,
        )
        if free >= min_free:
            time.sleep(15)
            free2, _ = torch.cuda.mem_get_info(0)
            if free2 >= min_free:
                print(f"[vram-gate] OK free={free2/(1024**3):.2f} GiB", flush=True)
                return
        time.sleep(poll_sec)


def load_frozen_phase4b_evals(paths: dict) -> dict:
    run = Path(paths["phase4b"]["run_dir"])
    out = {}
    for arm in ("ts2", "ts6"):
        out[arm] = {}
        for name in (
            "full_test24_ba_matched",
            "test18_ba_matched",
            "test16_ba_matched",
        ):
            p = run / arm / "evaluations" / f"{name}.json"
            out[arm][name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def write_comparison_report(run_dir: Path, summary: dict, frozen: dict) -> None:
    e1 = summary["evaluations"]
    lines = [
        "# Env ablation — comparison vs frozen Phase 4B / reference E1",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        f"**Run dir:** `{run_dir}`",
        f"**Physical GPU:** {summary.get('physical_gpu')}",
        f"**Seed:** 42 | **TS:** 6 | **Model:** recency_fusion | **Ablation:** {summary.get('ablation')}",
        "",
        "> Single-seed exploratory result. Do **not** claim statistical significance.",
        "",
        "## Training",
        "",
        f"- Best epoch: **{summary['best_epoch']}**",
        f"- Best val F1@0.5: **{summary['best_val_f1_at_0.5']:.6f}**",
        f"- Selected thr (val sweep): **{summary['selected_threshold']}**",
        f"- Params: **{summary['n_params']}**",
        f"- Train time (min): **{summary['train_elapsed_min']:.2f}**",
        f"- Peak GPU mem (GiB, allocated): **{summary.get('peak_gpu_mem_gib')}**",
        f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        "",
        "## Learned temporal weights (oldest→newest)",
        "",
        f"`{summary['learned_temporal_weights']}`",
        "",
        f"- Newer larger than older (w[5] > w[0]): "
        f"**{summary['newer_gt_older']}**",
        f"- Max−min weight mass: **{summary['weight_spread']:.6f}**",
        "",
        "## Test metrics vs frozen Phase 4B",
        "",
        "| Population | Metric | E1 | Phase4B TS=6 | Phase4B TS=2 | E1−TS6 | E1−TS2 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    pop_map = [
        ("full_test24_ba_matched", "Full-24"),
        ("test18_ba_matched", "Rule-A 18"),
        ("test16_ba_matched", "Manuscript 16"),
    ]
    for key, label in pop_map:
        for metric in ("f1", "iou", "precision", "recall"):
            a = e1[key][metric]
            b = frozen["ts6"][key][metric]
            c = frozen["ts2"][key][metric]
            lines.append(
                f"| {label} | {metric} | {a:.6f} | {b:.6f} | {c:.6f} | "
                f"{a-b:+.6f} | {a-c:+.6f} |"
            )
        for metric in ("tp", "fp", "fn", "n_windows"):
            a = e1[key][metric]
            b = frozen["ts6"][key][metric]
            c = frozen["ts2"][key][metric]
            lines.append(
                f"| {label} | {metric} | {a} | {b} | {c} | {a-b} | {a-c} |"
            )

    # Per-fire paired ΔF1 on manuscript-16 fires present in all three
    lines += [
        "",
        "## Paired per-fire ΔF1 (manuscript-16 fires present in E1 eval)",
        "",
        "| Fire | E1 F1 | TS6 F1 | TS2 F1 | E1−TS6 | E1−TS2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    pf_e1 = e1["test16_ba_matched"]["per_fire"]
    pf6 = frozen["ts6"]["test16_ba_matched"]["per_fire"]
    pf2 = frozen["ts2"]["test16_ba_matched"]["per_fire"]
    for fid in sorted(set(pf_e1) & set(pf6) & set(pf2)):
        a, b, c = pf_e1[fid]["f1"], pf6[fid]["f1"], pf2[fid]["f1"]
        lines.append(
            f"| {fid} | {a:.4f} | {b:.4f} | {c:.4f} | {a-b:+.4f} | {a-c:+.4f} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- E1 trains from scratch (seed 42); does not warm-start from Phase 4B TS=6.",
        "- Fusion is content-agnostic positional weighting, not spatiotemporal attention.",
        "- `recency_scale` is expressively redundant with unrestricted logits for fixed T=6.",
        "",
    ]
    (run_dir / "PHASE4C_E1_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-free-vram-gib", type=float, default=20.0)
    ap.add_argument("--physical-gpu", type=str, default=None)
    ap.add_argument("--i-understand-full-train", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="Eval-only from existing ckpt")
    ap.add_argument(
        "--ablation",
        required=True,
        choices=sorted(ABLATION_GROUPS.keys()),
        help="Channel ablation group id (ENV-0, ENV-WIND, ...)",
    )
    args = ap.parse_args()

    require_train_approval(
        enabled=True,
        i_understand_full_train=bool(args.i_understand_full_train or args.skip_train),
    )

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if not str(run_dir).startswith(str(ROOT.resolve())):
        raise SystemExit(f"Run dir must be inside project: {ROOT}")

    paths = load_paths()
    set_seed(42)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for E1 training")
    device = torch.device("cuda:0")  # physical GPU selected via CUDA_VISIBLE_DEVICES

    ctx = build_matched_ctx(paths)
    key_check = verify_population_keys(ctx, paths)
    _write_json(run_dir / "population_key_check.json", key_check)
    if not key_check["ok"]:
        raise SystemExit(f"Population key mismatch: {key_check}")

    counts = {
        "train": len(ctx["train_rows"]),
        "val": len(ctx["val_rows"]),
        "test24": len(ctx["test24_rows"]),
        "test18": len(ctx["test18_rows"]),
        "test16": len(ctx["test16_rows"]),
    }
    for k, exp in EXPECTED_COUNTS.items():
        if counts[k] != exp:
            raise SystemExit(f"Count {k}={counts[k]} != {exp}")
    if abs(float(ctx["pos_weight"]) - EXPECTED_POS_WEIGHT) > 1e-9:
        # We force EXPECTED_POS_WEIGHT in bridge; double-check
        print(
            f"WARN: using locked pos_weight={EXPECTED_POS_WEIGHT} "
            f"(ctx had {ctx['pos_weight']})",
            flush=True,
        )
    pos_weight = EXPECTED_POS_WEIGHT

    preload_matched_windows, PreloadedMatchedDataset = import_preload(paths)

    print("=== PRELOAD train+val float16 (test deferred) ===", flush=True)
    train_wins = preload_matched_windows(
        ctx["train_rows"],
        ctx["fire_by_id"],
        6,
        ctx["fp_mean"],
        ctx["fp_std"],
        split_name="train",
    )
    val_wins = preload_matched_windows(
        ctx["val_rows"],
        ctx["fire_by_id"],
        6,
        ctx["fp_mean"],
        ctx["fp_std"],
        split_name="val",
    )
    assert len(train_wins) == 983 and len(val_wins) == 135

    mask_channels = channels_for_ablation(args.ablation)
    print(f"Ablation={args.ablation} masked_channels={mask_channels}", flush=True)
    train_ds = MaskedPreloadedDataset(
        PreloadedMatchedDataset(train_wins, augment=True), mask_channels
    )
    val_ds = MaskedPreloadedDataset(
        PreloadedMatchedDataset(val_wins, augment=False), mask_channels
    )
    bs = args.batch_size
    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=MetaCollate(False),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=MetaCollate(False),
    )

    print(f"=== WAIT FOR >= {args.min_free_vram_gib:.1f} GiB FREE VRAM ===", flush=True)
    wait_for_cuda_free_gib(args.min_free_vram_gib)
    torch.cuda.empty_cache()

    # Re-seed after preload I/O so RNG for init/shuffle aligns more closely
    set_seed(42)
    model = RecencyFusionUNet3D(
        in_ch=27, enc=(64, 128, 256, 512), bneck=1024, max_t=6, init_recency_scale=0.0
    ).to(device)
    n_params = count_params(model)
    print(f"Ablation params={n_params} expected=24279568", flush=True)
    if n_params != 24_279_568:
        raise RuntimeError(f"Param count {n_params} != 24279568")
    init_w = model.fusion_weights(6).detach().cpu().tolist()
    print(f"Init fusion weights: {init_w}", flush=True)

    use_amp = True
    criterion = PredLoss(pos_weight=pos_weight).to(device)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "best.pth"

    meta = {
        "experiment": f"env_ablation_{args.ablation}_recency_matched_ts6_seed42",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "ts": 6,
        "seed": 42,
        "epochs": args.epochs,
        "batch_size": bs,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "pos_weight": pos_weight,
        "n_params": n_params,
        "n_train": 983,
        "n_val": 135,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": args.physical_gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "data_mode": "full_float16_preload_train_val_deferred_test",
        "grad_clip": 1.0,
        "init_fusion_weights": init_w,
        "warm_start_from_phase4b": False,
        "ablation": args.ablation,
        "masked_channels": mask_channels,
        "fp_stats_path": str(ctx["stats_path"]),
        "run_dir": str(run_dir),
        "pid": os.getpid(),
    }
    _write_json(run_dir / "launch_meta.json", meta)

    best_f1, best_ep = 0.0, 0
    hist = []
    t0 = time.time()

    if not args.skip_train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            epochs=args.epochs,
            steps_per_epoch=max(len(train_dl), 1),
            pct_start=0.1,
            anneal_strategy="cos",
        )
        scaler = GradScaler(enabled=use_amp)

        print(f"=== TRAINING {args.ablation} ===", flush=True)
        for ep in range(1, args.epochs + 1):
            te = time.time()
            tl, tf = train_epoch(
                model, train_dl, optimizer, scheduler, scaler, criterion, device, use_amp
            )
            vl, vf, vi = validate_f1(
                model, val_dl, criterion, device, use_amp, thr=0.5
            )
            lr = optimizer.param_groups[0]["lr"]
            mark = ""
            if vf > best_f1:
                best_f1, best_ep = vf, ep
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": ep,
                        "best_val_f1": best_f1,
                        "pos_weight": pos_weight,
                        "seed": 42,
                        "ts": 6,
                        "n_params": n_params,
                        "model_name": "recency_fusion",
                        "ablation": args.ablation,
                        "fusion_weights": model.fusion_weights(6).detach().cpu().tolist(),
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
                "lr": lr,
                "epoch_sec": time.time() - te,
                "is_best": bool(mark),
                "fusion_weights": model.fusion_weights(6).detach().cpu().tolist(),
            }
            hist.append(row)
            print(
                f"{ep:3d} tl={tl:.4f} tf={tf:.4f} vl={vl:.4f} "
                f"vf={vf:.4f} vi={vi:.4f} lr={lr:.6f} t={row['epoch_sec']:.0f}s "
                f"w={row['fusion_weights']}{mark}",
                flush=True,
            )

        if not ckpt_path.exists() or best_ep == 0:
            raise RuntimeError("Best checkpoint never written")

        with open(run_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
            # flatten fusion weights for CSV
            flat = []
            for r in hist:
                rr = {k: v for k, v in r.items() if k != "fusion_weights"}
                for i, wi in enumerate(r["fusion_weights"]):
                    rr[f"w{i}"] = wi
                flat.append(rr)
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)

        train_result = {
            "status": "TRAIN_OK",
            "best_epoch": best_ep,
            "best_val_f1_at_0.5": best_f1,
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": _sha256(ckpt_path),
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
    learned_w = model.fusion_weights(6).detach().cpu().tolist()

    print("=== VAL THRESHOLD SWEEP ===", flush=True)
    sweep = threshold_sweep(model, val_dl, criterion, device, use_amp)
    thr = float(sweep["best_thr"])
    _write_json(
        run_dir / "selected_threshold.json",
        {
            "selected_threshold": thr,
            "val_f1_at_selected": sweep["best_f1"],
            "checkpoint_epoch": best_ep,
            "sweep": sweep["rows"],
        },
    )

    print("=== PRELOAD + EVAL test sets ===", flush=True)
    test24 = preload_matched_windows(
        ctx["test24_rows"],
        ctx["fire_by_id"],
        6,
        ctx["fp_mean"],
        ctx["fp_std"],
        split_name="eval_test24",
    )
    keys18 = {r["target_key"] for r in ctx["test18_rows"]}
    keys16 = {r["target_key"] for r in ctx["test16_rows"]}
    def _mask_wins(wins):
        out = []
        for w in wins:
            nw = dict(w)
            nw["x"] = apply_channel_mask_numpy(w["x"], mask_channels)
            out.append(nw)
        return out

    test24_m = _mask_wins(test24)
    test_sets = {
        "full_test24_ba_matched": test24_m,
        "test18_ba_matched": [w for w in test24_m if w["target_key"] in keys18],
        "test16_ba_matched": [w for w in test24_m if w["target_key"] in keys16],
    }
    assert len(test_sets["full_test24_ba_matched"]) == 561
    assert len(test_sets["test18_ba_matched"]) == 553
    assert len(test_sets["test16_ba_matched"]) == 531

    eval_dir = run_dir / "evaluations"
    eval_dir.mkdir(exist_ok=True)
    evals = {}
    for name, wins in test_sets.items():
        print(f"Eval {name} n={len(wins)} thr={thr}", flush=True)
        res = evaluate_windows(
            model, wins, device, thr, use_amp, PreloadedMatchedDataset, bs
        )
        res.update(
            {
                "protocol": name,
                "ts": 6,
                "selected_threshold": thr,
                "checkpoint_epoch": best_ep,
                "model": "recency_fusion",
                "ablation": args.ablation,
            }
        )
        _write_json(eval_dir / f"{name}.json", res)
        evals[name] = res

    peak_mem = torch.cuda.max_memory_allocated(0) / (1024**3)
    summary = {
        "experiment": f"env_ablation_{args.ablation}",
        "best_epoch": best_ep,
        "best_val_f1_at_0.5": best_f1,
        "selected_threshold": thr,
        "n_params": n_params,
        "train_elapsed_min": (time.time() - t0) / 60.0
        if not args.skip_train
        else train_result.get("train_elapsed_min"),
        "peak_gpu_mem_gib": peak_mem,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": _sha256(ckpt_path),
        "ablation": args.ablation,
        "masked_channels": mask_channels,
        "learned_temporal_weights": learned_w,
        "newer_gt_older": bool(learned_w[5] > learned_w[0]),
        "weight_spread": float(max(learned_w) - min(learned_w)),
        "physical_gpu": args.physical_gpu or os.environ.get("CUDA_VISIBLE_DEVICES"),
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
    frozen = load_frozen_phase4b_evals(paths)
    write_comparison_report(run_dir, summary, frozen)
    print(f"Wrote {run_dir / 'PHASE4C_E1_COMPARISON.md'}", flush=True)
    print(f"{args.ablation} COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
