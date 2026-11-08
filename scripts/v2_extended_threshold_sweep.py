#!/usr/bin/env python3
"""Validation-only extended threshold sweep (0.01–0.99) via re-inference.

Frozen models: E1 RecencyFusion, Phase 4B TS=2, Phase 4B TS=6.
Does NOT change historical headline metrics or overwrite run directories.
Writes only under outputs/v2_protocol/threshold_extended/.

Leaves ENV-0 training untouched — pick a free GPU that is not ENV-0's GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.data.phase4b_bridge import (  # noqa: E402
    EXPECTED_POS_WEIGHT,
    build_matched_ctx,
    import_preload,
)
from wtp.eval.ba_eval import (  # noqa: E402
    MetaCollate,
    collect_val_logits,
    threshold_sweep_from_logits,
)
from wtp.models import RecencyFusionUNet3D  # noqa: E402
from wtp.utils.external import import_phase3a_modules  # noqa: E402
from wtp.utils.paths import load_paths  # noqa: E402

E1_RUN = ROOT / "runs" / "phase4c_e1_spase_recency_matched_ts6_seed42_20261102_185755"
OUT_DIR = ROOT / "outputs" / "v2_protocol" / "threshold_extended"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def _load_headline_thr(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_arm(
    name: str,
    model: torch.nn.Module,
    val_windows: list,
    dataset_cls,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    frozen_selected_thr_path: Path,
) -> dict:
    ds = dataset_cls(val_windows, augment=False)
    dl = DataLoader(
        ds,
        batch_size=min(batch_size, max(len(ds), 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=MetaCollate(keep_meta=False),
        pin_memory=device.type == "cuda",
    )
    t0 = time.time()
    logits, labels = collect_val_logits(model, dl, device, use_amp)
    sweep = threshold_sweep_from_logits(logits, labels, 0.01, 0.99, 0.01)
    frozen = _load_headline_thr(frozen_selected_thr_path)
    # Find F1 at historical 0.75 for reference (eval-only; do not overwrite)
    row_075 = next((r for r in sweep["rows"] if abs(r["thr"] - 0.75) < 1e-9), None)
    return {
        "arm": name,
        "n_val_windows": len(val_windows),
        "inference_sec": time.time() - t0,
        "logits_shape": list(logits.shape),
        "extended_sweep": sweep,
        "historical_headline_threshold": frozen.get("selected_threshold"),
        "historical_val_f1_at_selected": frozen.get("val_f1_at_selected"),
        "extended_f1_at_0.75": None if row_075 is None else row_075["f1"],
        "extended_best_thr": sweep["best_thr"],
        "extended_best_f1": sweep["best_f1"],
        "headline_metrics_unchanged": True,
        "note": (
            "Extended sweep is diagnostic only. Do not replace frozen selected_threshold "
            "or published test metrics."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physical-gpu", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-free-vram-gib", type=float, default=10.0)
    ap.add_argument(
        "--i-understand-reinference-only",
        action="store_true",
        help="Required: confirms no training / no ENV-0 interference.",
    )
    args = ap.parse_args()
    if not args.i_understand_reinference_only:
        print("Refusing: pass --i-understand-reinference-only", flush=True)
        return 2

    # Never touch ENV-0 GPU (typically 1)
    env0_meta = ROOT / "logs" / "env_ablation_ENV-0_ACTIVE.json"
    if env0_meta.exists():
        meta = json.loads(env0_meta.read_text(encoding="utf-8"))
        if int(meta.get("gpu", -1)) == int(args.physical_gpu):
            print(
                f"Refusing: physical GPU {args.physical_gpu} is ENV-0. "
                "Pick another GPU.",
                flush=True,
            )
            return 3

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    paths = load_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(0)
        free_gib = free / (1024**3)
        print(f"[vram] visible cuda:0 free={free_gib:.2f} GiB", flush=True)
        if free_gib < args.min_free_vram_gib:
            print("Insufficient free VRAM; aborting without affecting ENV-0.", flush=True)
            return 4

    ctx = build_matched_ctx(paths)
    preload, PreloadedMatchedDataset = import_preload(paths)
    print("=== Preload val TS=6 (shared for E1 / P4B-TS6) ===", flush=True)
    val6 = preload(
        ctx["val_rows"], ctx["fire_by_id"], 6, ctx["fp_mean"], ctx["fp_std"], "val_ts6"
    )
    print("=== Preload val TS=2 ===", flush=True)
    val2 = preload(
        ctx["val_rows"], ctx["fire_by_id"], 2, ctx["fp_mean"], ctx["fp_std"], "val_ts2"
    )

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "physical_gpu": args.physical_gpu,
        "device": str(device),
        "protocol": "V2-eval-only-extended-threshold",
        "headline_metrics_policy": "unchanged",
        "arms": {},
    }

    p3 = import_phase3a_modules(paths)
    SEUNet3DPred = p3["SEUNet3DPred"]

    # E1
    e1_ckpt = E1_RUN / "checkpoints" / "best_e1.pth"
    e1 = RecencyFusionUNet3D(
        in_ch=27, enc=(64, 128, 256, 512), bneck=1024, max_t=6, init_recency_scale=0.0
    )
    obj = torch.load(e1_ckpt, map_location="cpu", weights_only=False)
    e1.load_state_dict(obj["model_state_dict"])
    e1.to(device).eval()
    results["arms"]["E1"] = run_arm(
        "E1",
        e1,
        val6,
        PreloadedMatchedDataset,
        device,
        use_amp,
        args.batch_size,
        E1_RUN / "selected_threshold.json",
    )
    _write_json(OUT_DIR / "e1_val_thr_0.01_0.99.json", results["arms"]["E1"])
    del e1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Phase 4B TS6
    p4b = Path(paths["phase4b"]["run_dir"])
    m6 = SEUNet3DPred(in_ch=27)
    obj6 = torch.load(p4b / "ts6" / "checkpoints" / "best_matched.pth", map_location="cpu", weights_only=False)
    m6.load_state_dict(obj6["model_state_dict"])
    m6.to(device).eval()
    results["arms"]["Phase4B_TS6"] = run_arm(
        "Phase4B_TS6",
        m6,
        val6,
        PreloadedMatchedDataset,
        device,
        use_amp,
        args.batch_size,
        p4b / "ts6" / "selected_threshold.json",
    )
    _write_json(OUT_DIR / "p4b_ts6_val_thr_0.01_0.99.json", results["arms"]["Phase4B_TS6"])
    del m6
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Phase 4B TS2
    m2 = SEUNet3DPred(in_ch=27)
    obj2 = torch.load(p4b / "ts2" / "checkpoints" / "best_matched.pth", map_location="cpu", weights_only=False)
    m2.load_state_dict(obj2["model_state_dict"])
    m2.to(device).eval()
    results["arms"]["Phase4B_TS2"] = run_arm(
        "Phase4B_TS2",
        m2,
        val2,
        PreloadedMatchedDataset,
        device,
        use_amp,
        args.batch_size,
        p4b / "ts2" / "selected_threshold.json",
    )
    _write_json(OUT_DIR / "p4b_ts2_val_thr_0.01_0.99.json", results["arms"]["Phase4B_TS2"])

    # Compact summary (no full row dumps)
    summary = {
        "generated_at_utc": results["generated_at_utc"],
        "headline_metrics_unchanged": True,
        "arms": {
            k: {
                "historical_thr": v["historical_headline_threshold"],
                "historical_val_f1": v["historical_val_f1_at_selected"],
                "extended_best_thr": v["extended_best_thr"],
                "extended_best_f1": v["extended_best_f1"],
                "extended_f1_at_0.75": v["extended_f1_at_0.75"],
                "selected_is_grid_ceiling": v["extended_sweep"]["selected_is_grid_ceiling"],
            }
            for k, v in results["arms"].items()
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    _write_json(OUT_DIR / "full_results.json", results)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
