#!/usr/bin/env python3
"""Prepare scientifically corrected V2 data/eval protocol (no retraining).

Preserves all Phase 3A / 4B / E1 / ENV artifacts. Does not interrupt ENV-0.
Does not modify DeepFire, DATA_ROOT, or Git state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.data.phase4b_bridge import build_matched_ctx, load_fp_stats  # noqa: E402
from wtp.data.v2.constants import (  # noqa: E402
    DIRECTIONAL_MODEL_CHANNELS,
    EXPECTED_COUNTS,
    GEO_MISMATCH_TRAIN_FIRES,
    PROTOCOL_VERSION,
)
from wtp.data.v2.geo_align import alignment_report  # noqa: E402
from wtp.data.v2.normalize import (  # noqa: E402
    compare_to_v1,
    compute_matched_train_fp_stats,
    save_fp_stats,
)
from wtp.data.v2.population import probe_v2_buildable, verify_matched_population  # noqa: E402
from wtp.data.v2.timestamps import (  # noqa: E402
    feature_availability_table,
    summarize_availability,
    window_temporal_contract,
)
from wtp.utils.paths import load_paths  # noqa: E402

OUT = ROOT / "outputs" / "v2_protocol"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def check_env0_and_queues() -> dict:
    """Report ENV-0 status; note that no ablation queue exists to cancel."""
    pid_path = ROOT / "logs" / "env_ablation_ENV-0_ACTIVE.pid"
    meta_path = ROOT / "logs" / "env_ablation_ENV-0_ACTIVE.json"
    status = {
        "env0_left_running": True,
        "queued_ablations_cancelled": False,
        "queued_ablations_found": False,
        "limitation": None,
    }
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        status["env0_meta"] = meta
        pid = int(meta.get("pid") or (pid_path.read_text().strip() if pid_path.exists() else -1))
        running = False
        try:
            os.kill(pid, 0)
            running = True
        except OSError:
            running = False
        status["env0_pid"] = pid
        status["env0_alive"] = running
    # No scheduler/queue files present in this project — only ENV-0 ACTIVE markers
    queue_like = list((ROOT / "logs").glob("*queue*")) + list(
        (ROOT / "outputs" / "env_ablation").glob("*queue*")
    )
    status["queue_files"] = [str(p) for p in queue_like]
    if not queue_like:
        status["limitation"] = (
            "No additional environmental ablations are queued. Only ENV-0 is active. "
            "Nothing to prevent from starting without affecting ENV-0."
        )
    return status


def geo_probes(ctx: dict, n_ok: int = 2) -> dict:
    """Alignment diagnostics for mismatch fires + matched-grid controls."""
    probes = []
    # Mismatch train fires
    for fid in GEO_MISMATCH_TRAIN_FIRES:
        info = ctx["fire_by_id"].get(fid)
        if not info:
            probes.append({"fire_id": fid, "error": "not_in_fire_by_id"})
            continue
        day = info["day_files"][len(info["day_files"]) // 2]
        fp = day.replace("_VIIRS_Day", "_FirePred").replace("/VIIRS_Day/", "/FirePred/")
        # day paths are absolute under fire_dir/VIIRS_Day
        fp_path = str(Path(info["fire_dir"]) / "FirePred" / Path(day).name.replace("_VIIRS_Day", "_FirePred"))
        try:
            rep = alignment_report(day, fp_path)
            rep["fire_id"] = fid
            rep["split"] = "train"
            rep["expected_mismatch"] = True
            probes.append(rep)
        except Exception as e:  # noqa: BLE001
            probes.append({"fire_id": fid, "error": str(e)})

    # Control: first val fire (grids should match per audit)
    if ctx["val_rows"]:
        r = ctx["val_rows"][0]
        info = ctx["fire_by_id"][r["fire_id"]]
        day = info["day_files"][int(r["label_idx"]) - 1]
        fp_path = str(
            Path(info["fire_dir"])
            / "FirePred"
            / Path(day).name.replace("_VIIRS_Day", "_FirePred")
        )
        try:
            rep = alignment_report(day, fp_path)
            rep["fire_id"] = r["fire_id"]
            rep["split"] = "val"
            rep["expected_mismatch"] = False
            probes.append(rep)
        except Exception as e:  # noqa: BLE001
            probes.append({"fire_id": r["fire_id"], "error": str(e)})

    n_mismatch = sum(
        1
        for p in probes
        if p.get("expected_mismatch") and not p.get("same_grid", True) and "error" not in p
    )
    return {
        "n_probes": len(probes),
        "n_confirmed_mismatches": n_mismatch,
        "v2_fix": "reproject FirePred→VIIRS (bilinear continuous / nearest landcover) then shared crop",
        "probes": probes,
    }


def temporal_probes(ctx: dict, n: int = 8) -> dict:
    rows = list(ctx["train_rows"][:n]) + list(ctx["val_rows"][:2]) + list(ctx["test24_rows"][:2])
    out = []
    for r in rows:
        info = ctx["fire_by_id"][r["fire_id"]]
        days = info["day_files"]
        L = int(r["label_idx"])
        for ts in (2, 6):
            if L < ts:
                continue
            contract = window_temporal_contract(days[L - ts : L], days[L], ts)
            contract["target_key"] = r["target_key"]
            contract["ts"] = ts
            out.append(contract)
    return {
        "n": len(out),
        "all_ok": all(c.get("ok") for c in out),
        "contracts": out,
    }


def build_v1_vs_v2_comparison(stats_cmp: dict | None, pop: dict, avail: dict) -> dict:
    return {
        "protocol_version_v2": PROTOCOL_VERSION,
        "immutable_v1_artifacts": [
            "Phase 3A checkpoints/evals",
            "Phase 4B TS2/TS6 checkpoints/evals/shared_populations",
            "E1 run dir + headline metrics",
            "ENV ablation artifacts; ENV-0 continues under V1 protocol",
            "fp_stats_matched.npz (V1 subsample stats)",
            "selected_threshold.json historical sweeps",
        ],
        "changes_affecting_model_training": [
            {
                "id": "GEO-MISREG",
                "v1": "Independent center-crops on VIIRS vs FirePred when grids differ",
                "v2": "Reproject FirePred onto VIIRS grid (bilinear/nearest) then shared crop",
                "affects_training": True,
            },
            {
                "id": "AUG-DIRECTIONAL",
                "v1": "Flip/rot90 pixels; wind dir + aspect angles unchanged",
                "v2": "Unit-vector remap of channels 11/17/23 after geometric ops",
                "affects_training": True,
            },
            {
                "id": "STATS-SUBSAMPLE",
                "v1": "FP mean/std from ≤20-fire subsample helper",
                "v2": "FP mean/std refit on matched train unique day-crops (geo-aligned)",
                "affects_training": True,
                "stats_delta_vs_v1": stats_cmp,
            },
            {
                "id": "TEMP-FORECAST",
                "v1": "GFS bands present; sometimes described ambiguously",
                "v2": "Explicit forecast availability flags; same channels retained as declared forecast",
                "affects_training": False,
                "affects_documentation": True,
            },
        ],
        "changes_affecting_evaluation_only": [
            {
                "id": "THR-CEILING / THR-EXT",
                "v1": "Val sweep 0.05–0.75; selected 0.75 ceiling",
                "v2": "Extended val-only 0.01–0.99 via re-inference; headline metrics frozen",
                "affects_training": False,
                "affects_evaluation_diagnostics": True,
            },
            {
                "id": "POPULATION",
                "v1": "983/135/561 matched",
                "v2": "Same keys preserved",
                "population_verify_ok": pop.get("ok"),
                "affects_training": False,
            },
        ],
        "unchanged_by_design": [
            "27-channel layout / FP_BAND_IDX (still skips source precip idx 2)",
            "Label construction clip(BA_L - BA_{L-1})",
            "Day-L exclusion from inputs",
            "pos_weight formula (recomputed from train n_burn_px when retraining)",
            "Model architectures (MeanPool / RecencyFusion)",
            "No test-based threshold tuning",
        ],
        "feature_availability_summary": {
            "forecast_channels": avail.get("forecast_model_channels"),
            "n_flagged": avail.get("n_flagged"),
            "policy": avail.get("policy"),
        },
    }


def write_retraining_matrix(path: Path) -> None:
    text = f"""# Proposed Controlled Retraining Matrix (V2) — NOT LAUNCHED

**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}
**Protocol:** `{PROTOCOL_VERSION}`
**Status:** Prepared only. Do **not** start these runs until explicitly approved.

## Preconditions

- ENV-0 (V1 protocol) finished or explicitly allowed to continue in parallel on a separate GPU.
- V2 artifacts present under `outputs/v2_protocol/` (`fp_stats_matched_v2.npz`, population verify OK).
- Regression tests green (`tests/v2/`).
- No writes to DeepFire / DATA_ROOT.

## Shared V2 locks

| Item | Value |
|------|-------|
| Population | matched Set C **983 / 135 / 561** (frozen keys) |
| Seed | 42 (primary); optional seeds 43/44 later |
| Crop | 256 center on **VIIRS** grid after FP→VIIRS resample |
| Norm | `fp_stats_matched_v2.npz` (train-only) |
| Aug | geometric + directional remap ch 11/17/23 |
| Val thr grid | 0.01–0.99 step 0.01 (select on val only) |
| Checkpoint | max val F1@0.5 (unchanged rule) |
| drop_last | document; default True for parity unless ablated |

## Arms (proposed)

| Arm ID | Model | TS | Channels | Purpose | Priority |
|--------|-------|----|----------|---------|----------|
| V2-P4B-TS2 | MeanPool / SEUNet3DPred | 2 | full 27 | Corrected matched TS2 baseline | P0 |
| V2-P4B-TS6 | MeanPool / SEUNet3DPred | 6 | full 27 | Corrected matched TS6 baseline | P0 |
| V2-E1 | RecencyFusionUNet3D | 6 | full 27 | Corrected temporal fusion | P0 |
| V2-ENV-0 | RecencyFusion | 6 | mask FP 8–25 | Satellite+BA only under V2 | P1 |
| V2-ENV-WIND | RecencyFusion | 6 | mask wind group | Wind necessity | P1 |
| V2-ENV-TERRAIN | RecencyFusion | 6 | mask terrain | Terrain necessity | P1 |
| V2-ENV-VEG | RecencyFusion | 6 | mask veg/LC | Vegetation necessity | P1 |
| V2-ENV-METEO | RecencyFusion | 6 | mask meteo | Meteo necessity | P1 |
| V2-ENV-DROUGHT | RecencyFusion | 6 | mask drought | Drought necessity | P1 |

## Comparison rules

- Compare V2 arms **only** to other V2 arms on absolute metrics.
- V1 headline numbers remain immutable historical results; cite with protocol tag `V1-frozen`.
- Phase 3A published-protocol population stays separate (not matched).

## Launch command placeholders (do not execute yet)

```bash
# Example only — requires future train entrypoint wired to wtp.data.v2
# CUDA_VISIBLE_DEVICES=<free_gpu> python -u scripts/v2_train_eval.py \\
#   --arm V2-P4B-TS2 --seed 42 --i-understand-full-train
```

## Out of scope for first controlled wave

- Changing FP_BAND_IDX to include Total Precipitation
- Multi-seed until P0 single-seed completes
- Rewriting GeoTIFFs on disk
"""
    _write_text(path, text)


def write_protocol_report(
    path: Path,
    *,
    env_status: dict,
    pop: dict,
    geo: dict,
    temporal: dict,
    avail: dict,
    stats_meta: dict | None,
    stats_cmp: dict | None,
    comparison: dict,
    ready: dict,
) -> None:
    lines = [
        "# Corrected Protocol Report (V2 Preparation)",
        "",
        f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
        f"**Protocol:** `{PROTOCOL_VERSION}`",
        "**Retraining:** not launched",
        "",
        "## ENV-0 / queue safety",
        "",
        f"- ENV-0 left running: **{env_status.get('env0_alive', env_status.get('env0_left_running'))}**",
        f"- Queued ablations cancelled: **{env_status.get('queued_ablations_cancelled')}** "
        f"(found={env_status.get('queued_ablations_found')})",
        f"- Limitation: {env_status.get('limitation')}",
        "",
        "## Issues corrected in code (V2 package)",
        "",
        "| ID | Correction | Module |",
        "|----|------------|--------|",
        "| GEO-MISREG | FP→VIIRS resample + shared crop | `wtp.data.v2.geo_align` / `windows` |",
        "| AUG-DIRECTIONAL | Unit-vector angle remap under flip/rot90 | `wtp.data.v2.directional_aug` |",
        "| STATS-SUBSAMPLE | Matched-train-only geo-aligned FP stats | `wtp.data.v2.normalize` |",
        "| TEMP-FORECAST (docs) | Issue/valid-time + availability flags | `wtp.data.v2.timestamps` |",
        "| THR-EXT-LIMITATION | Val re-inference sweep 0.01–0.99 | `scripts/v2_extended_threshold_sweep.py` |",
        "",
        "## Issues remaining unresolved / deferred",
        "",
        "| ID | Status | Notes |",
        "|----|--------|-------|",
        "| FP-OMIT-BAND | methodological | Still skipped; comment corrected in channel_map; not re-added |",
        "| CAL-GAPS | methodological | Documented; optional gap==1 filter not applied yet |",
        "| PROT-DROP-LAST | methodological | Unchanged default |",
        "| COMPARE-POPULATIONS | methodological | Phase3A ≠ matched; labeling only |",
        "| SPLIT-GEO-NEAR-SAME | methodological | Within-split residual dependence |",
        "| GFS dir range | partial | Remap applied via unit vectors; signed [-90,90] convention still dataset-odd |",
        "| Land-cover z-score | unchanged | Still continuous z-score of class ids (not one-hot) |",
        "",
        "## Population preservation",
        "",
        f"- Verify OK: **{pop.get('ok')}**",
        f"- Counts: `{json.dumps(pop.get('counts'))}` (expected `{json.dumps(EXPECTED_COUNTS)}`)",
        f"- Geo-mismatch train fires retained: {pop.get('geo_mismatch_train_fires_still_in_train')}",
        "",
        "## Geospatial",
        "",
        f"- Confirmed mismatches probed: {geo.get('n_confirmed_mismatches')}",
        f"- V2 strategy: {geo.get('v2_fix')}",
        "",
        "## Temporal / availability",
        "",
        f"- Window contract probes OK: **{temporal.get('all_ok')}**",
        f"- Forecast model channels flagged: {avail.get('forecast_model_channels')}",
        f"- Policy: {avail.get('policy')}",
        "",
        "## Normalization",
        "",
        f"- V2 stats meta: `{json.dumps(stats_meta, default=str)[:500] if stats_meta else 'pending'}...`"
        if stats_meta
        else "- V2 stats: see `fp_stats_matched_v2.npz`",
        f"- Δ vs V1: {json.dumps(stats_cmp) if stats_cmp else 'n/a'}",
        "",
        "## Ready for controlled retraining?",
        "",
        f"- **{ready.get('ready')}** — {ready.get('reason')}",
        f"- Blocking: {ready.get('blocking')}",
        f"- Non-blocking follow-ups: {ready.get('followups')}",
        "",
        "## Artifacts",
        "",
        "- `outputs/v2_protocol/` (this report, comparison JSON, stats, probes)",
        "- `docs/V1_VS_V2_PROTOCOL.md`",
        "- `tests/v2/` regression suite",
        "- `src/wtp/data/v2/` corrected preprocessor (does not overwrite V1)",
        "",
    ]
    _write_text(path, "\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-stats", action="store_true")
    ap.add_argument("--max-stat-windows", type=int, default=None)
    ap.add_argument("--skip-build-probe", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    paths = load_paths()

    env_status = check_env0_and_queues()
    _write_json(OUT / "env0_safety.json", env_status)
    print(json.dumps(env_status, indent=2), flush=True)

    print("=== Population verify ===", flush=True)
    pop = verify_matched_population(paths)
    _write_json(OUT / "population_verify.json", pop)
    print(f"population ok={pop['ok']} counts={pop['counts']}", flush=True)

    ctx = build_matched_ctx(paths)

    print("=== Geo probes ===", flush=True)
    geo = geo_probes(ctx)
    _write_json(OUT / "geo_alignment_probes.json", geo)

    print("=== Temporal / availability ===", flush=True)
    temporal = temporal_probes(ctx)
    _write_json(OUT / "temporal_contracts.json", temporal)
    avail_rows = feature_availability_table()
    avail = summarize_availability(avail_rows)
    _write_json(OUT / "feature_availability.json", {"rows": avail_rows, "summary": avail})

    stats_meta = None
    stats_cmp = None
    if not args.skip_stats:
        print("=== Recompute matched-train FP stats (V2) ===", flush=True)
        mean, std, stats_meta = compute_matched_train_fp_stats(
            ctx["train_rows"],
            ctx["fire_by_id"],
            ts=6,
            max_windows=args.max_stat_windows,
            verbose=True,
            unique_days=True,
        )
        stats_path = save_fp_stats(OUT / "fp_stats_matched_v2.npz", mean, std, stats_meta)
        v1_mean, v1_std, v1_path = load_fp_stats(paths)
        stats_cmp = compare_to_v1(mean, std, v1_mean, v1_std)
        stats_cmp["v1_stats_path"] = str(v1_path)
        stats_cmp["v2_stats_path"] = str(stats_path)
        _write_json(OUT / "fp_stats_v1_vs_v2.json", stats_cmp)
        print(f"stats L2_mean={stats_cmp['l2_mean']:.4f}", flush=True)
    else:
        print("Skipping stats recompute (--skip-stats)", flush=True)

    build_probe = None
    if not args.skip_build_probe:
        print("=== V2 build smoke probes ===", flush=True)
        # Use V1 stats only if V2 not ready — still tests geo path
        build_probe = probe_v2_buildable(ctx, n_per_split=1, ts=2)
        _write_json(OUT / "v2_build_probes.json", build_probe)
        print(f"build probe ok={build_probe['ok']}", flush=True)

    comparison = build_v1_vs_v2_comparison(stats_cmp, pop, avail)
    comparison["directional_channels"] = DIRECTIONAL_MODEL_CHANNELS
    comparison["env0_safety"] = env_status
    _write_json(OUT / "v1_vs_v2_comparison.json", comparison)

    # Human-readable docs
    docs = ROOT / "docs" / "V1_VS_V2_PROTOCOL.md"
    _write_text(
        docs,
        "\n".join(
            [
                "# Immutable V1 vs Proposed V2 Protocol Comparison",
                "",
                f"**Generated (UTC):** {datetime.now(timezone.utc).isoformat()}",
                f"**V2 protocol:** `{PROTOCOL_VERSION}`",
                "",
                "V1 artifacts are **frozen**. V2 is a parallel corrected protocol inside",
                "`src/wtp/data/v2/` — it does not overwrite DeepFire preprocessors.",
                "",
                "## Training-affecting changes",
                "",
                "| ID | V1 | V2 |",
                "|----|----|----|",
                "| GEO-MISREG | Independent crops on mismatched grids | FP resampled to VIIRS; shared crop |",
                "| AUG-DIRECTIONAL | Angles not remapped | Unit-vector remap for ch 11/17/23 |",
                "| STATS-SUBSAMPLE | ≤20-fire FP stats | Matched-train geo-aligned FP stats |",
                "",
                "## Evaluation-only / documentation changes",
                "",
                "| ID | V1 | V2 |",
                "|----|----|----|",
                "| THR-EXT | Val probs missing; grid to 0.75 | Re-inference sweep 0.01–0.99; headlines frozen |",
                "| TEMP-FORECAST | Present | Explicit availability flags |",
                "| POPULATION | 983/135/561 | Same keys verified |",
                "",
                "## Unchanged",
                "",
                "- Channel keep-list (still omits FirePred source index 2)",
                "- Label rule and day-L exclusion",
                "- Architectures and matched population membership",
                "",
                "Machine-readable detail: `outputs/v2_protocol/v1_vs_v2_comparison.json`.",
                "",
            ]
        ),
    )

    write_retraining_matrix(OUT / "RETRAINING_MATRIX.md")

    thr_summary = OUT / "threshold_extended" / "summary.json"
    thr_done = thr_summary.exists()

    blocking = []
    if not pop.get("ok"):
        blocking.append("population_verify_failed")
    if stats_meta is None and not args.skip_stats:
        blocking.append("stats_missing")
    if not (OUT / "fp_stats_matched_v2.npz").exists() and not args.skip_stats:
        blocking.append("fp_stats_matched_v2_missing")

    ready = {
        "ready": len(blocking) == 0,
        "reason": (
            "V2 preprocessor + population + train-only stats prepared; "
            "controlled retraining matrix documented; no train jobs launched."
            if not blocking
            else "See blocking list"
        ),
        "blocking": blocking,
        "followups": [
            "Complete extended threshold sweep if not yet run",
            "Wire train entrypoint to v2 loaders before launch",
            "Optional: gap==1 sensitivity; precip band ablation",
        ],
        "extended_threshold_complete": thr_done,
        "do_not_launch_retraining_yet": True,
    }
    _write_json(OUT / "readiness.json", ready)

    write_protocol_report(
        OUT / "CORRECTED_PROTOCOL_REPORT.md",
        env_status=env_status,
        pop=pop,
        geo=geo,
        temporal=temporal,
        avail=avail,
        stats_meta=stats_meta,
        stats_cmp=stats_cmp,
        comparison=comparison,
        ready=ready,
    )

    print(json.dumps(ready, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0 if ready["ready"] or args.skip_stats else 0


if __name__ == "__main__":
    raise SystemExit(main())
