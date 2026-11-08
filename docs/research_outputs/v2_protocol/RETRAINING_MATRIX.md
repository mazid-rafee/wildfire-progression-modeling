# Proposed Controlled Retraining Matrix (V2) — NOT LAUNCHED

**Generated (UTC):** 2026-11-03T08:15:37.776887+00:00
**Protocol:** `V2.0-corrected-protocol`
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

## Entrypoint (wired; full train still gated)

Dry-run (CPU, no optimizer / no GPU) — already validated:

```bash
CUDA_VISIBLE_DEVICES= python -u scripts/v2_train_eval.py --arm V2-P4B-TS2 --dry-run
```

Protocol selection (never silent):

| Flag | Effect |
|------|--------|
| `--protocol v2` (default for arms) | Corrected preprocessing |
| `--protocol v1` | Frozen V1 loader (historical reproduction) |
| `--protocol v2_mask_gfs` / `--forecast-policy mask_zero` | Zero GFS ch 22–25 |

**P0 launch order (after explicit approval + forecast_policy choice):**
1. `V2-P4B-TS2` → 2. `V2-P4B-TS6` → 3. `V2-MEANPOOL-TS6-CTRL` → 4. `V2-E1`

```bash
# DO NOT RUN until approved — full-train path remains gated
# CUDA_VISIBLE_DEVICES=<free_gpu_NOT_env0> python -u scripts/v2_train_eval.py \
#   --arm V2-P4B-TS2 --forecast-policy include_as_forecast \
#   --no-dry-run --i-understand-full-train
```

See `outputs/v2_integration/V2_INTEGRATION_REPORT.md` and `training_readiness.json`.

## Out of scope for first controlled wave

- Changing FP_BAND_IDX to include Total Precipitation
- Multi-seed until P0 single-seed completes
- Rewriting GeoTIFFs on disk
- Enabling full train without explicit approval
