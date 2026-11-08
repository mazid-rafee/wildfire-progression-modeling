# Immutable V1 vs Proposed V2 Protocol Comparison

**Generated (UTC):** 2026-11-03T08:15:37.776102+00:00
**V2 protocol:** `V2.0-corrected-protocol`

V1 artifacts are **frozen**. V2 is a parallel corrected protocol inside
`src/wtp/data/v2/` — it does not overwrite DeepFire preprocessors.

## Training-affecting changes

| ID | V1 | V2 |
|----|----|----|
| GEO-MISREG | Independent crops on mismatched grids | FP resampled to VIIRS; shared crop |
| AUG-DIRECTIONAL | Angles not remapped | Unit-vector remap for ch 11/17/23 |
| STATS-SUBSAMPLE | ≤20-fire FP stats | Matched-train geo-aligned FP stats |

## Evaluation-only / documentation changes

| ID | V1 | V2 |
|----|----|----|
| THR-EXT | Val probs missing; grid to 0.75 | Re-inference sweep 0.01–0.99; headlines frozen |
| TEMP-FORECAST | Present | Explicit availability flags |
| POPULATION | 983/135/561 | Same keys verified |

## Unchanged

- Channel keep-list (still omits FirePred source index 2)
- Label rule and day-L exclusion
- Architectures and matched population membership

Machine-readable detail: `outputs/v2_protocol/v1_vs_v2_comparison.json`.

## Training integration (2026-11-03)

- Explicit selection: `--protocol v1|v2|v2_mask_gfs` via `scripts/v2_train_eval.py`.
- V1 path remains available; V2 never silently replaces it.
- Shared P0 train locks: `configs/experiments/v2/_shared_train.yaml`.
- Integration report / readiness: `outputs/v2_integration/`.
