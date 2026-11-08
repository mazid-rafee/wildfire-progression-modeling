# Corrected V2 Protocol vs Historical V1

**Authoritative comparison:** `docs/V1_VS_V2_PROTOCOL.md`  
**Machine-readable:** `outputs/v2_protocol/v1_vs_v2_comparison.json`  
**Multi-seed protocol:** `V2.0-corrected-protocol` with launch policy **`mask_zero`** on GFS model channels 22–25.

---

## Principle

V1 artifacts remain frozen for historical reproduction. V2 is a **parallel corrected protocol** (`src/wtp/data/v2/`). V2 never silently replaces V1. All numbers in this manuscript package are **V2-only** unless explicitly labeled historical.

---

## Training-affecting corrections (V1 → V2)

| ID | V1 (historical) | V2 (corrected) |
|---|---|---|
| GEO-MISREG | Independent crops on mismatched grids | FirePred resampled to VIIRS; **shared crop** |
| AUG-DIRECTIONAL | Angles not remapped | Unit-vector remap for directional channels 11/17/23 |
| STATS-SUBSAMPLE | ≤20-fire FP stats | Matched-train **geo-aligned** FP stats (`outputs/v2_protocol/fp_stats_matched_v2.npz`) |

These changes affect learned weights. **Do not compare V2 multi-seed metrics to V1 single-run numbers as if they were the same experiment.**

---

## Evaluation / documentation corrections

| ID | V1 | V2 |
|---|---|---|
| THR-EXT | Historical grid often to 0.75; incomplete val-prob provenance in places | Validation-only grid **0.01–0.99** (99 steps); `tune_on_test=false` |
| TEMP-FORECAST | Implicit | Explicit availability flags; multi-seed matrix uses **`mask_zero`** for ch 22–25 |
| POPULATION | Matched keys 983/135/561 | **Same keys verified**; membership unchanged |

---

## Unchanged relative to V1 matched protocol

- Channel keep-list (still omits FirePred source index 2)
- Label rule and day-L exclusion
- Matched population membership keys
- Core SpaSE / RecencyFusion architectural families (V2 adds freeze-uniform control harness)

---

## Paper wording (recommended)

> All reported multi-seed results use the corrected V2 preprocessing protocol (geo-aligned shared crops, directional augmentation remap, matched-train FirePred statistics) with forecast GFS channels masked to zero at training/inference. Historical V1 runs are retained only for protocol archaeology and are not mixed into the V2 result tables.

---

## What not to claim

- That V2 “fixes” accuracy relative to V1 by a quantified Δ without a controlled V1↔V2 retrain matrix under identical seeds/arms (such a matrix is **not** part of the frozen 20-cell study).
- That `include_as_forecast` and `mask_zero` are interchangeable; the multi-seed paper numbers are **`mask_zero`**.
