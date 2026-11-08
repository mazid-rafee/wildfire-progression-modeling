# Approval Gate — Stop Before Large Downloads / Inference

This feasibility phase **stops here** until authors approve next steps.

---

## Completed without approval (safe)

- Read-only discovery of DATA_ROOT / manifests / audits  
- Named-fire structural BA probes (no model forward passes)  
- Exclusion registry + candidate/reject CSVs  
- Protocol proposal documents  

## Requires explicit approval before proceeding

1. **Download** MTBS/FEDS catalogs, VIIRS granules, GRIDMET/GFS, or other bulk rasters  
2. **Allocate** a new writable holdout cube directory (must not write into read-only `DATA_ROOT`)  
3. **Implement** holdout adapter / `v2_holdout_eval.py`  
4. **Build** Class A event cubes for 2022+  
5. **Lock** `HOLDOUT_MANIFEST.json`  
6. **Run** checkpoint inference on any holdout window  
7. **Git commit** of this package or future eval outputs  

## Recommended approval decision options

| Option | Action | Outcome |
|---|---|---|
| **A** | Approve Class A cube-build plan (budget storage + which years) | Path to lockable independent later-year holdout |
| **B** | Defer holdout; keep exploratory reused-benchmark framing in paper | Submit without new eval (disclose limitation) |
| **C** | Allow Class B smoke-check on 3×2022 AF clips **with contamination disclosure** | Weak appendix only; still not “external holdout” |
| **D** | Scope a separate external-dataset paper (Mesogeos/etc.) | Out of band for frozen V2 weights |

## Stage A2b / A3 update (scope reduced)

ODA EventPolygon made a future L1 pilot *technically* plausible (A2b). Stage A3 was **stopped without cubes** (`stage_a3/STAGE_A3_STATUS.md`). **Do not** schedule further acquisition or 2023–2025 expansion for this paper cycle.

## Current recommendation

Choose **B** for the present manuscript: keep the exploratory reused-benchmark disclosure and state the **absence of an independent external holdout** as a limitation. Defer Option **A** (new Class A cubes / locked holdout) to a separate future effort if needed. **Do not** treat Option C as clearing the blocker.
