# Stage A3 — Final Status (Scope Reduced / Stopped)

**Date (UTC):** 2026-11-08  
**Decision:** External-holdout cube pilot **stopped**. Do **not** expand to 2023–2025, schedule further acquisition, or build cubes.  
**Priority shift:** Finalize the existing V2 research paper from completed five-seed / ablation / qualitative / fire-level results. Treat absence of an independent external holdout as an explicit **limitation**.

---

## Executive status

| Item | Status |
|---|---|
| Stage A3 pilot (Cedar Creek + Moose cubes) | **NOT COMPLETED** |
| Running A3 acquisition / cube jobs | **None found** |
| `HOLDOUT_DATA_ROOT` event trees | **Never created** |
| `outputs/.../stage_a3/` build deliverables | **Status-only** (this file); no windows/cubes/visualizations |
| Inference / training / threshold changes | **None** (forbidden; not started) |
| Git commit / push | **None** |
| Prior stages A1 / A2 / A2b | **Preserved intact** |

**Safe stop point:** Reconnaissance only. No partial GeoTIFF trees, no half-written manifests under a holdout root, nothing to corrupt. All prior feasibility artifacts remain on disk unchanged.

---

## What A3 had started (before stop)

1. Read A1/A2/A2b + V2 I/O / window / channel-map code paths.
2. Confirmed DATA_ROOT has **no** `US_2022_*` / Cedar / Moose event folders.
3. Confirmed candidates appear only in feasibility docs (not in matched train/val/test24 pools as fire folders).
4. Reconfirmed ODA EventPolygon extents/stats for Cedar Creek and Moose (API metadata; no bulk geometry archive download).
5. Estimated full VIIRS+FirePred transfer ≈ **~2.6–3.6 GB** for two multi-week fires (order-of-magnitude from existing event trees).
6. Hit **blockers** before any raster acquisition:
   - Google Earth Engine refresh token `invalid_grant` (cannot export VIIRS/FirePred via GEE).
   - No Earthdata `.netrc` / FIRMS map key for LAADS / FIRMS archive pulls.
   - No `HOLDOUT_DATA_ROOT` allocated; `DATA_ROOT` remains read-only and untouched.

**Not started:** independence audit write-up, ODA geometry harvest, AF download, BA rasterization, 27-channel tensors, window eligibility CSV, BA/input visualizations, A3 GO/NO-GO cube lock.

---

## Unfinished operations (explicitly cancelled)

- Cedar Creek / Moose independence audit document  
- Bounded ODA perimeter GeoJSON harvest into `HOLDOUT_DATA_ROOT`  
- VIIRS AF + Day/Night + FirePred acquisition  
- AF ∪ ODA label construction and monotonicity checks  
- TS=6 window materialization and quality audit  
- Bolt Creek optional compatibility test  
- Any 2023–2025 archive extension  
- Model inference on any holdout candidate  

No background download or cube job was left running to kill.

---

## Preserved artifacts (do not delete)

| Path | Role |
|---|---|
| `outputs/v2_external_holdout_feasibility/` (root reports) | Feasibility package |
| `stage_a1/` | Label semantics / circularity / acquisition plan |
| `stage_a2/` | WFIGS Daily L1 metadata NO-GO for mid-fire public stack |
| `stage_a2b/` | ODA EventPolygon CONDITIONAL GO (metadata); recipe addendum |
| `outputs/v2_manuscript_prep/04_HOLDOUT_AND_REUSE_DISCLOSURE.md` | Paper limitation language |
| `outputs/v2_validation_audit/UNTOUCHED_HOLDOUT_FEASIBILITY.md` | Upstream holdout audit |
| `/tmp/oda_a2b/` | Ephemeral A2b query cache (also copied under `stage_a2b/queries/`) |

Canonical durable copies of A2b queries live under `stage_a2b/queries/` — prefer those over `/tmp`.

---

## Paper implication

V2 evaluation remains an **exploratory reused 2021 US matched benchmark**. An independent later-year holdout was **investigated** (A1–A2b) but **not constructed**. Do not claim external validation from Cedar Creek / Moose or from ODA metadata alone.

---

## Machine-readable summary

See `artifacts/STAGE_A3_STATUS.json`.
