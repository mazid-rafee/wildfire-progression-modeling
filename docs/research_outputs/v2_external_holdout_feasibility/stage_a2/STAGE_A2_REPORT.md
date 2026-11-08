# Stage A2 — Metadata-Only Investigation (L1)

**Approved strategy (from `stage_a1/APPROVAL_REQUEST_A2.md`):** **L1**  
Primary labels = reconstructed **accumulated VIIRS AF ∪ time-valid NIFC/WFIGS perimeters** (TS-SatFire family). FEDS = event catalog / ROI only — not primary labels.

**Constraints honored:** no bulk FEDS (~486 MB) download, no reflectance/FirePred cubes, no inference/training, no DATA_ROOT mutation, no Git commit.

---

## Executive verdict

| Item | Result |
|---|---|
| L1 with **time-valid** WFIGS mid-fire polygons via public API | **Not feasible** with current evidence |
| Public `WFIGS_Daily_Perimeters_Public` for 2022 large fires | Almost always **BurnPeriod = 1** with **late** `poly_PolygonDateTime` (months–years after the fire) |
| 2022 discovery features with `BurnPeriod ≥ 2` | **8** total; `≥ 5` → **0** |
| Semantics vs TS-SatFire | At best **approximate** (likely **AF-only fallback**); not verified parity |
| **Stage A3 pilot cubes** | **NO-GO** under strict L1 |
| Full FEDS archive download | Still **not** required / not done |

---

## 1. How WFIGS historical polygons are archived

### Service found (correct name)
`WFIGS_Daily_Perimeters_Public`  
`https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Daily_Perimeters_Public/FeatureServer/0`

(Earlier guessed `WFIGS_Daily_Perimeters` URL was invalid.)

### Documented behavior (Hub)
- Archives **geometry changes** from Interagency Perimeters as separate features.  
- Attribute-only updates modify the latest feature.  
- **`BurnPeriod`**: sequential order of geometry changes — **not** equal time steps.  
- Explicitly **not** official progression.  
- Perimeters **not available for every incident**.

### Timestamp / ID fields (probed)
See `tables/WFIGS_TIMESTAMP_FIELDS.csv`.

| Field | Role |
|---|---|
| `poly_PolygonDateTime` | Stated capture time of polygon |
| `poly_CreateDate` / `poly_DateCurrent` | Record create / last edit (UTC) |
| `attr_CreatedOnDateTime_dt` / `attr_ModifiedOnDateTime_dt` | IRWIN incident timestamps |
| `attr_UniqueFireIdentifier`, `attr_IrwinID`, `poly_IRWINID` | Incident keys |
| `BurnPeriod`, `AcreageChange` | Revision order / acre delta |

### Observed vs documented (critical)
For closed **2022** large CONUS fires queried by UniqueFireIdentifier:

| Fire | UniqueFireIdentifier | BurnPeriods retained | `poly_PolygonDateTime` |
|---|---|---:|---|
| McKinney | 2022-CAKNF-006177 | 1 | 2023-03-23 (fire discovery 2022-07-29) |
| MOSQUITO | 2022-CATNF-001371 | 1 | 2024-07-24 |
| BOLT CREEK | 2022-WANWS-000150 | 1 | 2026-09-28 |
| Cedar Creek (OR) | 2022-ORWIF-220180 | 1 | 2026-11-05 |
| Moose (ID) | 2022-IDSCF-022105 | 1 | 2023-01-12 |
| Pipeline (AZ) | 2022-AZCOF-000529 | 1 | 2023-09-20 |
| Calf Canyon | 2022-NMSNF-000028 | 1 | 2025-12-11 |

**Interpretation:** The public Daily Perimeters layer, as queried, does **not** expose a usable mid-fire revision stack for these events. Using the single retained polygon as “daily BA” for days during the fire would inject **final / late perimeter lookahead**.

`InterAgencyFirePerimeterHistory_All_Years_View` is **final** wildfire perimeters by `FIRE_YEAR` — also not daily progression.

NIFC FAQ points to **Operational Data Archives** (`GDB_FROM_DATE` / `GDB_TO_DATE` / `PolygonDateTime`) for reconstructing progression — **not** validated here as a public bulk API for 2022 pilots.

---

## 2. Can 2022+ WFIGS support historical daily BA without later/final leakage?

**No — not from the public Daily Perimeters service alone.**

Counts (discovery date in 2022):

- Features: **2528**  
- `BurnPeriod ≥ 2`: **8**  
- `BurnPeriod ≥ 5`: **0**  

Without multi-revision histories keyed to fire days **before** each prediction cutoff, L1’s WFIGS/NIFC arm cannot be built without lookahead.

---

## 3. TS-SatFire AF ∪ NIFC rules (from paper; no arXiv re-fetch)

From Zhao et al. *TS-SatFire* (Sci. Data / arXiv:2412.11555) as previously captured:

**Burned-area labels**
- Sources: **accumulation of VIIRS AF** + **NIFC daily burned-area perimeters**.  
- Training: typically **union** of accumulated AF and NIFC; sometimes **AF-only** when NIFC commission is high.  
- Test: visual QA choosing union vs AF-only against SWIR/NIR/MIR.  
- AF training labels: FIRMS/VIIRS AF with manual removal of bad regions; test AF uses manual I4/I5 thresholding on named sites.

**V2 encoding (this repo)**  
`isfinite(VIIRS_Day band 8)` → cumulative mask; \(y=\mathrm{clip}(BA_L-BA_{L-1})\); day L excluded from \(X\).

**No local TS-SatFire label-generation code is licensed for copy** (`docs/ATTRIBUTION.md`). Any reconstruction is a **reimplementation**, not a byte-identical pipeline.

---

## 4. Historical VIIRS AF access and cutoff availability

| Product | Access | Cutoff relevance |
|---|---|---|
| **VNP14IMGML** (S-NPP 375 m monthly locations) | UMD SFTP `fuoco.geog.umd.edu` `data/VIIRS/C2/VNP14IMGML` (see C2 AF User Guide); also FIRMS archive download | Each detection has acquisition datetime → can gate **AF ≤ end of day L−1** |
| **VJ114IMGML** (NOAA-20) | Same family from 2018+ | Optional; TS-SatFire/FEDS often S-NPP-centric — freeze choice before A3 |
| FIRMS NRT vs standard | NRT replaced by science-quality with lag | Prefer **standard** ML product for retrospective holdout |

**Before each prediction cutoff (end of newest input day L−1):**  
Only AF detections with acquisition time ≤ cutoff may enter the cumulative AF mask. Later detections must not appear in ch26 or in \(y\) for that window.

**Not downloaded** in A2 (metadata/docs only).

---

## 5. Bounded candidate metadata (no model scores)

Catalog sources used: WFIGS Daily (counts/fields) + Certified perimeters (273 large-acre 2022 discovery records) + targeted UniqueFireIdentifier probes.

Proposed **metadata** pilot pool (3–5 after exclusions):  
`tables/CANDIDATE_EVENTS_METADATA.csv`

**Include (if labels solved):** McKinney (CA), Bolt Creek (WA), Cedar Creek (OR), Moose (ID), Pipeline (AZ) [+ optional Rum Creek].  
**Exclude:** MOSQUITO (`mosquito_fire` AF-test), Calf Canyon / Hermits Peak complex (`calfcanyon_fire` AF-test).

All currently have **`wfigs_time_resolved = NO`**.

Spatial IoU vs US_2021 footprints **not** computed yet (needs geometry pull — defer to pre-A3 with explicit approval for small GeoJSON extracts).

---

## 6. Handling rules (deterministic; pre-prediction)

| Case | Rule |
|---|---|
| Zero-growth day | Keep window if BA-both; \(y=0\); F1 via `1e-8`; report empty-target FA separately |
| Missing WFIGS for that day | Do **not** forward-fill from later/final perimeter; use AF-only for that day **or** drop window if recipe requires union |
| Delayed / late `poly_PolygonDateTime` | Treat as **unusable** for that fire-day unless create/capture provenance proves ≤ cutoff |
| Unreliable boundary | Flag; exclude from primary if hull/ops QA fails pre-registered checks |
| Small fires | Pilot: final ≥1000 acres |
| Performance | **Never** used for inclusion |

Full eligibility: `tables/ELIGIBILITY_PROTOCOL_L1.csv`.

---

## 7. L1 recipe vs TS-SatFire semantics

| Component | Status |
|---|---|
| Accumulated VIIRS AF with cutoff gating | **Constructable** (data exist; not downloaded) |
| Time-valid NIFC/WFIGS daily polygons | **Not available** in public Daily Perimeters for probed 2022 large fires |
| Union + manual visual QA | Manual QA **not automated**; paper test QA cannot be reproduced at scale without human protocol |
| Resulting claim | **Cannot** honestly claim full TS-SatFire-compatible labels yet. AF-only would be a **documented subset** of their fallback path, still approximate (no identical GEE resampling / QA). |

---

## 8. GO / NO-GO for Stage A3

### **NO-GO** — pilot cube construction under strict L1

**Why:** Without mid-fire, time-valid WFIGS/NIFC polygons, building cubes would either (a) leak final perimeters into historical days, or (b) silently become AF-only while claiming L1 union semantics.

### Still allowed later (needs new approval)
1. **Find OD Archive / progression GDBs** with true historical revisions for 3–5 candidates → then revisit L1 A3.  
2. **Approve L1-AF-only** (explicit subset disclosure) → then A3 with AF masks + reflectance.  
3. **Switch to L2** (FEDS labels, cross-dataset disclosure) → separate path.  
4. Optional small GeoJSON extracts for independence IoU — still not full FEDS .fgb.

### Storage if A3 were approved later
Order-of-magnitude **7–30 GB** for 3–5 fires (`tables/PILOT_STORAGE_ESTIMATE.csv`).

---

## 9. Remaining evidence required before any A3 approval

1. A **public or licensed** source of **mid-fire** perimeter revisions with proven capture times ≤ each cutoff for pilot IDs, **or**  
2. Written approval to proceed with **AF-only** labels (L1-subset) including paper wording limits, **or**  
3. Written approval to switch to **L2 FEDS**.  
4. Small geometry extracts for independence IoU vs US_2021 / AF-named ROIs.  
5. Locked `PILOT_MANIFEST.json` + label recipe hash **before** reflectance download.

---

## Artifacts

| Path | Content |
|---|---|
| `STAGE_A2_REPORT.md` | This report |
| `L1_LABEL_RECIPE.md` | Intended recipe + blockers |
| `tables/*` | Candidates, timestamps, acquisition, storage, eligibility |
| `queries/*` | Bounded ArcGIS JSON responses |
| `artifacts/STAGE_A2_SUMMARY.json` | Machine-readable verdict |
