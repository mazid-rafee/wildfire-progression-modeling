# Stage A1 — Label Semantics, Circularity, and Pilot Feasibility

**Decision context:** Option A (independent 2022+ evaluation), staged feasibility-first  
**Constraints honored:** no bulk download, no cube construction, no inference, no training, no threshold/checkpoint changes, no Git commit  
**Output root:** `outputs/v2_external_holdout_feasibility/stage_a1/`

---

## Executive verdict

| Question | Answer |
|---|---|
| Do FEDS / FEDS-MTBS provide true daily burned-area labels? | **No.** They provide **12-hourly AF-derived estimated perimeters** (alpha-hull around VIIRS active-fire pixels). FEDS-MTBS adds a **final Landsat/MTBS footprint constraint**, but time-resolved growth remains AF-derived. |
| Match to TS-SatFire BA band-8 semantics? | **No — not drop-in comparable.** |
| VIIRS input ↔ FEDS label dependence? | **Material.** Especially if channel 26 (historical BA) is also FEDS-derived (**high circularity**). |
| Evaluation class if FEDS used as y? | **Cross-dataset evaluation with changed label semantics** (+ AF circularity risk) |
| Path to comparable later-year benchmark? | Reconstruct **AF ∪ WFIGS/NIFC-like** cumulative BA (TS-SatFire family), not raw FEDS hulls |
| **Pilot cube construction now?** | **NO-GO** until a label strategy is locked |
| **Metadata acquisition (FEDS 2022 catalog ~0.5 GB)?** | **CONDITIONAL GO** for Stage A2 |

---

## 1. Authoritative 2022+ progression-relevant products

| Product | Years | What it actually is | Access |
|---|---|---|---|
| **FEDS CONUS archive** | 2012–2025 | Sub-daily (12-h) perimeters from Suomi-NPP VIIRS AF (`VNP14IMGML` C2 v4), FEDS algorithm v1.4.1; supersedes CA/Western releases | [Zenodo 22921773](https://zenodo.org/records/22921773) (~15.8 GB listed files; **2022 perimeter trio ≈ 486 MB**) |
| **FEDS-MTBS** | 2012–2024 | 7,739 large fires (>1000 acres); FEDS tracking **constrained** by MTBS final Landsat perimeter; 12-h perimeters, active lines, new AF pixels+FRP | [Zenodo 10.5281/zenodo.20187962](https://doi.org/10.5281/zenodo.20187962); viewer [FEDS25MTBS](https://www.ess.uci.edu/~ychen17/Shared/FEDS25MTBS/) |
| **MTBS** | ongoing | Final Landsat burned-area boundaries / severity — **not daily progression** | [mtbs.gov](https://www.mtbs.gov/); USDA FS EDW yearly layers |
| **WFIGS Daily Perimeters** | ~2021+ archive of updates | Operational incident perimeter **change archive** (Burn Period order; **not** official equal-interval progression) | [NIFC Hub](https://wfigs-nifc.hub.arcgis.com/pages/wfigs-daily-perimeters) |
| **TS-SatFire BA/progression** | 2017–Oct 2021 | Native V2 corpus; **no official 2022+ progression release** | Local `DATA_ROOT` (read-only) |

Primary references: Chen et al. 2022 (*Sci. Data*); project site [uci-nasa-firetracking](https://www.ess.uci.edu/~uci-nasa-firetracking/); Zhao et al. TS-SatFire (*Sci. Data* / arXiv:2412.11555).

---

## 2. Time-resolved BA vs AF-derived estimated perimeters

### FEDS / FEDS-MTBS
- **Input detections:** VIIRS **active fire** product (I-band fire locations + FRP), not burned-scar reflectance mapping.  
- **Geometry:** alpha-shape / alpha-hull around clustered AF pixels → **estimated perimeter**.  
- **Time step:** 12-hourly (AM/PM), not calendar-day VIIRS_Day files.  
- **“New growth”:** new AF pixels / perimeter expansion between steps — **AF detection timing**, including omission when cloud/smoke hides thermal signatures and commission from false AF.  
- **FEDS-MTBS:** MTBS final perimeter constrains **where** tracking is allowed; it does **not** convert each 12-h step into Landsat BA.

**Conclusion:** FEDS products are **not** independent daily burned-area labels. They are **active-fire-derived estimated progression**.

### MTBS alone
Final extent only → insufficient for next-day \(BA_L - BA_{L-1}\).

### WFIGS / NIFC-like
Closest operational analog to the **NIFC component** of TS-SatFire BA, but updates are irregular and explicitly “not official progression.”

---

## 3. Comparison to exact TS-SatFire / V2 BA band-8 semantics

V2 code (`load_ba_mask_v2`, `label_crop_v2`):

- Cumulative mask: \(\mathbf{1}[\mathrm{isfinite}(\mathrm{band}\,8)]\)  
- Target: \(\mathrm{clip}(BA_L - BA_{L-1}, 0, 1)\)  
- Channel 26 on newest input uses \(BA_{L-1}\); day \(L\) excluded from \(X\)

TS-SatFire paper BA construction:

- Training: **union** of **accumulated VIIRS AF** and **NIFC daily perimeters** (sometimes AF-only when NIFC commissions)  
- Test: visual QA choosing union vs AF-only against SWIR/NIR/MIR  

| Property | TS-SatFire BA | FEDS perimeter |
|---|---|---|
| Cumulative BA mask | Yes (band 8) | Only after custom rasterization |
| Temporal increment | Daily file difference | 12-h hull expansion |
| Resolution | Event VIIRS GeoTIFF grid | Vector hull ~AF 375 m |
| Confidence / QA | Manual test QA; known AF omission + NIFC commission | AF detection efficiency limits; hull fill-in |
| Zero-growth days | Possible (kept in V2; F1≈0 convention) | Possible when no new AF / no hull growth |
| Drop-in for V2 | Native | **Semantics change** |

Full table: `tables/LABEL_PRODUCT_COMPARISON.csv`.

---

## 4. Dependence between VIIRS inputs and FEDS labels (circularity)

| Setup | What evaluation measures | Risk |
|---|---|---|
| Reflectance ch 0–7 → predict FEDS growth | Agreement with **AF-tracking product**, not Landsat BA | Medium (same sensor family, different product) |
| ch 26 = FEDS cumulative → predict FEDS Δ | Near-**tautological** difference of same product | **High — avoid** |
| ch 26 = AF∪WFIGS, y = FEDS | Related AF lineages | High/Medium |
| ENV-0 reflectance(+BA) → FEDS | Still AF-perimeter agreement | Medium |
| Reflectance + AF∪WFIGS BA → AF∪WFIGS Δ | **Same label family as training** | Preferred for comparable later-year claim |

**Bottom line:** Using FEDS as \(y\) while feeding FEDS (or closely related AF) history in channel 26 would **not** measure independently observed fire growth. Even reflectance-only → FEDS primarily tests whether the model anticipates **AF-derived hull expansion**.

See `tables/VIIRS_LABEL_DEPENDENCE.csv`.

---

## 5. Later-year constructability of the 27 V2 channels

| Block | Constructable for 2022+? | Notes |
|---|---|---|
| VIIRS Day 0–5 | Yes | LAADS/GEE; match resampling/CRS |
| VIIRS Night 6–7 | Yes w/ gaps | Missing → zeros (V2) |
| FirePred non-GFS 8–21 | Yes in principle | GRIDMET, DEM, landcover, VNP13A1; **do not refit** `fp_stats_matched_v2.npz` |
| GFS 22–25 | Optional | Rebuild then **`mask_zero`** to match frozen multi-seed |
| BA ch 26 + labels | **Semantics-critical** | Must choose strategy (§8) before cubes |

Frozen preserved: SAT stats, FP stats, weights, val thresholds, next-day target definition, `mask_zero`.

Detail: `tables/CHANNEL_CONSTRUCTABILITY_AUDIT.csv`.

---

## 6. Event independence (pre-prediction)

Hard excludes (`EXCLUSION_REGISTRY`, 144 unique IDs):

- Train / val / test24 protocol fires  
- Missing-13 train-roster zeros  
- Official US_2021 zero-window trio  
- All **17 AF named** showcase folders (incl. 2022 `mosquito`, `calfcanyon`, `double_creek`)

Geographic/temporal checks (proposed):

1. Exact ID / IRWIN / MTBS ID collision  
2. Centroid < 25 km **and** date overlap ±30 days vs excluded ROIs  
3. Polygon IoU > 0.1 vs US_2021 test footprints  
4. Multi-ID complexes (e.g., Hermits Peak / Calf Canyon): treat as **one** event; reject if any alias overlaps AF-named `calfcanyon_fire` ROI  

Pilot criteria: `tables/PILOT_SELECTION_CRITERIA.csv` (metadata-only; **no model scores**).

---

## 7. Eligibility rules (locked before predictions)

Proposed in `tables/ELIGIBILITY_RULES_PROPOSAL.csv`, including:

- TS=6 history; day-L exclusion; BA-both; size ≥256  
- Keep empty-progression windows; report separately  
- Pilot: final area ≥1000 acres  
- Missing FirePred → full-channel exclude (ENV-0 secondary only)  
- Label-uncertainty flags; no post-hoc F1-based dropping  

---

## 8. Evaluation classification (required disclosure)

| Label strategy | Classification |
|---|---|
| **FEDS / FEDS-MTBS as y (and especially as ch26)** | **Cross-dataset evaluation with changed label semantics** + AF circularity risk — **not** a directly comparable TS-SatFire BA benchmark |
| **AF ∪ WFIGS/NIFC-like cumulative BA as y and ch26** | **Later-year same-semantics reconstruction attempt** — closest path to comparable V2 evaluation |
| Mesogeos / SeasFire / MTBS-final-only | **Incompatible** with frozen V2 next-day BA protocol |

---

## 9. Limited pilot (3–5 events) — technical feasibility only

- Slots **unassigned** pending FEDS/MTBS/WFIGS metadata (`tables/PILOT_SLOTS_UNASSIGNED.csv`)  
- Prespecified strata: western large, Southwest/Rockies, Northern Rockies (+ optional)  
- Selection **after** Stage A2 catalog parse using P1–P11 only  
- Explicitly **not** a performance benchmark; no inference in A1/A2  

---

## 10. Writable cube layout (when approved)

```text
$HOLDOUT_DATA_ROOT/          # NEW path; NOT DATA_ROOT
  stage_a2_catalog/          # FEDS/MTBS/WFIGS metadata extracts
  stage_a3_cubes/
    <event_id>/
      VIIRS_Day/ VIIRS_Night/ FirePred/  # V2-like layout
      LABEL_PROVENANCE.json              # BA recipe + hashes
  manifests/
    PILOT_MANIFEST.json                  # locked before reflectance pull
    HOLDOUT_MANIFEST.json                # locked before any inference
```

Do not write into TS-SatFire `DATA_ROOT`, DeepFire trees, or completed `outputs/v2_multiseed` / `v2_validation_audit` result artifacts.

---

## 11. Acquisition plan (nothing downloaded this stage)

See `tables/ACQUISITION_PLAN.csv`.

**A2 (conditional):** FEDS CONUS **2022** `.fgb` trio (~486 MB) + MTBS 2022 boundaries + FEDS-MTBS index + WFIGS queries for shortlisted IDs.  
**A3 (only after label lock):** VIIRS reflectance + FirePred rebuild for 3–5 pilot ROIs (multi-GB).

Licensing: confirm each Zenodo/USGov/NASA record before redistribution; cite Chen et al., MTBS, NIFC/WFIGS, Zhao et al. as appropriate.

---

## 12. GO / NO-GO

| Action | Recommendation | Why |
|---|---|---|
| Build pilot reflectance/FirePred cubes now | **NO-GO** | Label strategy undecided; risk of irreversible AF-circular evaluation |
| Use FEDS as primary BA labels for “comparable V2 holdout” claim | **NO-GO** | Changed semantics + AF dependence |
| Stage A2 metadata acquisition + assign 3–5 pilot IDs | **CONDITIONAL GO** | Needed to operationalize independence filters |
| Prefer AF∪WFIGS reconstruction as primary label path | **GO (strategy)** | Best chance at later-year **comparable** benchmark |
| Optional secondary FEDS-y arm later | Optional | Only with explicit cross-dataset / circularity disclosure |

### Exact next-step requirements for approval

1. Approve **label strategy**: (i) AF∪WFIGS primary, or (ii) FEDS primary as disclosed cross-dataset, or (iii) dual-arm.  
2. If (i) or (iii): approve Stage **A2** download of FEDS 2022 catalog (~0.5 GB) + MTBS 2022 vectors + WFIGS pilot queries.  
3. After A2: lock `PILOT_MANIFEST.json` (3–5 IDs) **before** any VIIRS reflectance bulk pull.  
4. Only then Stage **A3** cube construction under `$HOLDOUT_DATA_ROOT`.  
5. Inference remains forbidden until a full `HOLDOUT_MANIFEST.json` lock.

---

## Artifacts index

| Path | Content |
|---|---|
| `STAGE_A1_REPORT.md` | This report |
| `APPROVAL_REQUEST_A2.md` | Concise approval ask |
| `tables/LABEL_PRODUCT_COMPARISON.csv` | Product semantics |
| `tables/VIIRS_LABEL_DEPENDENCE.csv` | Circularity matrix |
| `tables/CHANNEL_CONSTRUCTABILITY_AUDIT.csv` | 27-channel later-year audit |
| `tables/ELIGIBILITY_RULES_PROPOSAL.csv` | Pre-prediction rules |
| `tables/PILOT_SELECTION_CRITERIA.csv` | Metadata-only pilot rules |
| `tables/PILOT_SLOTS_UNASSIGNED.csv` | 3–5 slot template |
| `tables/ACQUISITION_PLAN.csv` | URLs, sizes, stages |
| `artifacts/STAGE_A1_SUMMARY.json` | Machine-readable verdict |
