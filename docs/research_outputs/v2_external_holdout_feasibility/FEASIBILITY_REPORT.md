# Independent External Holdout — Feasibility Report

**Protocol context:** frozen V2.0-corrected multi-seed models (`mask_zero`)  
**Question:** Can we construct a genuinely independent external fire-event holdout without touching completed experiments?  
**Verdict:** **Not with currently available on-disk data.** A credible holdout requires approved acquisition and reconstruction of later-year (preferably 2022+) event cubes **or** an explicitly labeled non-blind same-source mini-set that is scientifically weak.

---

## 1. Available candidate datasets and years

| Dataset / corpus | Years | Geography | Compatible with frozen 27-ch V2? | On disk now? | Blind for this project? |
|---|---|---|---|---|---|
| **TS-SatFire BA/progression** (Zhao et al.; Kaggle/HF CC BY 4.0) | 2017–Oct 2021 | CONUS | **Yes** (native layout) | Yes (`DATA_ROOT`) | **No** — train/val/test already used |
| **TS-SatFire AF named showcase** (17 fires) | 2017–2022 clips | US + intl. | Layout-compatible; short clips | Yes | **No** — official AF test; prior DeepFire AF/BA audits |
| **TS-SatFire 2022 named subset** (`calfcanyon`, `double_creek`, `mosquito`) | 2022 (10 days each) | US | Layout-compatible | Yes | **No** (AF contamination); **n≈12 windows** only |
| Phase-1 missing-13 train IDs | 2017–2020 roster | US | N/A — 0 TS=6 windows | Mostly absent/empty | **Invalid** |
| Official 2021 test zero-window (3) | 2021 | US | 0 BA-valid windows | Folders exist | **Invalid** |
| **MTBS / FEDS-MTBS event lists** | through ~2024 | US | Event IDs only — need cube build | Not as V2 cubes | Potentially yes **if never inspected** |
| VIIRS VNP/VJ surface reflectance + BA construction | ongoing | Global/US | Need pipeline to match TS-SatFire bands | Not assembled | Yes if new |
| GRIDMET / GFS / DEM / landcover (FirePred ingredients) | ongoing | US | Need FirePred-equivalent builder | Not assembled | Yes if new |
| **Mesogeos** (Mediterranean daily 1 km) | ~2006–2022 | Mediterranean | **No** (MODIS/EFFIS; different task/channels) | No | External but **incomparable** without new model |
| **SeasFire Cube** | 2001–2021 | Global 0.25° / 8-day | **No** (seasonal scale) | No | External but **incomparable** |

**Upstream fact:** TS-SatFire Scientific Data / Kaggle cards state BA/progression coverage **January 2017 – October 2021** (179 events). There is **no official same-source later-year progression release** beyond the short AF named clips.

---

## 2. Compatible vs incompatible components

### Compatible with frozen V2 (if reconstructed or already present)
- VIIRS Day I1–I5 + M11 and Night I4/I5 GeoTIFFs co-registered to event ROI  
- Cumulative BA in VIIRS_Day **band 8** (`isfinite` → mask)  
- FirePred 19-band cube with DeepFire keep-list (skip source index 2)  
- Spatial size ≥ 256×256; shared V2 geo-align (FP→VIIRS)  
- Frozen SAT mean/std + `fp_stats_matched_v2.npz` (do **not** refit)  
- Forecast policy `mask_zero` on model channels 22–25  
- Labels: next-step progression \(\mathrm{clip}(BA_L - BA_{L-1}, 0, 1)\); day L excluded from inputs  

### Incompatible / breaks frozen comparability
- Mesogeos / SeasFire / generic climate cubes as drop-in inputs  
- Replacing BA band-8 semantics with MTBS final perimeters without a daily cumulative product  
- Refitting normalization stats on holdout years  
- Changing crop size, channel count, threshold grid, or GFS policy  
- Using AF active-fire labels as progression targets  
- Zero-filling all FirePred for full E1/TS2/UNIFORM while claiming parity with V2 headlines (acceptable **only** as ENV-0-style secondary arm with disclosure)

---

## 3. Estimated eligible fires and windows (from source evidence)

### A. On-disk same-source 2022 named fires (probed)

| Fire | Dates | VIIRS days | FirePred | BA-both windows (center-256 probe) | new_ba≥3 |
|---|---|---:|---:|---:|---:|
| `calfcanyon_fire` | 2022-04-05→04-14 | 10 | 10/10 | 4 | 4 |
| `double_creek_fire` | 2022-08-29→09-07 | 10 | 10/10 | 4 | 4 |
| `mosquito_fire` | 2022-09-03→09-12 | 10 | 10/10 | 4 | 4 |
| **Total** | | | | **12** | **12** |

These are **upper-bound structural counts**, not a locked manifest. BA quality notes: `calfcanyon` shows non-monotonic cumulative BA in the center crop on at least one step (label BA can shrink)—flag as label-risk (matches prior DeepFire BA audit: PARTIAL for calfcanyon).

**Scientifically:** ~3 fires / ~12 windows is far below the exploratory 21-fire / 561-window test; bootstrap CIs would be essentially uninformative.

### B. In-corpus rejected pools
- Missing-13: **0** usable windows (invalid holdout)  
- Zero-window 2021 test folders: **0** windows  

### C. Reconstructed 2022+ CONUS (not yet built)
Not estimable from local cubes. Order-of-magnitude from public catalogs (requires approval to acquire):
- MTBS large fires per year in CONUS: typically **hundreds** of mapped perimeters  
- After VIIRS daily coverage + BA daily product + FirePred build + V2 eligibility (≥7 VIIRS days, BA on L−1 and L, size≥256): expect **tens of fires** and **hundreds of windows** if a multi-month season is processed—**only after** a locked event list and cube build  
- This report **does not** invent a precise eligible count without source cubes

---

## 4. Independence risks

| Risk | Assessment |
|---|---|
| ID overlap with train/val/test24 | Named fires **not** in Phase 4B population JSONs |
| Spatial near-duplicate to US_2021 | 2022 trio nearest US_2021 centers **≳100 km**; `carr_fire` (2018) is ~36 km from a 2021 test fire (pre-2022 reject anyway) |
| Same incident, different naming | 2021 named `dixie_fire` etc. are **not** protocol IDs but are same-year showcase clips—exclude from blind claims |
| Prior inspection / AF evaluation | **Critical:** all 17 named fires are the TS-SatFire **AF test set**; DeepFire AF/BA audits and AF inference already touched `calfcanyon`, `mosquito`, `double_creek` |
| Protocol development leakage | Missing-13 and 2021 test already shaped V2 docs—cannot be “untouched” |
| Label leakage (day L in inputs) | V2 contract excludes day L; adapter must preserve `window_temporal_contract` |
| Performance-based selection | **Forbidden** for locking; current CSV flags are structure/BA probes only |

**Conclusion:** Even the three 2022 fires fail a **blind / untouched** standard. They can only be discussed as a **contaminated same-source smoke check**, not as the paper’s independent holdout.

---

## 5. Data acquisition requirements (approval needed)

To build a lockable later-year same-source-style holdout:

1. **Event catalog** — MTBS burned-area boundaries and/or FEDS-MTBS tracked events for **2022+** (US), filtered to never-seen IDs.  
2. **VIIRS** — daily surface-reflectance stacks matching TS-SatFire band set (I1–I5, M11 day; I4/I5 night) for each event ROI.  
3. **Daily BA** — cumulative burned-area masks aligned to VIIRS grid (band-8 semantics). Using only final MTBS perimeters is **insufficient** for next-step progression.  
4. **FirePred-equivalent** — NDVI/EVI, GRIDMET meteo/wind/ERC/PDSI/precip, DEM slope/aspect/elev, landcover, GFS forecast bands—**same keep-list** as training.  
5. **Storage / compute** — multi-GB to tens of GB per season; write **outside** read-only `DATA_ROOT` (new holdout tree).  
6. **Licensing** — cite TS-SatFire CC BY 4.0 methods; respect USGS/NOAA/USFS product terms for reconstructed inputs.

**Stop point:** this package does **not** download these products.

---

## 6. Pipeline / adapter assessment

| Layer | Can holdout plug in? |
|---|---|
| `preload_matched_windows_v2` + `evaluate_windows` | Yes, given rows + `fire_by_id` |
| Frozen checkpoints + val-selected thresholds | Yes — must reuse; no retune |
| Stock `scripts/v2_train_eval.py` | **No** — hardcodes 561/553/531 asserts |
| `discover_fires` | Ignores named dirs; needs holdout indexer |
| Normalization | Must use frozen `fp_stats_matched_v2.npz` |
| Recommended adapter | Read-only **holdout loader** under `src/wtp/data/holdout/` (not yet implemented) that builds windows from a locked manifest without touching V2 stats/weights |

No adapter code was added in this phase (discovery only).

---

## 7. New labels / preprocessing assumptions that undermine comparability

- Non-TS-SatFire BA definition (e.g., Landsat dNBR perimeter rasterized once)  
- Different VIIRS product version / resampling than TS-SatFire  
- FirePred band order or units mismatch vs frozen stats  
- Refit of FP/SAT statistics on 2022+  
- Including forecast GFS as observations (`include_as_forecast`) while models trained with `mask_zero`  
- Dropping empty-progression or BA-sparse fires post hoc after seeing scores  
- Evaluating only ENV-0 because FirePred is missing, then comparing to full E1 headlines without disclosure  

---

## 8. Feasibility verdict

| Holdout type | Feasible now? | Ready to lock? |
|---|---|---|
| Blind external (new dataset / new year cubes never inspected) | **No** — cubes absent | **No** |
| Same-source later-year (TS-SatFire 2022 AF clips) | Structurally tiny + AF-contaminated | **No** (not for primary claims) |
| In-corpus missing-13 / zero-window 2021 | Invalid | **No** |
| Genuinely external Mesogeos/SeasFire | Data exist upstream | **No** for frozen V2 weights (incompatible inputs) |

**Recommended path:** treat current paper test as exploratory reused benchmark until an approved **2022+ reconstructed CONUS cube build** yields a locked manifest **before** any checkpoint inference.
