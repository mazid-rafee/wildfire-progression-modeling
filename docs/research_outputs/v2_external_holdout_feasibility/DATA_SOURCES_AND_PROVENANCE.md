# Data Sources, Provenance, Licensing, and Access

---

## 1. TS-SatFire (current V2 native corpus)

| Field | Detail |
|---|---|
| Role | Native 27-channel event GeoTIFF tree used by V2 |
| Local path | `/a/bear.cs.fiu.edu./disk/bear-b/users/mmazi007/Desktop/datasets/ts-satfire` (`configs/paths.yaml`, **read-only**) |
| Paper | Zhao, Gerard & Ban — *Scientific Data* / arXiv:2412.11555 |
| Distribution | [Kaggle](https://www.kaggle.com/datasets/z789456sx/ts-satfire), [Hugging Face](https://huggingface.co/datasets/GT-Fire-Research-and-Innovation/ts-satfire) |
| License | **CC BY 4.0** (dataset card) |
| Temporal coverage (BA/progression) | **Jan 2017 – Oct 2021** (179 events; CONUS focus) |
| AF named test | 17 named fires (clips; some dates into **2022**; multi-continent) |
| Access method | Already mirrored locally; do not re-download into DATA_ROOT |
| Code repo | https://github.com/zhaoyutim/TS-SatFire — **no clear OSS license** at prior review; do not copy scripts |

### On-disk layout used by V2
`{fire_id}/VIIRS_Day/`, `VIIRS_Night/`, `FirePred/` (+ optional `ESRI_LULC/`). BA = VIIRS_Day band 8.

---

## 2. Event catalogs for later-year candidate selection (not yet acquired as cubes)

### MTBS — Monitoring Trends in Burn Severity (USFS/USGS)
| Field | Detail |
|---|---|
| Role | Large-fire perimeters / occurrence locations for CONUS (+ AK/HI/PR) |
| Access | [mtbs.gov](https://www.mtbs.gov/); USDA FS EDW MapServer layers by year (e.g. 2022 Burned Area Boundaries layer 79) |
| Temporal | Landsat-era through recent mapped years (2022–2025 layers listed in EDW service) |
| License / use | US government public data; follow USFS/USGS redistribution terms |
| Holdout use | **Event footprints + dates only** until daily BA/VIIRS cubes exist |
| Limitation | Final perimeters ≠ daily cumulative BA progression labels |

### FEDS / FEDS-MTBS (UCI–UBC–NASA tracking)
| Field | Detail |
|---|---|
| Role | 12-hourly VIIRS-based fire tracking constrained by MTBS |
| Access | Project page / Zenodo (FEDS-MTBS cited for 2012–2024) |
| Holdout use | Candidate spatiotemporal tracks for cube ROI selection |
| Limitation | Still requires reflectance + env cube construction for V2 |

---

## 3. Ingredients to rebuild FirePred-like / VIIRS cubes (Class A)

| Component | Typical source | Notes |
|---|---|---|
| VIIRS Day/Night reflectance | NASA/NOAA VNP/VJ products | Must match TS-SatFire band set & ROI CRS |
| Daily BA | Dataset-specific BA algorithm or published daily BA | Must support band-8 cumulative semantics |
| NDVI/EVI | VIIRS VNP13A1-derived | FirePred src 0–1 |
| Meteo / drought / ERC | GRIDMET / NFDRS ecosystem | FirePred src 3–8, 12, 14 |
| DEM derivatives | SRTM-class DEM | slope/aspect/elev |
| Land cover | MODIS/ESRI-style map used by TS-SatFire | categorical nearest resample |
| GFS forecast bands | NOAA GFS | model ch 22–25; then **`mask_zero`** |

Exact TS-SatFire preprocessing scripts should be **reimplemented or licensed**, not copied blindly from the unlicensed code repo (`docs/ATTRIBUTION.md`).

---

## 4. Genuinely external datasets (incompatible drop-in)

### Mesogeos
- Zenodo / Orion-AI-Lab; Mediterranean daily ~1 km; ~2006–2022  
- Drivers + EFFIS burned areas; **not** VIIRS 27-ch TS-SatFire  
- License: check Zenodo record before use  
- **Cannot** feed frozen V2 weights without a new architecture/training run  

### SeasFire Cube
- Global 0.25°, 8-day, 2001–2021 seasonal fire modeling  
- Wrong spatiotemporal scale for next-day 256² progression  
- **Out of scope** for frozen V2 holdout  

---

## 5. Local exclusion / provenance artifacts already in this project

| Artifact | Path |
|---|---|
| Train/val/test populations | DeepFire `phase4b_.../shared_populations/population_*.json` |
| Matched windows | DeepFire `phase4a_.../matched_ts2_ts6_windows.csv` |
| V2 FP stats | `outputs/v2_protocol/fp_stats_matched_v2.npz` |
| Attribution | `docs/ATTRIBUTION.md` |
| Prior AF/BA audits (named fires) | DeepFire `docs/AF_LABEL_AUDIT_REPORT.md`, `BA_LABEL_AUDIT_REPORT.md` |
