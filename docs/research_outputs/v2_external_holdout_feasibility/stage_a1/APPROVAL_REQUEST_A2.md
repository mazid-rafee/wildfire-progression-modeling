# Approval Request — Stage A2 (Metadata Only)

## Stage A1 finding (one line)
FEDS/FEDS-MTBS are **AF-derived 12-h estimated perimeters**, not TS-SatFire BA band-8 labels; using them as \(y\) (especially with FEDS history in ch26) yields a **cross-dataset / circular** evaluation, not a directly comparable V2 holdout.

## Please approve one label strategy

- [ ] **L1 — Preferred:** Primary labels = reconstructed **AF ∪ WFIGS/NIFC-like** cumulative BA (TS-SatFire family); FEDS used only for event catalog / ROI  
- [ ] **L2 — Disclosed cross-dataset:** Primary labels = FEDS (or FEDS-MTBS) rasterized increments; paper must **not** claim BA-semantic parity; ch26 must **not** be FEDS cumulative  
- [ ] **L3 — Dual-arm:** L1 primary + L2 secondary with separate reporting  

## Please approve Stage A2 downloads (if L1/L2/L3 chosen)

- [ ] FEDS CONUS **2022** perimeter + fireline + newfirepix from https://zenodo.org/records/22921773 (~**486 MB**)  
- [ ] MTBS 2022(+optional 2023) burned-area boundaries (vector; tens–hundreds MB)  
- [ ] FEDS-MTBS **index/summary only** first from https://doi.org/10.5281/zenodo.20187962  
- [ ] WFIGS Daily Perimeters **queries for shortlisted pilot IDs only** (MB-scale)  

**Still forbidden after A2:** VIIRS reflectance bulk pull, FirePred rebuild, cube assembly, model inference.

## Writable path to create (when A2 approved)

`$HOLDOUT_DATA_ROOT/stage_a2_catalog/` — outside read-only `DATA_ROOT`.

## Success criteria for A2

1. Candidate event table with independence flags vs exclusion registry  
2. Assigned 3–5 pilot IDs in `PILOT_MANIFEST.json` (metadata criteria only)  
3. Written label-construction recipe matching approved L1/L2/L3  
4. Storage estimate for A3 reflectance+FP for those pilots only  

## Not requested now

Pilot cube construction (A3), inference, threshold changes, Git commit.
