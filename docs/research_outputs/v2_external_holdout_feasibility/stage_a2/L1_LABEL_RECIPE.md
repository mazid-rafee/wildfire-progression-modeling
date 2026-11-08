# L1 Label Construction Recipe (Intended) + A2 Blockers

## Strategy
**L1** per `stage_a1/APPROVAL_REQUEST_A2.md`:  
\(BA_t = \mathrm{AF\_accum}_{\le t} \cup \mathrm{WFIGS/NIFC\_perimeter}_{\le t}\) (rasterized to event VIIRS grid), then V2 target \(y=\mathrm{clip}(BA_L-BA_{L-1})\).

FEDS is **not** used in \(y\) or channel 26.

## Intended steps (not executed)
1. Define ROI from MTBS/FEDS/WFIGS final footprint + buffer (catalog only).  
2. Build daily cumulative AF from VNP14IMGML detections with `acq_datetime ≤ end_of_day(t)`.  
3. Select WFIGS geometry for day \(t\) with capture/create provenance **≤ end_of_day(t)**; never use a later BurnPeriod.  
4. Rasterize union to VIIRS grid; encode as band-8-compatible mask (`isfinite` semantics).  
5. Assemble ch26 from \(BA_{L-1}\); exclude day L from inputs.  
6. Keep empty-progression windows; no post-hoc F1 filtering.

## A2 blocker
Step 3 fails for probed 2022 large fires on `WFIGS_Daily_Perimeters_Public` (single late BurnPeriod).

## Allowed fallbacks (require new approval — do not silently apply)
- **L1-AF-only:** skip WFIGS; disclose as TS-SatFire subset/fallback path.  
- **L2:** FEDS increments as \(y\) with cross-dataset disclosure; ch26 must not be FEDS cumulative.
