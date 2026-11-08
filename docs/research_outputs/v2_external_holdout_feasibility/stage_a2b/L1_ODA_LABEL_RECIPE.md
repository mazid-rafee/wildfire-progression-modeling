# L1 Label Recipe Addendum — ODA EventPolygon (post-A2b)

Supersedes the A2 blocker in `stage_a2/L1_LABEL_RECIPE.md` for the NIFC/WFIGS arm **when** Operational Data Archive EventPolygon is used.

## Strategy (unchanged)
\(BA_t = \mathrm{AF\_accum}_{\le t} \cup \mathrm{ODA\_perimeter}_{t}\) rasterized to the event VIIRS grid;  
\(y=\mathrm{clip}(BA_L-BA_{L-1})\); day L excluded from \(X\); FEDS not in \(y\) or ch26.

## ODA perimeter selection (intended; not executed in A2b)
1. Match incident by **IRWINID**.
2. Filter `FeatureCategory = 'Wildfire Daily Fire Perimeter'`.
3. Drop rows with null `PolygonDateTime` unless an approved fallback exists.
4. Group by UTC calendar day of `PolygonDateTime`.
5. Within day: keep the version with latest `GDB_FROM_DATE` (or max `GISAcres` if versions disagree on acres — lock one rule); discard tiny fragment acres below a QA floor.
6. **Retrospective GT:** use that day’s polygon for \(BA_t\) regardless of CreateDate.
7. **Operational ch26 / sensitivity:** only include day \(t\) if some retained row has `CreateDate ≤ end_of_day(t)` (or ≤ window cutoff).
8. Do **not** forward-fill a later day’s polygon onto earlier missing days without an explicit, approved carry policy that forbids using future CreateDate/PolygonDateTime.
9. Union with AF detections with `acq_datetime ≤ end_of_day(t)`.
10. Encode mask with `isfinite`-compatible semantics for band-8 parity.

## Pilot fire gate
Eligible if distinct PolygonDateTime days ≥ 8 and estimated retro TS=6 windows ≥ 3 after day collapse.  
A2b: Cedar Creek, Moose pass strongly; Bolt Creek marginal; McKinney, Pipeline fail.
