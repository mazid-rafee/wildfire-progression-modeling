# Stage A2b — NIFS Operational Data Archive 2022 (EventPolygon)

**Purpose:** Re-open strict L1 after Stage A2 blocked on `WFIGS_Daily_Perimeters_Public`, by inspecting the official NIFS **Operational Data Archive 2022** `EventPolygon2022` layer for mid-fire daily perimeters, cutoff availability, and TS=6 eligibility.

**Constraints honored:** no bulk archive download, no VIIRS cube construction, no inference/training, no threshold or frozen-V2 changes, no DATA_ROOT writes, no Git commit/push. Queries were IRWINID-bounded metadata (+ minimal same-day geometry fingerprint samples only).

---

## Executive verdict

| Item | Result |
|---|---|
| Public API access to ODA 2022 EventPolygon | **Yes** |
| Mid-fire `Wildfire Daily Fire Perimeter` histories by IRWINID | **Yes** for Cedar Creek, Moose, Bolt Creek; **sparse** for McKinney, Pipeline |
| Vs Stage A2 WFIGS Daily (final-only BurnPeriod=1) | ODA **restores** multi-day progression for the same IRWINIDs |
| Distinct PolygonDateTime days (retro) | Cedar Creek 35; Moose 34; Bolt Creek 9; McKinney 3; Pipeline 2 |
| Acre-growth days vs same-day revisions | Growth exists; many days have multi-row GDB edits (not independent growth) |
| CreateDate operational gating | Material lag on some fires (Cedar Creek ~42% rows CreateDate > PolygonDateTime+24h) |
| Est. TS=6 windows (retro / operational) | Strong+marginal sum ≈ **60 / 41** |
| Equivalent archives 2023–2025 | **Exist** with Event Polygon + same key fields |
| Strict L1 | **CONDITIONAL GO** |
| Stage A3 pilot cubes | **Justified** under locked ODA recipe below — **stop before cubes until A3 approval** |

**Do not end the strict-L1 search here.** ODA is the archive NIFC documentation pointed to in A2; it is not another final-only mirror of WFIGS Daily. Ending L1 would only be warranted if ODA also lacked independent historical geometry — it does not, for at least two large 2022 candidates.

---

## 1. Public access and schema

**Endpoint (verified):**  
`https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/Operational_Data_Archive_2022/FeatureServer/7`  
Layer name: **EventPolygon2022** · geometry: polygon · `maxRecordCount`: 2000

**2022 archive layers (FeatureServer):**

| Id | Name | Relevance to BA |
|---:|---|---|
| 3 | EventPoint2022 | Not BA polygons |
| 5 | EventLine2022 | Not BA polygons |
| 6 | PerimeterLine2022 | Containment status lines (`Contained` / `Uncontained`) — **not** filled BA |
| 7 | EventPolygon2022 | **Primary** — includes `Wildfire Daily Fire Perimeter` |

**Schema fields used (25 total; key subset):**  
`IncidentName`, `IRWINID`, `FeatureCategory`, `PolygonDateTime`, `CreateDate`, `DateCurrent`, `GDB_FROM_DATE`, `GDB_TO_DATE`, `GISAcres`, `FeatureStatus`, `OBJECTID`.

**Population (count-only):** `FeatureCategory = 'Wildfire Daily Fire Perimeter'` → **84,411** features in 2022 EventPolygon (service-wide; not downloaded).

Artifacts: `queries/layer7.json`, `queries/service.json`, `queries/count_daily_perimeter.json`.

---

## 2. Incident matching (not name-only)

Candidates matched via WFIGS/IRWIN keys from Stage A2 catalog (`queries/irwin_map.json`):

| Candidate | UniqueFireIdentifier | IRWINID | State |
|---|---|---|---|
| McKinney | 2022-CAKNF-006177 | `{4630026A-7802-477E-9133-E094C32E8D8C}` | CA |
| Bolt Creek | 2022-WANWS-000150 | `{67F475B6-B2B7-460A-B4DF-601B8E614F3A}` | WA |
| Cedar Creek | 2022-ORWIF-220180 | `{21BB61C1-C479-4721-8295-9120974092D2}` | OR |
| Moose | 2022-IDSCF-022105 | `{7420DAD6-11D5-42CD-9779-41489600A63D}` | ID |
| Pipeline | 2022-AZCOF-000529 | `{F93BE77A-5DA1-43D1-B39C-86D6DDCC0440}` | AZ |

All subsequent counts filter: `IRWINID = <uuid> AND FeatureCategory = 'Wildfire Daily Fire Perimeter'`.

---

## 3. Per-fire perimeter availability

Full table: `tables/PER_FIRE_ODA_SUMMARY.csv`. Day rolls: `tables/DAY_SUMMARY_{CedarCreek,Moose,BoltCreek}.csv`.

| Fire | ODA rows | Distinct PolygonDateTime days | Acre-growth days | Multi-row days | First→last day | Acres first→last | Gaps >1 day | Retro TS=6 (upper) | Op. days (Create≤day end) | Late-only days | Op. TS=6 (upper) | Viability |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| McKinney | 78 | 3 | 2 | 2 | 2022-07-30→08-05 | 0.01→60,392 | 2 | 0 | 3 | 0 | 0 | INSUFFICIENT |
| Bolt Creek | 37 | 9 | 8 | 4 | 2022-09-10→10-20 | 7,660→14,821 | 6 | 3 | 7 | 2 | 1 | MARGINAL |
| Cedar Creek | 130 | 35 | 26 | 24 | 2022-08-01→10-19 | 2→127,311 | 16 | 29 | 23 | 12 | 17 | STRONG |
| Moose | 150 | 34 | 33 | 26 | 2022-07-18→10-03 | 300→130,198 | 12 | 28 | 29 | 5 | 23 | STRONG |
| Pipeline | 19 | 2 | 1 | 1 | 2022-06-12→06-13 | 0.1→26,532 | 0 | 0 | 2 | 0 | 0 | INSUFFICIENT |

**Cadence:** Cedar Creek and Moose are **near-daily but irregular** (median gap 1 day; max gaps 15–17 days). Bolt Creek is **sparse/irregular** (9 days over ~40 calendar days; max gap 23). McKinney/Pipeline are **few snapshots**, not usable progression series for TS=6.

**Same-day rows ≠ growth.** Cedar Creek has 24 multi-row days; geometry fingerprint samples show multiple distinct shapes on some days (edits / version supersessions). One geometry per observation day must be chosen (see §5–6).

---

## 4. Timestamp field semantics (independent inspection)

| Field | Observed role | Safe use |
|---|---|---|
| **PolygonDateTime** | Stated perimeter observation / capture time (nullable on rare rows) | Primary **retrospective observation day** for GT BA_t |
| **CreateDate** | System create time of the feature | Proxy for **operational availability**; gate ch26/history that must have been knowable at cutoff |
| **DateCurrent** | Often tracks create / last update | Secondary edit clock |
| **GDB_FROM_DATE / GDB_TO_DATE** | Geodatabase version validity interval for that row | Resolve which edit was “current” in the archive; **not** proof of daily independent geometries |

**Explicit caution (task 14):** Many rows share one `PolygonDateTime` with chained `GDB_FROM`/`GDB_TO` and changing `GISAcres`. That is **revision history**, not N independent daily fires. Counting GDB intervals as “days of progression” would inflate histories.

**CreateDate lag (median hours Create−PolygonDateTime; frac lag >24h):**

| Fire | Median lag (h) | Frac Create > Poly+24h |
|---|---:|---:|
| McKinney | −65.6 (Create often before Poly day stamp) | 0.015 |
| Bolt Creek | +14.6 | 0.13 |
| Cedar Creek | +10.6 | **0.42** |
| Moose | +5.9 | 0.14 |
| Pipeline | +18.7 | 0.46 |

Cedar Creek: **12 / 35** distinct polygon days have **no** row with `CreateDate ≤ end of that polygon day` — those geometries are usable for **retrospective GT** only, not as operationally available inputs on that day.

Sample attribute rows (Cedar Creek) in analysis notes show same `PolygonDateTime` with successive GDB windows and acre changes (e.g. 826 → 1054 acres) — classic same-day / version edits.

---

## 5. Growth vs revisions

**Procedure used (metadata):**
1. Group by calendar day of `PolygonDateTime`.
2. Within day: take acre range / multi-row count; optional minimal geom fingerprint sample.
3. Across days: acre-increase days = growth candidates after collapsing to one acre figure per day (max GISAcres among day rows, or last GDB version).

**Findings:**
- **Growth:** Cedar Creek 26 / Moose 33 / Bolt Creek 8 acre-increase days after day collapse.
- **Revisions:** High multi-row day rates (Cedar 24/35, Moose 26/34). Same-day distinct geometry fingerprints sampled on Cedar/Bolt confirm non-identical shapes, not mere attribute clones.
- **McKinney / Pipeline:** Acre jumps are large but across only 2–3 observation days → not a daily series.

---

## 6. Availability at prediction cutoff

Two constructions must stay distinct:

| Mode | Perimeter eligible for day \(t\) | Use |
|---|---|---|
| **Retrospective GT** | Collapse to one polygon per `PolygonDateTime` day \(t\) (resolve edits via latest `GDB_FROM` / max acres within day) | Labels \(y\); comparable to how historical NIFC stacks are used in literature after the fact |
| **Operational / no-leakage inputs** | Only rows with `CreateDate ≤ end_of_day(t)` (and preferably `≤` prediction cutoff for window ending L−1) | Strict claim that BA history in ch26 was knowable at cutoff |

**Never** assign a late-entered final perimeter to earlier days (the WFIGS Daily failure mode from A2).

Upper-bound TS=6 window counts assume consecutive eligible observation days can form length-6 histories (actual AF∪grid calendar may trim further in A3):

| Fire | Retro TS=6 | Operational TS=6 |
|---|---:|---:|
| Cedar Creek | 29 | 17 |
| Moose | 28 | 23 |
| Bolt Creek | 3 | 1 |
| McKinney | 0 | 0 |
| Pipeline | 0 | 0 |
| **Sum strong+marginal** | **60** | **41** |

Enough for a **small pilot** under retrospective GT; operational sensitivity should be reported as a secondary table, not silently equated.

---

## 7. Comparison to blocked WFIGS Daily and TS-SatFire BA rules

### Vs Stage A2 (`WFIGS_Daily_Perimeters_Public`)

| | WFIGS Daily Public (A2) | ODA EventPolygon 2022 (A2b) |
|---|---|---|
| Same IRWINIDs | BurnPeriod = 1, late `poly_PolygonDateTime` | Dozens of mid-fire Daily Fire Perimeter rows |
| Mid-fire progression | **No** | **Yes** (fire-dependent) |
| Lookahead risk if misused | Using single late polygon as daily BA | Using CreateDate-lagged or final-day geometry on early days |
| A2 conclusion | L1 blocked | L1 unblocked **conditionally** via ODA |

### Vs TS-SatFire / V2 encoding

From A2 capture of Zhao et al. and this repo:

- Labels: **accumulated VIIRS AF ∪ NIFC daily burned-area perimeters** (union; AF-only fallback when NIFC commission high).
- V2: `BA = isfinite(VIIRS_Day band 8)`; \(y=\mathrm{clip}(BA_L-BA_{L-1})\); day L out of \(X\).

**Remaining label-semantic differences (even with ODA):**

1. ODA polygons are **operational incident perimeters**, not guaranteed burned-area truth; commission/omission vs AF remains (Stage A1 concern).
2. Observation times are **irregular**, not a fixed daily grid — gaps must be handled (carry-forward policy must be locked; naive forward-fill can leak).
3. **CreateDate lag** means retrospective NIFC stacks ≠ what an operations desk had at cutoff.
4. No licensed local TS-SatFire label code — any AF∪ODA union is a **reimplementation**.
5. PerimeterLine Contained/Uncontained is **not** a BA fill substitute.
6. GDB versioning can leave tiny spurious acre rows (e.g. 0.3 ac fragments) alongside main polygons — QA rule required before rasterize.

---

## 8. Other 2022 layers (bounded)

- **PerimeterLine2022:** For Cedar Creek IRWINID, distinct `FeatureCategory` ∈ {`Contained`, `Uncontained`}. Useful for ops context, **not** for BA masks.
- **EventPoint / EventLine:** Not inspected beyond layer listing; not BA polygons.
- **No broadening** into unrelated incident feature types.

---

## 9. Archives for 2023 / 2024 / 2025

Public FeatureServers exist under the same org with Event Polygon layers and the same key fields (`IRWINID`, `FeatureCategory`, `PolygonDateTime`, `CreateDate`, `GDB_*`, `GISAcres`):

| Year | Service | Event Polygon layer | Count `Wildfire Daily Fire Perimeter` (count-only) |
|---|---|---|---:|
| 2022 | Operational_Data_Archive_2022 | /7 EventPolygon2022 | 84,411 |
| 2023 | Operational_Data_Archive_2023 | /7 Event Polygon | 99,674 |
| 2024 | Operational_Data_Archive_2024 | /6 Event Polygon | 121,527 |
| 2025 | Operational_Data_Archive_2025 | /6 Event_Polygon | 110,959 |

→ Holdout years after 2022 are **not** perimeter-source blocked a priori; candidate fire selection can extend with the same IRWINID recipe (still no bulk download here).

---

## 10. Strict L1 GO / NO-GO and A3 justification

### Strict L1 — **CONDITIONAL GO**

**GO because:**
- ODA supplies IRWINID-matched mid-fire Daily Fire Perimeter histories that WFIGS Daily Public lacked.
- ≥2 fires (Cedar Creek, Moose) have tens of distinct observation days and estimated dozens of TS=6 windows under retrospective day-collapse.
- AF∪ODA remains within the TS-SatFire-family L1 definition (NIFC/WFIGS arm = operational archive perimeters).

**Conditions (must lock before A3):**
1. Source = ODA EventPolygon `Wildfire Daily Fire Perimeter` matched by **IRWINID** (not name).
2. One geometry per `PolygonDateTime` day; resolve same-day edits via GDB validity / explicit acre+geom rule.
3. Never backfill final or later-day perimeter onto earlier days.
4. Union with **cutoff-gated** VIIRS AF (`acq ≤` end of day for that BA_t).
5. Pilot fire set: **Cedar Creek + Moose** (+ optional Bolt Creek); **replace** McKinney/Pipeline or add 2023 ODA fires for a 3–5 fire pilot.
6. Disclose irregular timing + CreateDate lag; publish **operational-availability** sensitivity separately from retrospective GT.
7. Still **no** bulk FEDS required for labels.

### Stage A3 pilot construction — **JUSTIFIED** (approval-gated)

A3 may proceed to **bounded** AF∪ODA rasterization for the strong fires only after recipe lock. This Stage A2b package **stops before any cube construction**.

### When to end the strict-L1 perimeter search

If A3 discovery showed ODA day-collapsed geometries still collapsed to ≤1–2 independent shapes per fire after QA, **then** stop chasing equivalent public perimeter services and fall back to approved L1-AF-only or L2. **That is not the case now** for Cedar Creek / Moose.

---

## 11. Deliverable index

| Path | Content |
|---|---|
| `STAGE_A2B_REPORT.md` | This report |
| `README.md` | Short pointer |
| `tables/PER_FIRE_ODA_SUMMARY.csv` | Per-fire counts / TS=6 / viability |
| `tables/DAY_SUMMARY_*.csv` | Day-level rows for strong/marginal fires |
| `artifacts/STAGE_A2B_SUMMARY.json` | Machine-readable verdict |
| `queries/*` | Bounded service/layer/count/analysis JSON copies |

---

## 12. Explicit non-actions

No cubes, no VIIRS downloads, no model runs, no threshold edits, no frozen V2 experiment mutation, no DATA_ROOT writes, no Git operations.
