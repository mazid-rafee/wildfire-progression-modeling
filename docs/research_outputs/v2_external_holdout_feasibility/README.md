# V2 External Holdout Feasibility Package

**Status:** Feasibility A1–A2b complete; Stage A3 **STOPPED** (no cubes). External holdout **not locked** — treat as a paper **limitation**. Do not expand to 2023–2025 acquisition.  
**Constraint honored:** no model inference, no retraining, no threshold changes, no mutation of V1/V2 experiment outputs or DATA_ROOT, no Git commit/push.

## Documents

| File | Purpose |
|---|---|
| [`FEASIBILITY_REPORT.md`](FEASIBILITY_REPORT.md) | Full evidence-backed feasibility assessment |
| [`PROPOSED_HOLDOUT_PROTOCOL.md`](PROPOSED_HOLDOUT_PROTOCOL.md) | Deterministic eligibility + locked-eval proposal (for approval) |
| [`DATA_SOURCES_AND_PROVENANCE.md`](DATA_SOURCES_AND_PROVENANCE.md) | Authoritative sources, coverage, licensing, access |
| [`SAME_SOURCE_VS_EXTERNAL.md`](SAME_SOURCE_VS_EXTERNAL.md) | Distinguishes later-year same-source vs genuinely external evaluation |
| [`APPROVAL_GATE.md`](APPROVAL_GATE.md) | What requires approval before downloads / cube build / inference |

## Tables / artifacts

- `tables/CANDIDATE_EVENT_MANIFEST.csv` — on-disk named-fire catalog with quality flags (**not** model-score selected)
- `tables/CANDIDATE_2022_WINDOWS_PROBE.csv` — length/BA center-crop probe for three 2022 fires
- `tables/REJECTED_IN_CORPUS_POOLS.csv` — missing-13 and zero-window 2021 folders (explicitly invalid)
- `artifacts/EXCLUSION_REGISTRY.json` — IDs barred from any holdout claim
- `artifacts/named_fire_disk_probe.json` — raw probe
- `artifacts/FEASIBILITY_SUMMARY.json` — machine-readable verdict

## Stage A1 (label semantics / circularity / pilot plan)

See [`stage_a1/`](stage_a1/) — FEDS vs TS-SatFire BA analysis, channel audit, acquisition plan, **NO-GO on pilot cubes** until label strategy approval; **CONDITIONAL GO** for metadata-only Stage A2.

## Stage A2 (L1 metadata)

See [`stage_a2/`](stage_a2/) — **L1 selected**; public WFIGS Daily Perimeters lack mid-fire 2022 revisions → **NO-GO for A3 pilot cubes** until an alternate time-valid perimeter source or approved AF-only/L2 path.

## Stage A2b (NIFS ODA EventPolygon)

See [`stage_a2b/`](stage_a2b/) — ODA 2022 restores mid-fire Daily Fire Perimeters for Cedar Creek / Moose (Bolt Creek marginal). Strict L1 was **CONDITIONAL GO** for a future pilot; **not executed**.

## Stage A3 (pilot cubes — stopped)

See [`stage_a3/`](stage_a3/) — Cedar Creek / Moose cube pilot **not completed**. No `HOLDOUT_DATA_ROOT`, no running jobs, no inference. Scope reduced: finalize V2 paper from existing experiments; keep missing independent holdout as an explicit limitation.
