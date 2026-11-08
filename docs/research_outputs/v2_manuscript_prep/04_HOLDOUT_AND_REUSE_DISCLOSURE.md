# Test Reuse and Untouched Holdout Disclosure

**Upstream:** `outputs/v2_validation_audit/UNTOUCHED_HOLDOUT_FEASIBILITY.md`, `FINAL_MULTISEED_REPORT.md` §8.

---

## Required disclosure (paper text)

The matched test populations (**test24 / test18 / test16**) were used during V2 protocol development and single-seed exploration. The five-seed study therefore reports an **exploratory reused 2021 US benchmark**, not an untouched final holdout. Bootstrap confidence intervals quantify fire-sampling uncertainty **within this fixed exploratory set**. They do **not** establish independent external validation or definitive population-level significance beyond that set.

---

## What was checked for a possible untouched holdout

### Phase-1 `missing_train_ids` (n=13)
Official **2017–2020 train-roster** events with **zero** TS=6 matched BA windows. They are eligibility failures, not a reserved blind test set. They influenced protocol documentation and are **unsuitable** as independent test holdout under the locked BA-matched protocol.

### Three non-contributing official 2021 test folders
`FL25210…`, `NM33402…`, `NM36768…` — **0** BA-valid windows. Cannot form an evaluation population.

### Conclusion
A genuinely untouched evaluation population **cannot** be constructed from these candidates. External later-year / disjoint events would be required, locked in a manifest **before** any checkpoint inference.

---

## External-holdout feasibility follow-up (A1–A3) — closed for this paper

A staged feasibility effort under `outputs/v2_external_holdout_feasibility/` assessed later-year L1 labels (FEDS vs AF∪NIFC/WFIGS; WFIGS Daily Public mid-fire gap; NIFS ODA EventPolygon as a possible perimeter archive). Stage A3 Cedar Creek / Moose cube construction was **stopped before any holdout cubes or inference** (`stage_a3/STAGE_A3_STATUS.md`). No independent external holdout was locked.

**Manuscript requirement:** State explicitly that this work **does not** include an independent external / later-year holdout evaluation. Claims remain limited to the exploratory reused 2021 matched benchmark (plus architecture / environmental ablations and qualitative / fire-level analyses on that protocol).

---

## Future holdout gate (deferred — not part of this submission)

1. Source years/regions unused in development (e.g., post-2021 events never inspected).
2. Freeze V2 eligibility rules (≥7 VIIRS_Day, BA on last input + label day, matched TS, GFS `mask_zero`, crop 256).
3. Exclude any ID in train/val/test manifests or prior audit decision sets.
4. Lock `HOLDOUT_MANIFEST.json` (IDs + window keys + hashes) **before** inference.
5. Use existing validation-selected thresholds; no post-hoc fire dropping; report micro + per-fire macro.

Until that gate is completed in a **future** effort, manuscript claims must remain scoped to the exploratory reused benchmark.
