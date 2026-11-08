# Untouched Holdout Feasibility Assessment

## Candidate pool: Phase-1 `missing_train_ids` (n=13)

- `20777452`
- `20777459`
- `20777482`
- `20777508`
- `20778225`
- `20900040`
- `21998287`
- `23036806`
- `24332700`
- `24332783`
- `24333277`
- `24461421`
- `24463187`

These are **official 2017–2020 train-roster** events that contributed **zero TS=6 matched windows** (eligibility failures), not a reserved blind set.

### Audit

| Check | Result |
|---|---|
| In V2 train/val/test matched rows | **No** |
| Suitable independent **test** holdout | **No** (train-roster identity; 0 BA-matched windows) |
| Influenced protocol documentation | **Yes** (Phase-1 audit) |

Disk probe (direct path only; no full-tree walk): `artifacts/MISSING13_DISK_AUDIT.csv`.

## Official test non-contributors (n=3)

`FL25210…`, `NM33402…`, `NM36768…` — no BA-valid matched windows under locked protocol.

## Recommendation

**A genuinely untouched holdout cannot be established from the 13 missing train events or the three non-contributing 2021 test folders.**

### Future external holdout (approval gate before any inference)

1. Source years / regions never used in development (e.g. 2022+ US events).
2. Frozen V2 eligibility: ≥7 VIIRS_Day, BA on last input + label day, matched TS, GFS `mask_zero`, crop 256.
3. Exclude any ID in train/val/test manifests or prior audit decision sets.
4. Lock `HOLDOUT_MANIFEST.json` (event list + window keys + hashes) **before** checkpoint inference.
5. Metrics: micro + per-fire macro; use existing val-selected thresholds; no post-hoc fire dropping.

Until then, label the current test population as an **exploratory reused benchmark**.
