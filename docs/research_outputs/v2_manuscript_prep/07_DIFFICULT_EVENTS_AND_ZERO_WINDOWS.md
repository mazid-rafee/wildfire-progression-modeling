# Difficult BA-Sparse Events and Zero-Window Official Fires

Sources: `SUBSET_MEMBERSHIP_AUDIT.csv`, `EXTRA5_E1_DETAIL.csv`, `FIRE_LEVEL_METRICS.csv`, `METRIC_RECONCILIATION.md`.

---

## Official 2021 test roster structure

| Set | Count | Role |
|---|---:|---|
| Official `US_2021_*` test roster | 24 | Naming “full-24” |
| BA-valid contributors (**headline**) | **21** | 561 windows |
| Rule-A 18 | 18 | Secondary subgroup |
| Manuscript-16 | 16 | Secondary subgroup (historically discussed) |

**Primary paper population = full BA-valid 21.** Subgroups never replace it without disclosure.

---

## Three official fires with **no** evaluation windows

| Fire ID | Windows | Notes |
|---|---:|---|
| `US_2021_FL2521008104520210308` | 0 | Non-contributing official test folder |
| `US_2021_NM3340210587120210426` | 0 | Non-contributing official test folder |
| `US_2021_NM3676810505920211120` | 0 | Non-contributing official test folder |

These are **not** scored (cannot contribute TP/FP/FN). They are **not** an untouched holdout. Document in methods as eligibility failures under the locked BA-matched protocol.

CSV: `tables/TABLE_S5_zero_window_official_fires.csv`.

---

## Five difficult BA-sparse events (extra-five vs manuscript-16)

Excluded from manuscript-16; **retained in full-21 headline**.

| Fire ID | In Rule-A 18? | E1 F1 mean±SD | GT+ px mean | Pred+ px mean | Pattern |
|---|---|---:|---:|---:|---|
| `US_2021_AZ3345510938920210616` | Yes | 0.028 ± 0.030 | 552 | 3592 | Small target, large FP |
| `US_2021_AZ3368910927620210616` | Yes | 0.036 ± 0.035 | 559 | 3389 | Small target, large FP |
| `US_2021_MT4714310953420211004` | No | 0.000 ± 0.000 | 0 | 2.4 | **Empty GT progression**; tiny FP |
| `US_2021_NM3323810847220210520` | No | 0.000 ± 0.000 | 2 | 2863 | Near-empty GT, mass FP |
| `US_2021_NM3344410803520210514` | No | 0.000 ± 0.000 | 0 | 0 | Empty TP=FP=FN → F1≈0 by `1e-8` convention |

CSV: `tables/TABLE_S3_extra5_difficult.csv`.

### Membership rules (audit)
- Rule-A 18 = official 24 − {FL25210, NM33402, NM36768, NM33444, NM33238, MT47143}
- Manuscript-16 = Rule-A 18 − {AZ33455, AZ33689}

---

## Empty-target evaluation protocol (supplementary)

**Preserve** full-21 unweighted macro as the primary macro metric.

Additionally report (`tables/TABLE_S1_empty_target_supplement.csv`):

1. **Zero-division convention:** \(F1 = 2TP/(2TP+FP+FN+10^{-8})\); if TP=FP=FN=0 then F1≈0.
2. **Positive-target-only macro** over the 19 fires with `gt_positive_px > 0` (E1: **0.447 ± 0.008** vs full-21 **0.405 ± 0.007**).
3. **Empty-target false alarms** separately: mean sum of FP (and pred positives) on `{MT47143, NM33444}` — E1≈2.4 FP; UNIFORM≈88.6 FP.

Near-empty `NM33238` (2 GT pixels) remains in the positive-target set; discuss under BA-sparse FP failures, not empty-target convention.

---

## Paper wording

> Five BA-sparse test events contribute near-zero per-fire F1 and pull unweighted macro below micro. We keep them in the primary 21-fire evaluation, report manuscript-16 / Rule-A 18 only as secondary robustness checks, and provide an empty-target supplement that isolates the two events with no positive ground-truth progression without replacing the headline macro.
