# V2 Paper-Quality Validation Audit

**Scope:** Analysis-only post-training audit of the completed 20-cell V2 multi-seed study.  
**Constraints honored:** no training, no checkpoint edits, no threshold retuning, no DATA_ROOT mutation, no git commits.  
**Output root:** `outputs/v2_validation_audit/`

---

## 1. Artifact verification

| Check | Result |
|---|---|
| Experimental cells with complete metrics | **20 / 20** |
| Seed-42 reused E1 / TS2 / ENV0 checkpoint SHA256 vs `RESULTS_TABLE` | **match** (`artifacts/CHECKPOINT_SHA256_AUDIT.csv`) |
| Eval keys / populations | test24=561, test18=553, test16=531 on all cells |
| Contributing BA-valid test fires | **21** on all cells |
| Forecast policy | `mask_zero` (GFS ch 22–25); ENV-0 additionally masks ch 8–25 |
| Checkpoint / threshold selection | val F1@0.5 for ckpt; thr grid 0.01–0.99 **validation-only** |
| Metric definitions | micro = pooled TP/FP/FN; macro = **unweighted mean of per-fire F1** |

Official naming “full-24” refers to the 24 `US_2021_*` test roster. Only **21** events contribute BA-matched windows.

---

## 2. Subset membership (24 / 21 / 18 / 16)

See `SUBSET_MEMBERSHIP_AUDIT.csv`.

### Three official test fires with **zero** BA-valid windows
- `US_2021_FL2521008104520210308`
- `US_2021_NM3340210587120210426`
- `US_2021_NM3676810505920211120`

### Five fires in full-21 but **excluded from manuscript-16**
| Fire ID | Role |
|---|---|
| `US_2021_AZ3345510938920210616` | In Rule-A 18; excluded from manuscript-16 |
| `US_2021_AZ3368910927620210616` | In Rule-A 18; excluded from manuscript-16 |
| `US_2021_MT4714310953420211004` | Excluded from Rule-A 18 and manuscript-16 |
| `US_2021_NM3323810847220210520` | Excluded from Rule-A 18 and manuscript-16 |
| `US_2021_NM3344410803520210514` | Excluded from Rule-A 18 and manuscript-16 |

Rule-A 18 = official 24 − {FL, NM33402, NM36768, NM33444, NM33238, MT47143}.  
Manuscript 16 = Rule-A 18 − {AZ33455, AZ33689}.

---

## 3. Priority A — Micro vs macro F1

**Five-seed E1 means:** micro F1 **0.612** vs macro F1 **0.405**.

### Verified cause (not a bug)
1. Micro pools pixels → large western fires dominate.
2. Macro averages 21 per-fire F1 values equally → five near-zero / FP-dominated events pull the mean down.
3. Independent recompute of micro and macro from TP/FP/FN matches stored artifacts within **1e-5** (`METRIC_RECONCILIATION.md`, `artifacts/METRIC_INTEGRITY_PER_CELL.csv`).

### Seed-42 arithmetic (manuscript-16 vs full-21)
| Quantity | Value |
|---|---:|
| Mean F1 manuscript-16 | 0.539290 |
| Mean F1 full-21 (macro) | 0.416629 |
| Mean F1 of five excluded | 0.024112 |
| Implied `(21·m21−16·m16)/5` | 0.024112 |

### Extra-five E1 performance (mean ± SD over seeds)
| Fire | F1 mean | GT+ px mean | Pred+ px mean | Notes |
|---|---:|---:|---:|---|
| AZ33455… | 0.028 | 552 | 3592 | Small target, large FP |
| AZ33689… | 0.036 | 559 | 3389 | Small target, large FP |
| MT47143… | 0.000 | 0 | ~2 | Empty GT progression |
| NM33238… | 0.000 | 2 | ~2863 | Near-empty GT, mass FP |
| NM33444… | 0.000 | 0 | 0 | Empty TP=FP=FN → F1=0 by `1e-8` convention |

**Do not drop these fires from the headline full evaluation.** Report subgroups separately (Rule-A 18 / manuscript-16).

Full per-fire table: `FIRE_LEVEL_METRICS.csv` (21 fires × 5 seeds × 4 arms).

---

## 4. Priority B — Qualitative maps

Real TS-SatFire VIIRS / BA / frozen seed-42 predictions (CPU-built windows, GPU-4 inference).  
Figures: `figures/qualitative/` · manifest: `artifacts/QUALITATIVE_FIGURE_MANIFEST.csv`.

Selection rules (reproducible; not cherry-picked for aesthetics only):
- **high_e1:** top-quartile E1 window F1 among GT>0, then max mass
- **median_e1:** median window F1 among GT>0
- **low_e1 / extra5:** lowest E1 F1 (preserve difficult events)
- **env0_beats_e1:** max ENV0−E1 among low-E1 candidates
- **e1_beats_uniform:** high E1 with large Δ vs UNIFORM

Panels (image-coordinate 256×256 crops; **not** claimed georeferenced maps): VIIRS last history, prev BA, GT progression, E1 heatmap, E1 binary, UNIFORM, SpaSE TS2, ENV-0, TP/FP/FN overlay.

**Patterns:** On large active fronts, E1 and often TS2 recover progression tightly; UNIFORM under-calls. ENV-0 can beat E1 on some mid-size windows. Extra-five cases show empty / near-empty targets or FP fill — consistent with metric tables.

---

## 5. Priority C — Fire size / progression

Pixel resolution probe: all 21 fires share **EPSG:4326**, pixel size ≈ **0.00337°**, crop **256×256**. Physical area (km²) is **not** reported; analyses use **pixel counts / fractions**.

Progression rate = `new_ba_px / hist_ba_px` (NaN if hist=0).

Descriptive Pearson correlations (E1 seed-mean per-fire F1 vs characteristics):
| Pair | r |
|---|---:|
| F1 vs hist BA (mean px) | 0.665 |
| F1 vs new BA sum | 0.727 |
| Recall vs progression rate | 0.338 |
| FP vs prevalence | 0.421 |
| (E1−ENV0) vs progression | 0.157 |
| (E1−UNIFORM) vs hist size | 0.585 |

Prespecified hist-size tertiles (`artifacts/SIZE_BIN_SUMMARY.csv`):
| Bin | n | Mean E1 F1 |
|---|---:|---:|
| small | 7 | 0.159 |
| medium | 7 | 0.429 |
| large | 7 | 0.627 |

These are **descriptive**, not causal, and were not used to tune models.

Figures: `figures/e1_f1_vs_hist_size.*`, `e1_f1_vs_new_ba.*`, `e1_recall_vs_progression.*`, `e1_minus_*`.

---

## 6. Priority D — Generalization / uncertainty

Seeds: 42, 7, 17, 29, 101.  
E1↔UNIFORM init pairing: **paired** on 7/17/29/101; **unpaired** on 42.

Nested fire-cluster bootstrap (21 fires as primary units; seeds as repeated measures) reproduced; point estimates match `FINAL_bootstrap.json` exactly for nested seed micro (`artifacts/BOOTSTRAP_VS_PUBLISHED.json`). CI endpoints differ slightly by RNG (expected).

Subset robustness (E1 mean micro / macro):
| Subset | Micro | Macro |
|---|---:|---:|
| Full-21 | 0.612 ± 0.023 | 0.405 ± 0.007 |
| Rule-A 18 | 0.616 ± 0.022 | 0.472 ± 0.008 |
| Manuscript-16 | 0.627 ± 0.018 | 0.527 ± 0.008 |

Ordering E1 ≻ ENV0 ≻ TS2 ≻ UNIFORM is preserved. Details: `GENERALIZATION_ANALYSIS.md`.

---

## 7. Priority E — Untouched holdout feasibility

The Phase-1 **13 missing train IDs** are official **train-roster** events with **0** TS=6 matched windows — not an independent holdout.  
The three non-contributing 2021 test folders lack BA-valid windows.

**Conclusion:** A genuinely untouched evaluation population **cannot** be constructed from these candidates under the locked BA-matched protocol. External later-year / disjoint events would be required, locked in a manifest **before** any inference (`UNTOUCHED_HOLDOUT_FEASIBILITY.md`).

---

## 8. Evaluation bugs?

**None found** in metric aggregation, subset filtering, checkpoint reuse, or threshold provenance. The micro–macro gap is a **weighting / fire-heterogeneity** phenomenon, amplified by five difficult BA-sparse events.

---

## 9. Deliverable index

| ID | Path |
|---|---|
| A | `VALIDATION_AUDIT_REPORT.md` (this file) |
| B | `FIRE_LEVEL_METRICS.csv` |
| C | `SUBSET_MEMBERSHIP_AUDIT.csv` |
| D | `METRIC_RECONCILIATION.md` |
| E | `figures/` (+ `figures/qualitative/`) |
| F | `GENERALIZATION_ANALYSIS.md` |
| G | `UNTOUCHED_HOLDOUT_FEASIBILITY.md` |
| H | `PAPER_READY_SUMMARY.md` |
