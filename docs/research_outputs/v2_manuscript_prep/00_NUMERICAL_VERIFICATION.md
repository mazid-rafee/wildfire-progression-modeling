# Independent Numerical Verification

**Scope:** Recompute headline claims from frozen CSVs/JSONs only.  
**Script output:** `artifacts/verification_recompute.json`  
**Constraint honored:** no training, checkpoint edits, or threshold changes.

---

## Integrity

| Check | Result |
|---|---|
| Experimental cells | **20 / 20** (4 arms × 5 seeds) |
| Seeds | 7, 17, 29, 42, 101 |
| Micro F1 recompute vs stored (20 cells) | max abs err **0.0** (≤1e-5 required) |
| Macro F1 recompute vs stored (20 cells) | max abs err **≈8e-17** |
| Eval population | test24 = **561 windows / 21 fires** on every cell |
| Secondary pops | test18=553/18, test16=531/16 |
| Publication-table `issues` column | empty for all 20 cells |

Sources: `PUBLICATION_RESULTS_TABLE.csv`, `METRIC_INTEGRITY_PER_CELL.csv`.

---

## Headline mean ± sample SD (ddof=1, n=5) — full BA-valid test

Independently recomputed from `test24_*` columns:

| Arm | Micro F1 | IoU | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| **E1** | **0.6118 ± 0.0234** | 0.4410 ± 0.0243 | 0.4048 ± 0.0069 | 0.5407 ± 0.0594 | 0.7141 ± 0.0486 |
| ENV0 | 0.5405 ± 0.0189 | 0.3706 ± 0.0177 | 0.3535 ± 0.0173 | 0.5440 ± 0.0410 | 0.5408 ± 0.0406 |
| TS2 | 0.4041 ± 0.0251 | 0.2534 ± 0.0196 | 0.2693 ± 0.0218 | 0.2965 ± 0.0190 | 0.6361 ± 0.0559 |
| UNIFORM | 0.2354 ± 0.0501 | 0.1341 ± 0.0320 | 0.2006 ± 0.0207 | 0.2879 ± 0.0276 | 0.2089 ± 0.0703 |

Matches `FINAL_MULTISEED_REPORT.md`, `PAPER_READY_SUMMARY.md`, and `PUBLICATION_SUMMARY_MEAN_STD.csv` within rounding.

Rounded paper-facing forms (3 decimals) also match the audit:

| Method | Micro F1 |
|---|---:|
| RecencyFusion E1 | **0.612 ± 0.023** |
| ENV-0 | 0.541 ± 0.019 |
| SpaSE TS=2 | 0.404 ± 0.025 |
| Frozen-uniform | 0.235 ± 0.050 |

---

## Paired seed-wise Δ micro F1 (E1 − baseline)

| Contrast | Seeds | Mean Δ ± SD | Per-seed Δ |
|---|---|---:|---|
| E1 − TS2 | 5 | **+0.2077 ± 0.0323** | 7:+0.170, 17:+0.233, 29:+0.215, 42:+0.178, 101:+0.242 |
| E1 − ENV0 | 5 | **+0.0712 ± 0.0290** | 7:+0.036, 17:+0.077, 29:+0.104, 42:+0.092, 101:+0.047 |
| E1 − UNIFORM (all 5) | 5 | **+0.3763 ± 0.0494** | 7:+0.408, 17:+0.333, 29:+0.447, 42:+0.339, 101:+0.355 |
| E1 − UNIFORM (**paired**) | 7,17,29,101 | **+0.3857 ± 0.0517** | excludes 42 |
| E1 − UNIFORM seed 42 | 42 | +0.3388 | **UNPAIRED** |

Nested fire-cluster bootstrap 95% CIs (from frozen `FINAL_bootstrap.json`, B=2000):

| Contrast | Point | 95% CI |
|---|---:|---|
| E1 − TS2 | +0.2077 | [0.164, 0.241] |
| E1 − ENV0 | +0.0712 | [0.007, 0.116] |
| E1 − UNIFORM all 5 | +0.3763 | [0.252, 0.459] |
| E1 − UNIFORM paired | +0.3857 | [0.254, 0.467] |

---

## Init pairing labels (verified)

| Seed | E1 label | UNIFORM label | Causal status |
|---:|---|---|---|
| 7, 17, 29, 101 | `PAIRED_exporter` | `PAIRED_via_exported_backbone` | **PAIRED** |
| 42 | `HISTORICAL_no_export` | `UNPAIRED` | **UNPAIRED** |

---

## Micro vs macro (E1)

- Micro mean: **0.6118**
- Macro mean (unweighted 21-fire): **0.4048**
- Gap is a weighting / heterogeneity effect, not an aggregation bug (integrity errs = 0).

Subset robustness (E1), recomputed match to audit:

| Subset | Micro | Macro |
|---|---:|---:|
| Full-21 | 0.612 ± 0.023 | 0.405 ± 0.007 |
| Rule-A 18 | 0.616 ± 0.022 | 0.472 ± 0.008 |
| Manuscript-16 | 0.627 ± 0.018 | 0.527 ± 0.008 |

---

## Empty-target supplementary (does not replace headline)

Two fires have mean `gt_positive_px = 0` across seeds: `MT47143…`, `NM33444…` (19 fires retain positives).

| Arm | Full-21 macro (primary) | Positive-target-only macro (n=19) | Empty-target mean FP sum |
|---|---:|---:|---:|
| E1 | 0.4048 ± 0.0069 | 0.4474 ± 0.0076 | 2.4 |
| ENV0 | 0.3535 ± 0.0173 | 0.3907 ± 0.0191 | 1.2 |
| TS2 | 0.2693 ± 0.0218 | 0.2976 ± 0.0241 | 1.6 |
| UNIFORM | 0.2006 ± 0.0207 | 0.2218 ± 0.0229 | 88.6 |

Convention: `F1 = 2·TP / (2·TP + FP + FN + 1e-8)`; TP=FP=FN=0 → F1≈0.

---

## Temporal weights (from frozen `run_summary.json`)

E1 newest-frame weight range across seeds: **0.584 – 0.640**.  
UNIFORM: exactly **1/6** on all six slots for all five seeds.

---

## Verdict

All audited headline numerical claims used in this manuscript package **reproduce**. No integrity failures were found in the frozen evaluation population.
