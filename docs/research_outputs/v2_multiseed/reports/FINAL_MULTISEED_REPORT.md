# V2 Multi-Seed Final Analysis Report

**Protocol:** `V2.0-corrected-protocol` (locked)  
**Matrix:** 4 arms × 5 seeds = **20 cells** (all complete)  
**Generated from frozen run artifacts only** — no retraining, no checkpoint edits, no protocol changes.  
**Test population label:** *exploratory reused benchmark* (see §8).

---

## 1. Completeness verification

| Check | Result |
|---|---|
| Expected cells | 20 |
| Loaded with `run_summary.json` | **20 / 20** |
| Missing / incomplete metrics | **none** |
| Arithmetic recompute of micro F1/IoU/P/R from TP/FP/FN | **all match** (≤1e-5) |
| Eval population sizes | test24=561, test18=553, test16=531 on all cells |
| Test24 fire count | 21 on all cells |
| Integrity issue list | **empty** |

Per-cell details: `FINAL_integrity.json`, `PUBLICATION_RESULTS_TABLE.csv`.

---

## 2. Checkpoint / config / threshold integrity

Verified per cell (see `PUBLICATION_RESULTS_TABLE.csv` for `ckpt_sha256`):

- `checkpoints/best.pth` present; SHA256 recorded.
- Where `train_result.json` / summary recorded a checkpoint hash, it matches the on-disk file.
- Forecast policy `mask_zero`; protocol id `V2.0-corrected-protocol` where recorded.
- Threshold selection: validation-only, grid **0.01–0.99** (99 steps), `tune_on_test=false`.
- Selected thresholds lie on the grid for all 20 cells.
- **UNIFORM** learned/reported temporal weights remain flat **1/6** (frozen fusion).

No silent substitutions were made.

---

## 3. Mean ± sample SD across five seeds (test24)

Sample SD uses **ddof=1** (n=5).

| Arm | Micro F1 | IoU | Macro F1 (unweighted per-fire) | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| **E1** | **0.6118 ± 0.0234** | 0.4410 ± 0.0243 | 0.4048 ± 0.0069 | 0.5407 ± 0.0594 | 0.7141 ± 0.0486 |
| **TS2** | 0.4041 ± 0.0251 | 0.2534 ± 0.0196 | 0.2693 ± 0.0218 | 0.2965 ± 0.0190 | 0.6361 ± 0.0559 |
| **ENV0** | 0.5405 ± 0.0189 | 0.3706 ± 0.0177 | 0.3535 ± 0.0173 | 0.5440 ± 0.0410 | 0.5408 ± 0.0406 |
| **UNIFORM** | 0.2354 ± 0.0501 | 0.1341 ± 0.0320 | 0.2006 ± 0.0207 | 0.2879 ± 0.0276 | 0.2089 ± 0.0703 |

**Nested fire-cluster bootstrap** on seed-averaged micro F1 (resample fires; for each seed recompute micro from sampled fires’ TP/FP/FN; average over seeds; B=2000), 95% CI:

| Arm | Point (mean seed micro) | 95% CI |
|---|---:|---|
| E1 | 0.6118 | [see `FINAL_bootstrap.json`] |
| TS2 | 0.4041 | … |
| ENV0 | 0.5405 | … |
| UNIFORM | 0.2354 | … |

Exact CIs: `FINAL_bootstrap.json` → `nested_seed_micro_bootstrap`.

---

## 4. Paired seed-wise differences (test24 micro F1)

Δ = E1 − baseline, then mean ± SD over seeds.

| Contrast | Seeds | Mean Δ ± SD | Per-seed Δ |
|---|---|---:|---|
| E1 − TS2 | 5 | **+0.2077 ± 0.0323** | 7: +0.170, 17: +0.233, 29: +0.215, 42: +0.178, 101: +0.242 |
| E1 − ENV0 | 5 | **+0.0712 ± 0.0290** | 7: +0.036, 17: +0.077, 29: +0.104, 42: +0.092, 101: +0.047 |
| E1 − UNIFORM | 5 | **+0.3763 ± 0.0494** | 7: +0.408, 17: +0.333, 29: +0.447, 42: +0.339, 101: +0.355 |
| E1 − UNIFORM (**paired only**) | 7,17,29,101 | **+0.3857 ± 0.0517** | (excludes seed 42) |
| E1 − UNIFORM seed42 only | 42 | +0.3388 | **UNPAIRED init** |

---

## 5. Initialization pairing (E1 ↔ UNIFORM)

| Seed | Pairing status | Notes |
|---:|---|---|
| 7, 17, 29, 101 | **Paired** | E1 exported backbone init; UNIFORM loaded same file then froze fusion |
| **42** | **UNPAIRED** | Historical E1 seed-42 run never exported init; UNIFORM used reconstructed `set_seed(42)` init only — **do not treat as identical-init causal contrast** |

Primary causal claim for learnable vs frozen-uniform temporal fusion should cite **paired seeds 7/17/29/101** (mean Δ **+0.386**). Seed-42 may be shown as a supplementary unpaired point.

---

## 6. Per-fire paired differences & cluster bootstrap

**Design:** Same 21 test fires appear in every seed. Seeds are repeated measures on fires; pixels/windows are **not** treated as independent.

### Fire-level statistic
For each fire, average F1 (or E1−baseline Δ) across seeds, then **resample fires** with replacement (B=5000) for CIs on the fire-mean.

### Nested paired micro bootstrap (recommended for headline Δ)
Resample fires; for each seed recompute micro F1 from pooled TP/FP/FN of the sampled fire multiset; take seed-mean of (E1−baseline); B=2000.

| Contrast | Point mean Δ (seed micro) | 95% CI (nested fire bootstrap) |
|---|---:|---|
| E1 − TS2 | +0.2077 | **[0.164, 0.241]** |
| E1 − ENV0 | +0.0712 | **[0.007, 0.116]** |
| E1 − UNIFORM (all 5) | +0.3763 | **[0.253, 0.459]** |
| E1 − UNIFORM (paired excl. 42) | +0.3857 | **[0.254, 0.467]** |

Per-fire mean Δ tables: `FINAL_per_fire_paired_deltas.json`.

---

## 7. Validation→test gaps & fire heterogeneity

### Val (selected thr) − test24 micro F1
| Arm | Mean gap ± SD |
|---|---:|
| E1 | 0.034 ± 0.014 |
| TS2 | 0.098 ± 0.017 |
| ENV0 | **0.129 ± 0.022** |
| UNIFORM | **0.159 ± 0.052** |

ENV0 and UNIFORM show larger optimistic val→test shifts than E1 (consistent with earlier single-seed ENV0 observation).

### Fire heterogeneity (seed-mean per-fire F1)
Several fires remain near-zero for all arms (e.g. `MT47143…`, `NM33238…`, `NM33444…`), so micro F1 is driven by larger fires. Macro F1 ≪ micro F1 for E1 (0.40 vs 0.61), confirming size imbalance. Details: `FINAL_fire_heterogeneity.json`.

E1 temporal weights (all seeds) concentrate on the newest frame (~0.58–0.64), while UNIFORM is fixed at 1/6 — aligning with the large E1−UNIFORM gap.

---

## 8. Limitation — exploratory test reuse

The matched **test24 / test18 / test16** populations were used during V2 protocol development and single-seed exploration. This multi-seed study therefore reports an **exploratory reused test benchmark**, not an untouched final holdout.

**Do not claim** independent external validation or definitive statistical significance from these intervals alone. Prefer a future held-out event population for final claims. Bootstrap CIs quantify fire-sampling uncertainty **within this fixed exploratory test set**.

---

## 9. Publication-ready tables (artifacts)

| File | Contents |
|---|---|
| `PUBLICATION_RESULTS_TABLE.csv` | All 20 cells: metrics, thr, ckpt SHA, pairing label, run dir |
| `PUBLICATION_SUMMARY_MEAN_STD.csv` | Arm-level mean±SD + bootstrap CI fields |
| `RESULTS_TABLE.csv` / `.json` | Refreshed operational table |
| `FINAL_aggregates.json` | Full aggregates + paired seed diffs |
| `FINAL_bootstrap.json` | Bootstrap methods + CIs |
| `FINAL_val_test_gaps.json` | Val−test gaps |
| `FINAL_fire_heterogeneity.json` | Per-arm fire difficulty |
| `FINAL_per_fire_paired_deltas.json` | Per-fire mean Δ |
| `FINAL_integrity.json` | Verification log (no issues) |

### Compact headline table (test24 micro F1)

| Method | Mean ± SD (5 seeds) |
|---|---:|
| RecencyFusion E1 (TS=6) | **0.612 ± 0.023** |
| SpaSE mean-pool TS=2 | 0.404 ± 0.025 |
| ENV-0 (VIIRS+BA only) | 0.541 ± 0.019 |
| Frozen-uniform RecencyFusion TS=6 | 0.235 ± 0.050 |

Paired E1−UNIFORM (init-paired seeds only): **+0.386 ± 0.052** micro F1; nested fire bootstrap 95% CI **[0.254, 0.467]**.

---

## 10. Preservation

- No experiment run directories, checkpoints, or frozen seed-42 artifacts were modified.
- Analysis outputs written only under `outputs/v2_multiseed/reports/`.
- No Git commit or push.

---

## Bottom line

All **20** cells are complete and internally consistent. Under the locked V2 protocol on this exploratory test set, **E1 outperforms TS2, ENV-0, and frozen-uniform** with stable multi-seed gaps; the largest, best-controlled contrast is **learnable vs frozen-uniform temporal fusion** on seeds **7/17/29/101**.
