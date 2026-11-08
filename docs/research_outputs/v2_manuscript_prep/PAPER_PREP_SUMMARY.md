# Paper Preparation Summary

**Package root:** `outputs/v2_manuscript_prep/`  
**Frozen evidence:** V2 multi-seed (20/20 cells) + validation audit  
**Independent verification:** all headline numbers reproduce (`00_NUMERICAL_VERIFICATION.md`)

---

## Proposed paper structure

1. **Abstract** — E1 vs TS2 / ENV-0 / frozen-uniform; full-21 micro+macro; paired fusion ablation; reused-benchmark disclosure  
2. **Introduction** — next-step progression; temporal aggregation; V2 multi-seed motivation  
3. **Related work** — author-curated citations only (placeholder themes in outline)  
4. **Methodology** — SpaSE mean pool; globally learned RecencyFusion (≠ attention); frozen-uniform control; ENV-0 mask  
5. **Experimental setup** — V2.0-corrected + `mask_zero`; 4×5 matrix; val-only thresholds; 21 fires / 561 windows; init pairing  
6. **Main results** — Table 1 full-21 mean±SD + nested fire CIs; micro and macro  
7. **Ablations** — paired E1−UNIFORM (primary); E1−TS2 / E1−ENV0 (secondary); empty-target supplement; subset robustness  
8. **Qualitative analysis** — successes, misses, FP-heavy, ENV-0 wins, E1 vs UNIFORM  
9. **Discussion** — what the gaps imply; fire heterogeneity  
10. **Limitations** — reuse, no external holdout, n=21 clusters, unpaired seed 42, pixel-not-km²  
11. **Reproducibility** — configs, SHA256, tests, protocol locks  

Full draft paragraphs: `02_MANUSCRIPT_OUTLINE.md`.

---

## Main contributions (supported)

1. **Corrected V2 multi-seed evaluation** of next-step BA progression on a locked matched population, separated from historical V1 runs.  
2. **RecencyFusion (E1):** minimal globally learned temporal mixture with recency prior on a shared SpaSE backbone—explicitly distinguished from input-adaptive attention.  
3. **Causal-leaning temporal ablation:** learnable vs frozen-uniform fusion under **identical backbone initialization** on seeds 7/17/29/101.  
4. **Transparent metric reporting:** micro and unweighted per-fire macro on the **full 21-fire** BA-valid test set, with documented BA-sparse / empty-target behavior.  
5. **Honest generalization posture:** exploratory reused benchmark; untouched holdout not available in-corpus.

---

## Supported numerical claims (verified)

| Claim | Evidence |
|---|---|
| E1 micro F1 **0.612 ± 0.023** (5 seeds, 21 fires, 561 windows) | TABLE1 / recompute |
| ENV-0 **0.541 ± 0.019**; TS2 **0.404 ± 0.025**; UNIFORM **0.235 ± 0.050** | TABLE1 |
| E1 macro F1 **0.405 ± 0.007**; micro≻macro due to size imbalance + 5 hard fires | TABLE4, EXTRA5 |
| E1−TS2 Δ **+0.208 ± 0.032**; nested CI **[0.164, 0.241]** | TABLE2 |
| E1−ENV0 Δ **+0.071 ± 0.029**; nested CI **[0.007, 0.116]** | TABLE2 |
| E1−UNIFORM paired-4 Δ **+0.386 ± 0.052**; nested CI **[0.254, 0.467]** | TABLE2/TABLE3 |
| Seed-42 E1−UNIFORM unpaired Δ **+0.339** | TABLE3 |
| E1 newest-frame weight ≈ **0.58–0.64**; UNIFORM flat **1/6** | TABLE_S4 |
| Ordering E1 ≻ ENV0 ≻ TS2 ≻ UNIFORM on Rule-A 18 and manuscript-16 | TABLE_S2 |
| Metric integrity 20/20 cells (micro/macro abs err ≤1e-5) | verification JSON |

---

## Required figures and tables

### Main
| Item | Content |
|---|---|
| **Table 1** | Full-21 comparison mean±SD (+ nested micro CI) |
| **Table 2** | Paired contrasts; mark paired-4 as primary fusion ablation |
| **Table 3** | Per-seed E1 vs UNIFORM with PAIRED/UNPAIRED labels |
| **Fig. micro–macro** | Existing `micro_vs_macro_f1.*` |
| **Fig. size–F1** | Existing `e1_f1_vs_hist_size.*` (highlight extra-five) |
| **Fig. qualitative** | 6–8 panels from `08_QUALITATIVE_CASE_INDEX.md` |
| **Fig. temporal weights** | Bar/line from TABLE_S4 (to be plotted for camera-ready) |

### Supplement
| Item | Content |
|---|---|
| Table S1 | Empty-target / positive-only macro (headline macro preserved) |
| Table S2 | Subset robustness |
| Table S3 | Extra-five difficult events |
| Table S4 | Learned temporal weights |
| Table S5 | Three zero-window official fires |
| Extra qualitative | Remaining audit figures |

---

## Unresolved publication blockers

1. **No untouched external holdout** — largest scientific integrity blocker; must disclose reused test or run a locked external manifest.  
2. **Related-work citations** not assembled (must not invent).  
3. **Venue decision** on whether exploratory reused-benchmark results are acceptable without external test.  
4. **Prose claim audit** still needed after drafting (see `05_UNSUPPORTED_CLAIMS.md`).  
5. **Camera-ready figure polishing** (temporal-weight plot; panel selection/layout).  
6. **Data/code availability + compute statements** for target venue.

Non-blockers already closed: 20-cell completeness, metric integrity, init-pairing documentation, micro/macro explanation, empty-target convention, V1/V2 separation text, qualitative real-map inventory.

---

## Constraints honored this phase

- No training, checkpoint edits, threshold retuning, or evaluation-population changes  
- Existing audit/multiseed artifacts preserved  
- No Git commit or push  
