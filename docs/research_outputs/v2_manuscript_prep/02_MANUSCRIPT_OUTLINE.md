# Manuscript Outline with Evidence-Linked Draft Paragraphs

**Working title (descriptive, not final):**  
*Recency-Aware Temporal Fusion for Next-Step Wildfire Progression Prediction under a Corrected Multi-Seed Evaluation Protocol*

**Venue posture:** methods + ablation paper on matched BA-valid next-step progression; exploratory reused 2021 US test benchmark.

Evidence keys use frozen paths under `outputs/v2_multiseed/` and `outputs/v2_validation_audit/`.

---

## Abstract (draft skeleton)

We study next-step burned-area progression prediction from multi-day VIIRS/BA/FirePred cubes under a locked corrected V2 preprocessing protocol. Building on a SpaSE-style 3-D U-Net with equal temporal mean pooling, we replace the final temporal collapse with a **globally learned** recency-aware soft mixture (RecencyFusion / E1) and compare it to (i) SpaSE mean-pool with shorter history (TS=2), (ii) an ENV-0 satellite+BA channel ablation, and (iii) a frozen-uniform temporal-fusion control under identical backbone initialization. On the full matched BA-valid 2021 test population (**21 fires, 561 windows**; five seeds), E1 achieves micro F1 **0.612 ± 0.023** versus **0.404 ± 0.025** (TS2), **0.541 ± 0.019** (ENV-0), and **0.235 ± 0.050** (frozen-uniform). The primary architectural contrast—learnable versus frozen-uniform fusion on four initialization-paired seeds—yields Δ micro F1 **+0.386 ± 0.052** (nested fire-cluster 95% CI **[0.254, 0.467]**). We report both micro and unweighted per-fire macro metrics, document BA-sparse failure modes, and disclose that the test events were reused during protocol development; an untouched external holdout has not been completed.

---

## 1. Introduction

### Claims this section may make
- Next-step progression from remote sensing cubes is operationally relevant and methodologically challenging under sparse BA labels.
- Temporal aggregation choices matter; equal averaging can dilute recent active-front signal.
- Prior single-seed / V1-protocol results are not interchangeable with the locked V2 multi-seed matrix.

### Claims this section must **not** make
- Nationwide operational readiness; km² skill; cartographic map products.
- That the present test set is a blind holdout.
- That E1 is “attention.”

### Draft paragraph
Wildfire progression forecasting from daily remote-sensing stacks requires aggregating short temporal histories into a next-day burned-area increment. Equal temporal mean pooling is a strong, parameter-free baseline, but it assigns identical weight to older and newer frames. We evaluate a minimal alternative: a **globally parameterized** soft temporal mixture with a fixed recency prior (RecencyFusion), sharing the SpaSE spatial backbone. Under the corrected V2 matched protocol and a five-seed matrix, we isolate the effect of learnable versus frozen-uniform temporal fusion with initialization pairing, and we report both pixel-pooled and fire-equal metrics on the full BA-valid test population of 21 events.

---

## 2. Related Work

### Status
**Needs author-curated citations.** Do not invent references. Placeholder themes only:

- Remote-sensing wildfire detection / burned-area mapping
- Data-driven fire spread / progression models
- Temporal fusion and 3-D U-Nets in Earth observation
- SpaSE / DeepFire-Forecaster attribution (`docs/ATTRIBUTION.md`)

---

## 3. Methodology

### 3.1 Task and labels
Predict next-step BA progression within a 256×256 crop from a length-\(T\) input stack (27 channels). Evaluation uses validation-selected thresholds; `tune_on_test=false`.

### 3.2 Temporal operators
See `01_METHOD_MATH.md`. Draft wording:

> SpaSE mean pooling collapses decoder features by equal averaging over time. RecencyFusion replaces that operator with \(w=\mathrm{softmax}(\ell + s\,p)\), where \(\ell\) and \(s\) are learned globally and \(p\) is a fixed increasing prior. This is **not** input-adaptive attention: the same weights apply to every sample. The frozen-uniform control zeros and freezes \((\ell,s)\), forcing \(w_t=1/T\) while retaining the RecencyFusion harness.

### 3.3 Experimental arms

| Arm | Model | \(T\) | Channels | Role |
|---|---|---:|---|---|
| E1 | RecencyFusion | 6 | full 27 (GFS `mask_zero`) | Proposed |
| TS2 | MeanPool | 2 | full 27 | Short-history SpaSE baseline |
| ENV0 | RecencyFusion | 6 | VIIRS+BA only (mask 8–25) | Env-channel necessity |
| UNIFORM | RecencyFusion frozen \(1/T\) | 6 | full 27 | Temporal-fusion control |

### 3.4 Corrected V2 protocol
Geo-aligned FP→VIIRS shared crop; directional augmentation remap; matched-train FP stats; threshold grid 0.01–0.99 on validation only; forecast GFS channels 22–25 set to `mask_zero` for the multi-seed matrix. Distinct from frozen V1 (`03_V2_PROTOCOL_AND_V1_DISTINCTION.md`).

---

## 4. Experimental Setup

### Draft paragraph
We train four arms × five seeds \(\{7,17,29,42,101\}\) under shared V2 locks (`configs/experiments/v2/_shared_train.yaml`). Checkpoints are selected by validation F1 at 0.5; decision thresholds are swept on validation only (0.01–0.99). The primary test population is the matched BA-valid subset of the official 2021 US test roster: **21 fires contribute 561 windows**. Three official roster fires contribute zero BA-valid windows and are documented but excluded from metric denominators. For E1 versus UNIFORM, seeds 7/17/29/101 share exported backbone initializations; seed 42 is unpaired and reported separately. Metrics: micro F1/IoU/P/R from pooled TP/FP/FN; macro F1 as the unweighted mean of 21 per-fire F1 values. Uncertainty uses nested fire-cluster bootstrap with fires as primary units and seeds as repeated measures.

### Reproducibility bullets
- Protocol id: `V2.0-corrected-protocol` + `mask_zero`
- Population counts: train 983 / val 135 / test24 561
- Checkpoint SHA256 per cell: `tables/TABLE3_*.csv`, `PUBLICATION_RESULTS_TABLE.csv`
- Architectural parity: `tests/v2/test_frozen_uniform_fusion.py`

---

## 5. Main Results

### Primary table
`tables/TABLE1_main_comparison_full21.csv`

### Draft paragraph
On the full matched BA-valid test set, RecencyFusion E1 attains micro F1 **0.612 ± 0.023** (five-seed mean ± SD), exceeding ENV-0 (**0.541 ± 0.019**), SpaSE TS=2 (**0.404 ± 0.025**), and frozen-uniform (**0.235 ± 0.050**). Nested fire-cluster bootstrap 95% CIs for seed-mean micro F1 are reported in Table 1. Unweighted per-fire macro F1 for E1 is **0.405 ± 0.007**, substantially below micro F1 because large active fronts dominate pixel-pooled scores while five BA-sparse events contribute near-zero per-fire F1. We retain the full 21-fire population for primary reporting.

### Paired contrasts
`tables/TABLE2_paired_contrasts.csv` — include both all-seed and paired-4 E1−UNIFORM.

---

## 6. Ablation Analysis

### 6.1 Primary: learnable vs frozen-uniform fusion
`tables/TABLE3_e1_uniform_init_pairing.csv`

> Restricting to the four initialization-paired seeds, E1 improves micro F1 by **+0.386 ± 0.052** over frozen-uniform fusion (nested fire bootstrap 95% CI **[0.254, 0.467]**). Seed 42 shows a same-direction unpaired gap of +0.339 and is labeled separately. Learned E1 weights place ≈0.58–0.64 mass on the newest frame, whereas UNIFORM remains flat at 1/6 (`TABLE_S4`).

### 6.2 Secondary: history length (E1 vs TS2) and channels (E1 vs ENV-0)
Report as secondary; they are not identical-init temporal-operator ablations.

### 6.3 Micro vs macro and empty-target supplement
`TABLE4`, `TABLE_S1`. Preserve full-21 macro as headline; positive-target-only (n=19) is supplementary.

### 6.4 Subset robustness
`TABLE_S2`. Ordering E1 ≻ ENV0 ≻ TS2 ≻ UNIFORM preserved on Rule-A 18 and manuscript-16. Do not replace full-21 headlines with manuscript-16 without disclosure.

---

## 7. Qualitative Analysis

Index: `08_QUALITATIVE_CASE_INDEX.md`, figures under `outputs/v2_validation_audit/figures/qualitative/`.

Suggested main-text panel set (6–8):
1. Success / large front: MT45790 L85 (E1≈0.98; UNIFORM≈0.03)
2. Success: MT45683 L73 or WA48568 L29
3. Median difficulty: CA34517 L10 or ID46638 L19
4. ENV-0 advantage: ID44532 L10 (ENV0≈0.74 ≫ E1≈0.35)
5. E1 vs UNIFORM gap: ID45585 L22 (E1≈0.97 vs U≈0.06)
6. False-positive / BA-sparse: NM33238 L19; AZ33455 L9
7. Empty target: MT47143 L6; NM33444 L21

Disclose: image-coordinate 256×256 crops; not georeferenced maps; thresholds = frozen validation-selected values; seed-42 checkpoints used for maps per audit.

---

## 8. Discussion

### Supported discussion points
- Learnable global recency weighting materially improves over frozen uniform averaging under paired inits.
- Environmental channels help on average (E1 > ENV-0) but are not uniformly helpful (ENV-0 wins on some windows).
- Micro/macro divergence reflects fire-size imbalance and BA-sparse events, not metric bugs.
- Val→test gaps are larger for ENV-0 and UNIFORM than E1 (`FINAL_val_test_gaps.json`).

### Unsupported leaps (see `05_UNSUPPORTED_CLAIMS.md`)
- Causal claims about meteorology vs vegetation from ENV-0 alone
- Operational deployment skill
- Geographic regional superiority beyond the listed US_2021 event IDs

---

## 9. Limitations

1. Test events reused during V2 development — exploratory reused benchmark, not blind holdout.
2. Untouched external holdout not completed; Phase-1 missing train IDs are ineligible.
3. Only 21 fire clusters → wide fire-bootstrap CIs.
4. Windows within a fire are dependent.
5. Seed-42 E1↔UNIFORM unpaired.
6. Macro sensitive to empty/near-empty targets; F1≈0 when TP=FP=FN=0.
7. Pixel counts on a degree grid; no km² claims.
8. Qualitative panels are image crops, not cartographic products.
9. SpaSE TS=6 mean-pool is archival only (dilution limitation from prior audit)—not a multi-seed headline arm.

---

## 10. Reproducibility

- Code: `src/wtp/models/`, `scripts/v2_train_eval.py`, `configs/experiments/v2/`
- Protocol docs: `docs/V1_VS_V2_PROTOCOL.md`, `configs/protocol/v2_corrected*.yaml`
- Frozen results: `outputs/v2_multiseed/reports/PUBLICATION_RESULTS_TABLE.csv`
- Audit: `outputs/v2_validation_audit/`
- Tests: `tests/v2/test_frozen_uniform_fusion.py`, `tests/v2_integration/`
- No Git commit was created as part of this preparation package.

---

## Figure / table inventory for camera-ready drafting

| ID | Content | Source |
|---|---|---|
| Table 1 | Main comparison full-21 | `tables/TABLE1_*.csv` |
| Table 2 | Paired contrasts | `tables/TABLE2_*.csv` |
| Table 3 | E1–UNIFORM pairing | `tables/TABLE3_*.csv` |
| Table 4 | Micro vs macro | `tables/TABLE4_*.csv` |
| Fig. A | Micro vs macro bars | `v2_validation_audit/figures/micro_vs_macro_f1.*` |
| Fig. B | Size–F1 scatter (extra-five highlighted) | `e1_f1_vs_hist_size.*` |
| Fig. C | Seed variability per fire | `e1_per_fire_seed_variability.*` |
| Fig. D | Qualitative multi-panel cases | `figures/qualitative/` |
| Fig. E | Temporal weights E1 vs UNIFORM | `TABLE_S4` (plot for paper) |
| Table S1–S5 | Empty-target, subsets, extra-five, weights, zero-window | `tables/` |
