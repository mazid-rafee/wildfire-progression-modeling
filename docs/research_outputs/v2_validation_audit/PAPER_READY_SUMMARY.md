# Paper-Ready Summary — V2 Multi-Seed Validation

**Protocol:** V2.0-corrected (matched BA windows; GFS `mask_zero`)  
**Design:** 4 arms × 5 seeds = 20 cells; thresholds validation-only  
**Test population:** exploratory reused 2021 US benchmark (21 BA-valid fires / 561 windows)

---

## Headline results (full-21, mean ± sample SD)

| Arm | Micro F1 | IoU | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| **E1 (RecencyFusion)** | **0.612 ± 0.023** | 0.441 ± 0.024 | 0.405 ± 0.007 | 0.541 ± 0.059 | 0.714 ± 0.049 |
| ENV-0 | 0.541 ± 0.019 | 0.371 ± 0.018 | 0.354 ± 0.017 | 0.544 ± 0.041 | 0.541 ± 0.041 |
| SpaSE TS=2 | 0.404 ± 0.025 | 0.253 ± 0.020 | 0.269 ± 0.022 | 0.297 ± 0.019 | 0.636 ± 0.056 |
| Frozen-uniform | 0.235 ± 0.050 | 0.134 ± 0.032 | 0.201 ± 0.021 | 0.288 ± 0.028 | 0.209 ± 0.070 |

**Paired causal contrast (learnable vs frozen-uniform fusion):** seeds **7/17/29/101** only — mean Δ micro F1 **+0.386**, nested fire-cluster 95% CI **[0.254, 0.467]**. Seed 42 is unpaired; report separately (+0.339).

---

## Micro vs macro (must state in text)

E1 micro ≈ **0.61** while unweighted per-fire macro ≈ **0.41** because micro is pixel-mass-weighted toward large fires where E1 does well, whereas macro equal-weights all 21 fires including five BA-sparse / FP-dominated events (mean F1 ≈ 0.024 on seed 42). Both metrics are correctly computed; prefer reporting **both**, plus manuscript-16 / Rule-A 18 as secondary.

Suggested wording:  
*“Pooled (micro) F1 emphasizes large progression events; the unweighted per-fire mean is lower because five BA-sparse test fires contribute near-zero F1. We retain the full BA-valid population for primary reporting and present manuscript-16 as a secondary subgroup.”*

---

## Figure captions (draft)

**Fig. Size–performance.** Per-fire E1 F1 (five-seed mean) versus mean historical cumulative BA in the 256×256 crop (pixel counts). Red markers: five fires excluded from manuscript-16. *Descriptive association only.*

**Fig. Micro vs macro.** Mean ± SD micro and macro F1 across seeds for four arms.

**Fig. Seed variability.** Per-fire E1 F1 for five seeds; red means mark manuscript-excluded fires.

**Fig. Case studies (multi-panel).** For each selected `target_key`: VIIRS last history, previous BA, GT progression, E1 probability, E1 binary, frozen-uniform, SpaSE TS=2, ENV-0, TP/FP/FN overlay. Image-coordinate crops; thresholds = validation-selected values from frozen runs. Manifest: `artifacts/QUALITATIVE_FIGURE_MANIFEST.csv`.

---

## Qualitative findings (careful)

- **Successes:** On large active fronts (e.g. MT45790, MT45683, WA48568 high-F1 windows), E1 recovers next-step progression with F1 ≳ 0.97 at the selected window; frozen-uniform collapses.
- **ENV-0 wins:** Some mid-mass windows (e.g. ID44532 L10) where ENV-0 F1 ≫ E1 — environmental channels are not uniformly helpful.
- **Failures:** Extra-five / low-BA events show empty targets, tiny GT (2 px), or FP fill — label/eligibility issues dominate model ranking.

---

## Fire-size / progression (descriptive)

Larger historical and newly burned pixel mass associates with higher per-fire E1 F1 (r≈0.67–0.73). Tertile means: small 0.16 / medium 0.43 / large 0.63 (n=7 each). Do **not** claim causality or retune on these splits.

---

## Untouched holdout

**Not feasible** from the 13 Phase-1 missing train IDs or the 3 non-contributing 2021 test folders. Recommend labeling current results as an exploratory reused benchmark; any future holdout requires a locked external manifest before inference.

---

## Limitations (paper checklist)

1. Reused 2021 test events (not blind).
2. Only 21 fire clusters → wide fire-bootstrap CIs.
3. Windows within a fire are dependent.
4. Seed-42 E1 vs UNIFORM unpaired.
5. Macro sensitive to BA-sparse fires; F1=0 convention when TP=FP=FN=0.
6. Maps are image-coordinate crops, not cartographic products.
7. Pixel counts only (degree grid); no km² claims.

---

## Prioritized recommendations for paper prep

1. **Primary table:** full-21 micro F1/IoU/P/R mean±SD + nested fire-bootstrap CI; include macro alongside with one-sentence weighting explanation.
2. **Causal fusion claim:** restrict to paired seeds 7/17/29/101 for E1−UNIFORM.
3. **Secondary tables:** Rule-A 18 and manuscript-16; never replace full-21 without disclosure.
4. **Figure set:** micro–macro bar; size–F1 scatter with extra-five highlighted; 6–8 qualitative case panels spanning high / median / low / ENV0-win / uniform-gap / extra-five.
5. **Do not** drop difficult fires for headlines or retune thresholds on test.
6. **Holdout:** defer; state reused-benchmark limitation until an approved external manifest exists.
7. **SpaSE TS=6:** keep archival only (prior audit: real dilution limitation).

---

## Paths

All deliverables under:

`/aul/homes/mmazi007/Desktop/Source Code (Research)/Wildfire-Temporal-Prediction/outputs/v2_validation_audit/`
