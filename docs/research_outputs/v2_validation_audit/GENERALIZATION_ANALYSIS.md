# Generalization and Uncertainty Analysis

## Seeds and pairing

| Seed | E1↔UNIFORM init pairing |
|---:|---|
| 7, 17, 29, 101 | **Paired** |
| 42 | **Unpaired** |

## Mean ± sample SD — full-21 test

        test24_f1           test24_iou           test24_macro_f1            test24_p            test24_r          
             mean       std       mean       std            mean       std      mean       std      mean       std
arm                                                                                                               
E1       0.611766  0.023448   0.441007  0.024292        0.404754  0.006883  0.540661  0.059411  0.714121  0.048604
ENV0     0.540540  0.018879   0.370552  0.017663        0.353486  0.017324  0.544029  0.040998  0.540820  0.040558
TS2      0.404069  0.025058   0.253433  0.019583        0.269263  0.021806  0.296529  0.019013  0.636124  0.055926
UNIFORM  0.235434  0.050130   0.134149  0.031959        0.200648  0.020679  0.287924  0.027577  0.208935  0.070340

## Paired micro Δ (E1 − UNIFORM)

- All 5 seeds: **0.3763**
- Paired excl. 42: **0.3857**

## Nested fire-cluster bootstrap

Primary unit: **21 fire events**. Nested micro resamples fires, recomputes each seed’s micro from TP/FP/FN, averages seeds (B=2000).

Reproduction: `artifacts/BOOTSTRAP_REPRODUCTION.json`  
vs published: `artifacts/BOOTSTRAP_VS_PUBLISHED.json` (points match; CI endpoints differ by RNG).

| Contrast | Published point | Published 95% CI |
|---|---:|---|
| E1−TS2 | 0.2077 | [0.16369794591985215, 0.24107036913116214] |
| E1−ENV0 | 0.0712 | [0.006971583836541385, 0.11581158306103162] |
| E1−UNIFORM paired | 0.3857 | [0.2540271129864398, 0.46686095128032046] |

## Robustness (descriptive)

1. Micro vs macro aggregation — both reported.
2. Full-21 / Rule-A 18 / manuscript-16 — `SUBSET_ROBUSTNESS.csv`.
3. All-5 vs paired-4 for E1−UNIFORM — paired-4 is the causal temporal-fusion contrast.

## Limitations

- Test set is an **exploratory reused benchmark**.
- Only 21 fire clusters; CIs are wide.
- Windows from the same fire are dependent.
- Seed variation ≠ fire-event uncertainty.
