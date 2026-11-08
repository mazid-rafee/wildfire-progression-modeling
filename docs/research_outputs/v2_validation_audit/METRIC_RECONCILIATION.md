# Metric Reconciliation — V2 Multi-Seed

## Definitions (from `src/wtp/eval/ba_eval.py`)

- **Micro F1 / IoU / P / R:** pool TP, FP, FN over all pixels in all windows, then compute the metric once.
- **Per-fire F1:** pool TP/FP/FN over that fire’s windows, then `F1 = 2·TP / (2·TP + FP + FN + 1e-8)`.
- **Macro F1 (`test24_macro_f1`):** **unweighted arithmetic mean of the 21 per-fire F1 values** — not per-window F1, not size-weighted.
- Empty TP=FP=FN=0 yields F1≈0 via the `1e-8` stabilizer.

## Integrity

- All 20 cells: recomputed micro F1 from TP/FP/FN matches stored JSON within **1e-5**.
- All 20 cells: unweighted mean of `per_fire[*].f1` matches `RESULTS_TABLE.test24_macro_f1` within **1e-5**.
- Populations: test24=561/21, test18=553/18, test16=531/16 on every cell.
- Seed-42 reused checkpoints SHA256 match the results table (`artifacts/CHECKPOINT_SHA256_AUDIT.csv`).

## Seed-42 E1 manuscript-16 vs full-21 arithmetic

| Quantity | Value |
|---|---:|
| Mean F1 over manuscript-16 fires | 0.539290 |
| Mean F1 over full-21 fires (macro) | 0.416629 |
| Mean F1 over the five excluded fires | 0.024112 |
| Implied five-fire mean `(21·m21 − 16·m16)/5` | 0.024112 |
| Absolute discrepancy | 3.747e-16 |

The five fires excluded from manuscript-16 average ~0.024 F1 and pull full-21 macro from ~0.54 to ~0.42.

## Why micro ≈ 0.612 while macro ≈ 0.405 (five-seed E1 means)

1. **Micro is pixel-mass weighted.** Large fires with many progression pixels dominate pooled TP/FP/FN; E1 does relatively well there.
2. **Macro equal-weights fires**, including five difficult / near-empty events with near-zero F1.
3. **Not an evaluation bug** — both metrics are correctly computed from the same TP/FP/FN tables.

### Seed-42 E1 extra-five

```
                      fire_id       f1    tp     fp    fn  gt_positive_px  pred_positive_px  n_windows
US_2021_AZ3345510938920210616 0.044554 135.0 5373.0 417.0           552.0            5508.0         10
US_2021_AZ3368910927620210616 0.076007 217.0 4934.0 342.0           559.0            5151.0         12
US_2021_MT4714310953420211004 0.000000   0.0    1.0   0.0             0.0               1.0          1
US_2021_NM3323810847220210520 0.000000   0.0 4808.0   2.0             2.0            4808.0          6
US_2021_NM3344410803520210514 0.000000   0.0    0.0   0.0             0.0               0.0          1
```

- **NM33444:** TP=FP=FN=0 → F1=0 by convention.
- **MT47143:** essentially no GT positives; tiny FP → F1=0.
- **NM33238:** almost no GT positives but large FP → F1=0.
- **AZ×2:** small GT positives with large FP → very low F1.

## Subset robustness

See `artifacts/SUBSET_ROBUSTNESS.csv`. Ordering E1 ≻ ENV0 ≻ TS2 ≻ UNIFORM is preserved on Rule-A 18 and manuscript-16.
