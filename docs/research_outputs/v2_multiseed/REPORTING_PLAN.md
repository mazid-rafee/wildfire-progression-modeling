# Multi-seed reporting plan (publication-oriented)

## Benchmark label (required in all write-ups)

> **Exploratory reused test benchmark.** The matched `test24` / `test18` / `test16` populations were used during V2 protocol development and single-seed exploration. They are **not** an untouched final holdout. If an independent event population can be established later, that should be the **final** evaluation; until then, report test metrics as exploratory with this label.

## Locked primary metrics (per seed × arm)

Computed with the frozen selected threshold (val-only 0.01–0.99):

| Metric | Aggregation |
|---|---|
| Micro F1, IoU, precision, recall | Pooled TP/FP/FN over all windows/pixels in the eval set |
| Per-fire F1 / IoU | From that fire’s TP/FP/FN |
| Macro F1 / Macro IoU | Unweighted mean over fires |
| Paired per-fire Δ | Same fire_id across arms (e.g. E1−UNIFORM, E1−ENV0, E1−TS2) |

Also always report **val F1 @0.5** (ckpt-selection metric) and **val F1 @selected thr**.

## Uncertainty: fire-event-level resampling

**Do not** treat pixels or overlapping windows as independent for CIs.

### Recommended procedure
1. Let \(\mathcal{F}\) be the set of test fire IDs (e.g. 21 fires on test24).
2. For each seed×arm, retain per-fire TP/FP/FN (already in `evaluations/*.json`).
3. **Cluster bootstrap** over fires (\(B=2000\) or \(B=5000\)):
   - Resample fires with replacement (same size as \(|\mathcal{F}|\)).
   - For each draw, recompute **micro** metrics from summed TP/FP/FN of sampled fires.
   - Optionally recompute **macro** F1 as mean of that draw’s per-fire F1s.
4. Report percentile intervals (2.5%, 97.5%) and seed-mean ± interval.
5. For multi-seed summary: first average per-fire or micro metrics across seeds, then bootstrap fires **within each seed** and combine via nested/bootstrap-of-bootstrap or report seed-wise intervals separately.

### Paired arm contrasts
For E1 vs UNIFORM (same seed, shared init): bootstrap fires using **paired** ΔF1 per fire (or paired micro via resampling the same fire multiset for both arms’ TP/FP/FN). This respects the matched design.

### Explicitly avoid
- Pixel-level binomial CIs
- Window-level bootstrap (windows within a fire are dependent)
- Tuning thresholds on bootstrap test draws

## Multi-seed summary tables (planned)

1. Mean ± SD across seeds of test24 micro F1 / IoU / P / R  
2. Mean ± SD of macro F1  
3. Mean paired Δ (E1−UNIFORM) with fire-cluster bootstrap CI  
4. Learned temporal weights for E1/ENV0 (mean ± SD over seeds)  
5. Selected thresholds distribution (val-only; diagnostic)

## Artifacts to write after training (future; not now)
- `outputs/v2_multiseed/results/per_seed_metrics.json`
- `outputs/v2_multiseed/results/fire_bootstrap_ci.json`
- `outputs/v2_multiseed/results/PAIRED_PER_FIRE_MULTISEED.md`
