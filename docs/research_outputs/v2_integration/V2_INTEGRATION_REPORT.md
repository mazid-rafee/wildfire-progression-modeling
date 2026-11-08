# V2 Training Integration Report

**Generated (UTC):** 2026-11-03T09:33:19.668047+00:00
**Mode:** integration + CPU dry-run only — **no training, no GPU jobs**

## Verdict

- Essential integrity: **True**
- Train entrypoint wired: **True**
- Dry-runs pass: **True**
- Training-ready (protocol+harness): **True**
- Full launch still requires: explicit approval + forecast_policy choice
- ENV-0 left running: **True**

## What was integrated

| Component | Path |
|---|---|
| Protocol configs | `configs/protocol/v1_frozen.yaml`, `v2_corrected.yaml`, `v2_corrected_mask_gfs_forecast.yaml` |
| P0 arm configs | `configs/experiments/v2/*.yaml` + `_shared_train.yaml` |
| Protocol API | `src/wtp/protocol/` |
| Explicit loader | `src/wtp/data/protocol_loader.py` (V1 and V2 selectable) |
| Entrypoint | `scripts/v2_train_eval.py --dry-run` |
| Integrity gate | `scripts/v2_integrity_gate.py` |

## Exact configuration comparison

| Arm | Model | TS | Protocol | config_sha256 | thr | seed | lr | pos_w |
|-----|-------|----|----------|---------------|-----|------|----|-------|
| V2-P4B-TS2 | mean_pool | 2 | V2.0-corrected-protocol | `f0bf16972c1c3fc2…` | 0.01-0.99/0.01 | 42 | 0.0005 | 112.87821151853076 |
| V2-P4B-TS6 | mean_pool | 6 | V2.0-corrected-protocol | `7a914de704051a18…` | 0.01-0.99/0.01 | 42 | 0.0005 | 112.87821151853076 |
| V2-E1 | recency_fusion | 6 | V2.0-corrected-protocol | `4bc60ecb946c8607…` | 0.01-0.99/0.01 | 42 | 0.0005 | 112.87821151853076 |
| V2-MEANPOOL-TS6-CTRL | mean_pool | 6 | V2.0-corrected-protocol | `663a2ad27eead229…` | 0.01-0.99/0.01 | 42 | 0.0005 | 112.87821151853076 |

Shared train locks OK: **True** (`v2_p0_shared_20261103`)

## Forecast-time availability (blocker for observational claims)

- Established: **False**
- Affected model channels: **22, 23, 24, 25** (GFS forecast wind speed/dir, temp, humidity)
- Default arm policy: `include_as_forecast` (declared forecast, not past obs)
- Conservative alternative: protocol `v2_mask_gfs` / `--forecast-policy mask_zero` (zeros ch 22–25, keeps 27-ch model)

## Dry-run results (CPU, no optimizer step)

| Run dir | protocol | forecast_policy | smoke_ok | gpu_used |
|---|---|---|---|---|
| `V2-P4B-TS2_v2` | v2 | include_as_forecast | True | False |
| `V2-P4B-TS6_v2` | v2 | include_as_forecast | True | False |
| `V2-MEANPOOL-TS6-CTRL_v2` | v2 | include_as_forecast | True | False |
| `V2-E1_v2_forecast_incl` | v2 | include_as_forecast | True | False |
| `V2-E1_v2_mask_zero` | v2 | mask_zero | True | False |
| `V2-E1_v1` | v1 | include_as_forecast | True | False |

## Tests

- 39 passed, 1 deselected (data_backed run separately)
- data-backed integrity: 1 passed
- Artifacts: `outputs/v2_integration/pytest_fast.txt`, `integrity_checks.json`

## Suggested safe P0 launch order

1. **V2-P4B-TS2** — Smallest TS; establishes V2 SpaSE baseline
2. **V2-P4B-TS6** — Matched TS=6 mean-pool under V2
3. **V2-MEANPOOL-TS6-CTRL** — E1-harness mean-pool control before RecencyFusion
4. **V2-E1** — RecencyFusion after mean-pool controls exist

```bash
# AFTER explicit approval — example only, do not run yet
# export CUDA_VISIBLE_DEVICES=<free_gpu_not_env0>
# python -u scripts/v2_train_eval.py --arm V2-P4B-TS2 --forecast-policy include_as_forecast \
#   --no-dry-run --i-understand-full-train   # currently still gated until train path enabled
```

## Remaining blockers / decisions

1. **Forecast policy choice** before launch (include_as_forecast vs mask_zero).
2. **Explicit approval** to enable full-train path (currently dry-run / gated).
3. Non-blocking: CAL-GAPS sensitivity, multi-seed later.
