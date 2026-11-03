# Phase 4C — proposed temporal progression models

Planning / registry space for original architectures and ablations after Phase 4B.

## Motivation (from frozen baselines)

- Phase 3A: SpaSE FP TS=2 published-protocol seed-42 baseline is frozen.
- Phase 4A: matched TS=2∩TS=6 window populations defined.
- Phase 4B: matched SpaSE TS=6 underperformed matched SpaSE TS=2 on BA micro metrics (single seed) — longer raw history through SE/mean-pool is not sufficient.

## Current status (Phase 4C CPU diagnostics)

Started: CPU-only diagnostics + minimal recency-fusion SpaSE variant. **No GPU training.**

| Artifact | Path |
|----------|------|
| Temporal aggregation audit | `docs/SPASE_TEMPORAL_AGGREGATION.md` |
| Comparison plan vs Phase 4B | `experiments/phase4c/COMPARISON_PLAN.md` |
| Model | `src/wtp/models/recency_fusion.py` (`recency_fusion`) |
| Config stub (not launched) | `configs/experiments/phase4c/recency_fusion_matched_ts6_seed42.yaml` |
| Diagnostic outputs | `outputs/phase4c_diagnostics/` |

### Scripts

```bash
conda activate ts-satfire
cd Wildfire-Temporal-Prediction
python scripts/verify_external_refs.py
python scripts/phase4c_cpu_diagnostics.py
python scripts/phase4c_smoke_test_recency.py
pytest tests/test_recency_fusion.py -q
```

## Intended directions

1. ~~Document SpaSE equal temporal mean and permutation invariance~~ (done in docs + diagnostics)
2. Learnable recency-aware fusion on matched TS=6 (implemented; GPU train gated)
3. Explicit change / residual input features under matched eval
4. Stronger temporal operators (factorized conv, bottleneck attention) as follow-on ablations

## Rules

- Implement code under `src/wtp/`.
- Store configs under `configs/experiments/phase4c/`.
- Write run artifacts only under this project’s `runs/`, `checkpoints/`, `logs/`, `outputs/`.
- Do not alter DeepFire-Forecaster or TS-SatFire trees.
- Do not launch GPU training without explicit approval (`train.enabled` + gate flag).
