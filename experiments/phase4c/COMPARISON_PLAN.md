# Phase 4C comparison plan vs frozen matched Phase 4B

**Status:** plan only — no GPU training/evaluation launched from this document.

## Locked baselines (immutable)

| Arm | Meaning | Artifact |
|-----|---------|----------|
| Phase 4B matched TS=2 | SpaSE mean-pool, matched pop, seed 42 | `.../phase4b/.../ts2/checkpoints/best_matched.pth` |
| Phase 4B matched TS=6 | SpaSE mean-pool, same keys, seed 42 | `.../phase4b/.../ts6/checkpoints/best_matched.pth` |
| Phase 3A published TS=2 | Separate population (not matched) | Reference only; do **not** treat as matched TS=2 |

Frozen Phase 4B headline metrics (seed 42):

| Metric | TS=2 | TS=6 |
|--------|-----:|-----:|
| Full-24 micro F1 | 0.3919 | 0.2239 |
| Rule-A 18 F1 | 0.3944 | 0.2259 |
| Manuscript 16 F1 | 0.4094 | 0.2298 |
| Selected thr (val-only) | 0.75 | 0.75 |

## Proposed Phase 4C primary arm

| Field | Value |
|-------|--------|
| Model | `RecencyFusionUNet3D` (registry: `recency_fusion`) |
| Change vs SpaSE | Replace final `mean(dim=2)` with learnable + recency prior weights |
| Spatial encoder/decoder | Unchanged SpaSE design |
| Population | **Matched** Phase 4A/4B keys only (train 983 / val 135 / test BA 561·553·531) |
| TS | **6** (stress-test longer history with ordered fusion) |
| Seed (first run) | 42 |
| Eval | BA-labelled micro F1/IoU on BA-24 / Rule-A 18 / manuscript 16; thr from **validation only** |
| FP stats / pos_weight | Reuse Phase 4B shared `fp_stats_matched.npz` and recompute pos_weight on matched train labels (same procedure) |

Optional control arms (after primary succeeds smoke + one GPU run):

1. `recency_fusion` TS=2 — isolates fusion vs history length.
2. Mean-pool SpaSE TS=6 reload of frozen Phase 4B — sanity that eval harness matches published numbers.
3. Ablate `recency_scale` fixed to 0 (learnable logits only) vs fixed uniform (pure SpaSE).

## Success criteria (pre-registered, single-seed exploratory)

Primary comparison is against **matched Phase 4B TS=6**, secondarily against **matched Phase 4B TS=2**.

| Gate | Criterion |
|------|-----------|
| Harness parity | Re-eval frozen TS=6 ckpt with this project’s eval → within 1e-4 micro F1 of Phase 4B tables |
| Non-collapse | Val F1@0.5 during training exceeds Phase 4B TS=6 best val F1 (0.3268) **or** selected-thr BA-24 F1 exceeds 0.2239 |
| Competitive | BA-24 micro F1 ≥ matched TS=2 (0.3919) is a stretch goal; ≥ midway (0.308) is encouraging |
| Weight sanity | Learned fusion weights not uniform; newer frames receive higher mass unless data says otherwise |
| No leakage | Thresholds from val only; test keys identical to Phase 4B JSONs |

Single-seed deltas remain exploratory until multi-seed confirmation is pre-registered.

## Metrics to report (same tables as Phase 4B)

1. Best val F1@0.5 and selected threshold  
2. Full-24 / Rule-A 18 / manuscript 16 micro Precision, Recall, F1, IoU  
3. Paired per-event ΔF1 on manuscript 16 vs Phase 4B TS=2 and TS=6  
4. Parameter count and wall-clock  
5. Learned temporal weights (mean over checkpoints / final)

## Launch checklist (future — not now)

- [ ] CPU diagnostics + smoke tests green (this phase)
- [ ] Explicit train approval (`train.enabled` + `--i-understand-full-train`)
- [ ] One config YAML under `configs/experiments/phase4c/`
- [ ] Run dir only under this project’s `runs/`
- [ ] No mutation of DeepFire / TS-SatFire / checkpoints / manifests / DATA_ROOT

## Out of scope for first GPU run

- Published-protocol Phase 3A population retraining  
- Dataset regeneration or duplication  
- Changing BA label definition or test exclusions  
- Claiming statistical superiority from one seed  
