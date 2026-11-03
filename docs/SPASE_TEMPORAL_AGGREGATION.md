# Temporal aggregation audit (mean-pool baseline)

Read-only analysis of the frozen matched mean-pool U-Net3D checkpoint class (`SEUNet3DPred`) used as the external baseline.  
Upstream file: `DeepFire-Forecaster/phase3a_spase_fp_reproduction/phase3a_fp_model.py` (MIT).

## Tensor layout

| Stage | Shape |
|-------|--------|
| Window assembly | `(C=27, T, H=256, W=256)` then batched → `(B, 27, T, H, W)` |
| Channels per frame | 8 sat (6 day + 2 night) + 18 FirePred + 1 BA = 27 |
| TS=2 inputs | indices `L-2, L-1` |
| TS=6 inputs | indices `L-6 … L-1` (last two equal TS=2) |

## Where time is (and is not) mixed

| Module | Temporal behavior |
|--------|-------------------|
| `Conv3d` / `ConvTranspose3d` kernels `(1, ·, ·)` | **No** temporal mixing — applied independently per frame |
| `MaxPool3d((1, 2, 2))` | Spatial only |
| `SEBlock3D` → `AdaptiveAvgPool3d(1)` | Equal pool over **T×H×W** for channel squeeze (permutation-invariant) |
| `ASPP3D` dilated branches | Spatial dilations; temporal kernel size 1 |
| `ASPP3D.gp` → `AdaptiveAvgPool3d((None, 1, 1))` | Pools H×W, **keeps T** |
| **`x.mean(dim=2)` after decoder** | **Equal temporal average** — collapses `(B,C,T,H,W)` → `(B,C,H,W)` before 2D head |

### Primary equal temporal averaging site

```python
# phase3a_fp_model.py — SEUNet3DPred.forward (end of decoder)
x = x.mean(dim=2)   # equal weight 1/T for every timestep
return self.head(x)
```

This is the operator that was lengthened from T=2 to T=6 in the matched mean-pool bake-off without changing temporal selectivity. Because earlier layers are time-separable, **order information is discarded** at this mean.

### Secondary equal pooling (not a temporal fusion head)

`SEBlock3D`’s global pool averages all spacetime locations equally to produce channel gates. It does not implement ordered temporal fusion; it is invariant to permuting T.

## Implication for matched mean-pool baselines

Matched SpaSE TS=6 underperformed TS=2. Architecture-level explanation consistent with this audit:

1. Extra history is encoded with the same per-frame spatial filters.
2. SE gating cannot prefer recent frames by position.
3. Final mean treats t−5 and t identically.
4. Therefore longer raw history is not “used as a sequence,” only as a larger equal-weight bag of frames (plus whatever content differences older frames carry).

This project’s minimal intervention: replace **only** `mean(dim=2)` with learnable, recency-aware fusion while freezing the spatial encoder/decoder design.

## Local reimplementation

- Baseline mean-pool: `src/wtp/models/spase_backbone.py` → `MeanPoolUNet3D`
- Recency fusion variant: `src/wtp/models/recency_fusion.py` → `RecencyFusionUNet3D`
