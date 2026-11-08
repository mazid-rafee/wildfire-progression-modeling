# Mathematical Description of Temporal Operators

Sources: `src/wtp/models/spase_backbone.py`, `src/wtp/models/recency_fusion.py`, `tests/v2/test_frozen_uniform_fusion.py`.

All three operators act on decoder features \(X \in \mathbb{R}^{B \times C \times T \times H \times W}\) and produce a temporally collapsed map \(Z \in \mathbb{R}^{B \times C \times H \times W}\) before the shared 2-D head.

---

## 1. SpaSE / MeanPool equal temporal averaging

**Arm:** TS2 (`MeanPoolUNet3D`, \(T=2\)). The archival TS=6 mean-pool arm is not a multi-seed headline comparator.

\[
Z_{b,c,h,w} = \frac{1}{T}\sum_{t=1}^{T} X_{b,c,t,h,w}
\]

Equivalently, `x.mean(dim=2)`. Weights are fixed, non-learnable, and identical for every sample, channel, and spatial location.

Spatial encoder / ASPP / decoder / 2-D head match the RecencyFusion backbone; only the temporal collapse differs for E1-class models.

---

## 2. E1 learnable RecencyAware temporal fusion

**Arm:** E1 (`RecencyFusionUNet3D`, \(T=6\)).

Learnable parameters (content-agnostic, shared across batch / channel / space):

- logits \(\ell \in \mathbb{R}^{T_{\max}}\)
- scalar recency scale \(s \in \mathbb{R}\)
- fixed prior buffer \(p = \mathrm{linspace}(-1, 1, T_{\max})\) (not trained)

For sequence length \(T \le T_{\max}\):

\[
\tilde{\ell}_t = \ell_t + s\, p_t,\qquad
w = \mathrm{softmax}(\tilde{\ell}_{1:T}),\qquad
Z_{b,c,h,w} = \sum_{t=1}^{T} w_t\, X_{b,c,t,h,w}.
\]

At initialization with \(\ell=\mathbf{0}\) and \(s=0\), \(w_t = 1/T\) (mean-pool equivalent start). Training updates \(\ell\) and \(s\) only; the spatial backbone remains architecturally identical to MeanPool.

### Critical distinction: globally learned weights ≠ input-adaptive attention

| Property | E1 RecencyFusion | Input-adaptive temporal attention |
|---|---|---|
| Weight depends on input features? | **No** | Yes (queries/keys from \(X\)) |
| Weight varies across batch samples? | **No** (same \(w\) for all \(b\)) | Typically yes |
| Weight varies across space / channels? | **No** | Often yes |
| Degrees of freedom (\(T=6\)) | \(T+1 = 7\) fusion params | Usually much larger |
| What is learned? | A **global** temporal preference (with optional recency prior) | Content-dependent mixing |

**Do not describe E1 as attention, self-attention, or temporal transformer.** It is a globally parameterized soft temporal mixture with a fixed recency prior.

Frozen E1 weights concentrate on the newest frame (\(w_6 \approx 0.58\text{–}0.64\)); see `tables/TABLE_S4_learned_temporal_weights.csv`.

---

## 3. Frozen-uniform control

**Arm:** UNIFORM (`RecencyFusionUNet3D` + `freeze_temporal_fusion_uniform`).

Procedure (`freeze_temporal_fusion_uniform`):

1. Zero \(\ell\) and \(s\).
2. Disable gradients on fusion parameters.
3. Resulting weights are exactly \(w_t = 1/T\).

With identical backbone weights, forward outputs match `MeanPoolUNet3D` within numerical tolerance (`tests/v2/test_frozen_uniform_fusion.py`). Trainable parameter count equals MeanPool (**24,279,561**).

This isolates **learnable vs frozen-uniform temporal fusion** under the E1 harness, rather than changing the spatial architecture.

---

## 4. Initialization pairing for the primary ablation

For seeds **7, 17, 29, 101**:

1. Train E1; export backbone state dict (exclude `temporal_fusion.*`).
2. Construct UNIFORM; load the exported backbone; freeze fusion to \(1/T\); train under the same harness.

Seed **42** lacks an exported E1 init; UNIFORM used a reconstructed `set_seed(42)` initialization only. Report seed 42 separately; **do not** treat it as an identical-init causal contrast.

---

## 5. Related arm (ENV-0)

ENV-0 uses the same RecencyFusion operator (\(T=6\)) but masks FirePred channels 8–25 (retains VIIRS 0–7 + BA 26). It is a **channel ablation**, not a temporal-operator ablation.
