# Qualitative Case Index (Real Frozen Predictions)

**Figures:** `outputs/v2_validation_audit/figures/qualitative/`  
**Manifest:** `artifacts/QUALITATIVE_FIGURE_MANIFEST.csv` → copied summary `tables/TABLE_S6_qualitative_case_index.csv`  
**Coordinate note:** image-coordinate 256×256 center crops — **not** georeferenced maps.  
**Checkpoints:** seed-42 frozen SHA256s recorded in the manifest; thresholds = validation-selected values from those runs.

Selection rules were prespecified in the validation audit (high / median / low / env0_beats_e1 / e1_beats_uniform / extra5)—not aesthetic-only cherry-picking.

---

## Recommended narrative roles

### A. Successful progression capture (large active fronts)

| Fire / window | E1 | ENV0 | TS2 | UNIFORM | Groups | Why |
|---|---:|---:|---:|---:|---|---|
| `MT45790…` L85 | 0.980 | 0.977 | 0.976 | 0.030 | high_e1, e1_beats_uniform | Tight recovery; UNIFORM collapses |
| `MT45683…` L73 | 0.977 | 0.956 | 0.969 | 0.479 | high_e1 | Strong multi-arm success; U weaker |
| `WA48568…` L29 | 0.974 | 0.001 | 0.188 | 0.074 | high_e1, e1_beats_uniform | E1 succeeds where ENV0/TS2/U fail |

### B. Median / partial success (missed or weak progression)

| Fire / window | E1 | ENV0 | TS2 | UNIFORM | Groups |
|---|---:|---:|---:|---:|---|
| `CA34517…` L10 | 0.179 | 0.178 | 0.122 | 0.034 | median_e1 |
| `ID46638…` L19 | 0.164 | 0.124 | 0.037 | 0.054 | median_e1 |
| `WA48285…` L32 | 0.206 | 0.284 | 0.030 | 0.029 | median_e1 |

### C. ENV-0 advantages (environmental channels not uniformly helpful)

| Fire / window | E1 | ENV0 | Δ (ENV0−E1) | Groups |
|---|---:|---:|---:|---|
| `ID44532…` L10 | 0.348 | 0.742 | +0.394 | env0_beats_e1 |
| `CA35687…` L17 | 0.417 | 0.698 | +0.281 | env0_beats_e1 |
| `CA36047…` L19 | 0.333 | 0.444 | +0.111 | env0_beats_e1 |

(`WA48778…` appears in env0_beats_e1 picks with near-zero F1 for all arms—use only with explicit “both fail” caption.)

### D. E1 versus UNIFORM differences

| Fire / window | E1 | UNIFORM | Groups |
|---|---:|---:|---|
| `MT45790…` L85 | 0.980 | 0.030 | e1_beats_uniform |
| `WA48568…` L29 | 0.974 | 0.074 | e1_beats_uniform |
| `ID45585…` L22 | 0.966 | 0.055 | e1_beats_uniform |

### E. False-positive-heavy / BA-sparse / empty targets

| Fire / window | E1 | Pattern | Groups |
|---|---:|---|---|
| `NM33238…` L19 | 0.000 | Near-empty GT, mass FP | low_e1, extra5 |
| `AZ33455…` L9 | 0.000 | Small GT, FP fill (window-level) | extra5 |
| `AZ33689…` L9 | 0.000 | Small GT, FP fill | extra5 |
| `MT47143…` L6 | 0.000 | Empty GT progression | low_e1, extra5 |
| `NM33444…` L21 | 0.000 | Empty TP=FP=FN | low_e1, extra5 |

---

## Panel contents (each figure)

VIIRS last history · previous BA · GT progression · E1 heatmap · E1 binary · UNIFORM · SpaSE TS2 · ENV-0 · TP/FP/FN overlay.

---

## Caption template

> Image-coordinate 256×256 crop for event `{fire_id}` window `{Lk}` under frozen seed-42 checkpoints and validation-selected thresholds. Not a georeferenced map product. Selection group: `{primary_group}`.
