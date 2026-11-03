# Wildfire Progression Modeling

Research code for **next-day wildfire progression** prediction from satellite time series.

This repository develops temporal fusion variants for satellite wildfire progression, evaluates them under a fixed matched BA-labelled protocol, and keeps dataset / baseline artifacts outside the tree (read-only references only).

**Current focus:** `RecencyFusionUNet3D` — learnable, recency-aware temporal weighting instead of equal mean-pooling over history.

---

## Why this exists

A common baseline collapses the time axis with a uniform average over past days. On a **matched** train/eval population (same windows eligible for both short and long history), simply using more past days under that mean-pool design does **not** improve burned-area progression metrics.

This project asks whether a **better temporal operator**—on the same spatial backbone and the same matched data—can use multi-day context more effectively.

---

## Model

| Name | Module | Idea |
|------|--------|------|
| **RecencyFusionUNet3D** | `src/wtp/models/recency_fusion.py` | Shared spatial encoder–decoder + `RecencyAwareTemporalFusion` (softmax over learnable logits with an optional recency prior) |
| Mean-pool baseline | `src/wtp/models/spase_backbone.py` | Same backbone; equal `mean` over time |

Registry ids: `recency_fusion`, `mean_pool`.

---

## Evaluation protocol (locked)

- **Task:** next-day burned-area **progression** (BA increment), not cumulative BA
- **Inputs:** day stack ending at \(L-1\) (no future day \(L\) in features)
- **Test sets (matched BA-labelled):** full-24 / rule-A 18 / manuscript 16 windows — **561 / 553 / 531**
- **Metrics:** micro F1, IoU, precision, recall + per-fire tables
- **Thresholds:** chosen on **validation only** (never tuned on test)
- **Reporting:** single-seed results are exploratory until multi-seed confirmation

Baseline comparisons use frozen mean-pool SpaSE checkpoints trained on the same matched population with history length 2 and 6. Paths and freeze notes live in `docs/BASELINES.md` (machine-local; not vendored here).

---

## Repository layout

```
├── configs/           # paths + experiment YAMLs
├── docs/              # attribution, baselines index, reproducibility
├── src/wtp/           # installable package
│   ├── models/        # mean-pool backbone + recency fusion
│   ├── data/          # matched-population bridges (read-only data access)
│   ├── train/         # training loop + launch gate
│   ├── eval/          # BA micro-metric harness
│   └── utils/         # path / external adapters
├── scripts/           # diagnostics, train/eval, validation audit CLIs
├── experiments/       # experiment notes
├── tests/
├── runs/              # local run dirs (gitignored contents)
├── outputs/           # local reports (gitignored contents)
├── logs/              # local logs (gitignored)
└── checkpoints/       # local weights (gitignored)
```

Large artifacts (`.pth`, run histories, eval dumps) are **gitignored**. Keep them on the research machine; commit code and docs only.

---

## Setup

```bash
conda activate ts-satfire   # or another env with torch + deps
cd /path/to/this/repo
pip install -e .
```

Point `configs/paths.yaml` at your local **TS-SatFire** `DATA_ROOT` and any baseline checkpoint directories. Do not copy the GeoTIFF dataset into this repo.

```bash
python scripts/verify_external_refs.py
```

---

## Common entrypoints

```bash
# Architecture / temporal-aggregation sanity (CPU)
python scripts/phase4c_smoke_test_recency.py
python scripts/phase4c_cpu_diagnostics.py

# Matched train + BA eval for RecencyFusionUNet3D (GPU; gated)
python scripts/phase4c_e1_train_eval.py --run-dir runs/<name> --i-understand-full-train

# Read-only result audit (re-eval checkpoint; writes under outputs/)
python scripts/phase4c_e1_validation_audit.py --out-dir outputs/<audit_name>
```

Training / GPU jobs require an explicit approval flag (`--i-understand-full-train` or `--skip-train`). See `docs/REPRODUCIBILITY.md`.

---

## Data & non-interference

| Resource | Policy |
|----------|--------|
| TS-SatFire GeoTIFFs | Read-only via absolute `DATA_ROOT` |
| Upstream DeepFire baseline trees | Reference only; **do not modify** |
| This repo | Own models, scripts, configs, and local `runs/` / `outputs/` |

Licenses and citations: `docs/ATTRIBUTION.md`.

---

## Status

- RecencyFusionUNet3D implemented and trained on the matched TS=6 population (seed 42)
- Independent checkpoint re-evaluation reproduces the reported matched full-24 metrics
- Next: controlled ablations (last-frame-only, fixed recency, multi-seed) before further architecture work
