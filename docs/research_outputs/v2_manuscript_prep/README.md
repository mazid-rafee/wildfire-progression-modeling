# V2 Manuscript Preparation Package

**Status:** Evidence-grounded draft package from frozen V2 multi-seed experiments  
**Protocol:** `V2.0-corrected-protocol` with forecast policy `mask_zero`  
**Generated from:** frozen run artifacts + validation audit (no retraining, no threshold retuning)  
**Test population label:** exploratory reused 2021 US BA-valid benchmark (21 fires / 561 windows)

This directory is additive. Existing `outputs/v2_multiseed/` and `outputs/v2_validation_audit/` artifacts are preserved untouched.

---

## Start here

| File | Purpose |
|---|---|
| [`PAPER_PREP_SUMMARY.md`](PAPER_PREP_SUMMARY.md) | Executive summary: structure, contributions, supported claims, figures/tables, blockers |
| [`02_MANUSCRIPT_OUTLINE.md`](02_MANUSCRIPT_OUTLINE.md) | Structured outline with evidence-linked draft paragraphs |
| [`06_SUBMISSION_CHECKLIST.md`](06_SUBMISSION_CHECKLIST.md) | Remaining work before journal/conference submission |

## Verification and method

| File | Purpose |
|---|---|
| [`00_NUMERICAL_VERIFICATION.md`](00_NUMERICAL_VERIFICATION.md) | Independent recompute of all headline numbers |
| [`01_METHOD_MATH.md`](01_METHOD_MATH.md) | SpaSE mean pool, E1 learnable fusion, frozen-uniform control |
| [`03_V2_PROTOCOL_AND_V1_DISTINCTION.md`](03_V2_PROTOCOL_AND_V1_DISTINCTION.md) | Corrected V2 preprocessing vs historical V1 |
| [`04_HOLDOUT_AND_REUSE_DISCLOSURE.md`](04_HOLDOUT_AND_REUSE_DISCLOSURE.md) | Test reuse + no untouched holdout |
| [`05_UNSUPPORTED_CLAIMS.md`](05_UNSUPPORTED_CLAIMS.md) | Claims that lack experimental support |

## Event / qualitative documentation

| File | Purpose |
|---|---|
| [`07_DIFFICULT_EVENTS_AND_ZERO_WINDOWS.md`](07_DIFFICULT_EVENTS_AND_ZERO_WINDOWS.md) | Extra-five BA-sparse events + 3 zero-window official fires |
| [`08_QUALITATIVE_CASE_INDEX.md`](08_QUALITATIVE_CASE_INDEX.md) | Real case studies mapped to figure paths |

## Tables (CSV)

Under [`tables/`](tables/):

- `TABLE1_main_comparison_full21.csv` — primary full-21 mean ± SD + nested fire CIs
- `TABLE2_paired_contrasts.csv` — seed-wise paired Δ including primary E1−UNIFORM paired-4
- `TABLE3_e1_uniform_init_pairing.csv` — four paired seeds + seed 42 unpaired
- `TABLE4_micro_vs_macro.csv` — micro vs unweighted per-fire macro
- `TABLE_S1_empty_target_supplement.csv` — positive-target-only macro + empty-target false alarms
- `TABLE_S2_subset_robustness.csv` — full-21 / Rule-A 18 / manuscript-16
- `TABLE_S3_extra5_difficult.csv` — five difficult BA-sparse events
- `TABLE_S4_learned_temporal_weights.csv` — E1 vs UNIFORM weights
- `TABLE_S5_zero_window_official_fires.csv` — three official fires with 0 windows
- `TABLE_S6_qualitative_case_index.csv` — qualitative figure index

Machine-readable recompute: [`artifacts/verification_recompute.json`](artifacts/verification_recompute.json).

## Upstream frozen sources (do not modify)

- `outputs/v2_multiseed/reports/`
- `outputs/v2_validation_audit/`
- `runs/v2_ms_*` checkpoints referenced by SHA256 in publication tables
