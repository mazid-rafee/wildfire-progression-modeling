# Frozen baseline index

External absolute paths only. **Do not modify these artifacts from this project.**

## Phase 3A — published-protocol SpaSE FP (seed 42)

- **Meaning:** Official-protocol SpaSE TS=2 reproduction (full TS=2 train population after seed-42 neg sampling → 1280 train / 167 val). Separate from matched Phase 4B TS=2.
- Freeze:  
  `/aul/homes/mmazi007/Desktop/Source Code (Research)/DeepFire-Forecaster/phase3a_spase_fp_reproduction/outputs/PHASE3A_SPASE_FP_BASELINE_FREEZE_20261101.md`
- Run:  
  `/aul/homes/mmazi007/Desktop/Source Code (Research)/DeepFire-Forecaster/phase3a_spase_fp_reproduction/runs/spase_fp_seed42_20261031_191636/`
- Checkpoint: `checkpoints/best_fp_v6.pth` (SHA256 recorded in freeze report)
- Authoritative eval: `evaluations_ba_labelled_corrected_20261031_233842/`

## Phase 4A — matched temporal coverage audit

- **Meaning:** Defines Set C matched TS=2 ∩ TS=6 windows and BA-labelled test intersections used for controlled comparisons.
- Audit dir:  
  `/aul/homes/mmazi007/Desktop/Source Code (Research)/DeepFire-Forecaster/phase4a_temporal_coverage_audit/outputs/phase4a_audit_20261101_064329/`
- Key files: `matched_ts2_ts6_windows.csv`, `temporal_window_manifest.csv`, `split_coverage_summary.json`, `PHASE4A_TEMPORAL_COVERAGE_REPORT.md`

## Phase 4B — matched SpaSE TS=2 vs TS=6 (seed 42)

- **Meaning:** Same SpaSE architecture family trained on matched windows only; primary evidence that longer SpaSE history under SE/mean-pool did not help matched BA metrics.
- Run:  
  `/aul/homes/mmazi007/Desktop/Source Code (Research)/DeepFire-Forecaster/phase4b_matched_ts_experiment/runs/matched_seed42_parallel_preload_20261101_200211/`
- Comparison: `PHASE4B_MATCHED_COMPARISON.md`
- Checkpoints: `ts2/checkpoints/best_matched.pth`, `ts6/checkpoints/best_matched.pth`

## Reporting rules for new methods

1. Always label **published-protocol** vs **matched** populations.
2. Never present Phase 3A’s checkpoint as a matched TS=2 arm.
3. Primary matched metrics: BA-24 / BA-18 / BA-16 micro F1 & IoU + per-fire tables.
4. Single-seed deltas are exploratory until multi-seed confirmation is pre-registered.
