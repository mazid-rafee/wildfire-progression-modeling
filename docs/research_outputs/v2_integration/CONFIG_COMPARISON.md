# Exact Configuration Comparison (V2 P0 Arms)

**Generated (UTC):** 2026-11-03T16:23:57.455645+00:00

| Arm | Model | TS | Protocol | config_sha256 | thr grid | seed | lr | pos_weight |
|-----|-------|----|----------|---------------|----------|------|----|------------|
| V2-P4B-TS2 | mean_pool | 2 | V2.0-corrected-protocol | `f0bf16972c1c…` | 0.01–0.99/0.01 | 42 | 0.0005 | 112.87821151853076 |
| V2-P4B-TS6 | mean_pool | 6 | V2.0-corrected-protocol | `7a914de70405…` | 0.01–0.99/0.01 | 42 | 0.0005 | 112.87821151853076 |
| V2-E1 | recency_fusion | 6 | V2.0-corrected-protocol | `4bc60ecb946c…` | 0.01–0.99/0.01 | 42 | 0.0005 | 112.87821151853076 |
| V2-E1-TS6 | recency_fusion | 6 | V2.0-corrected-protocol | `d4a8558a4ca9…` | 0.01–0.99/0.01 | 42 | 0.0005 | 112.87821151853076 |
| V2-ENV0-TS6 | recency_fusion | 6 | V2.0-corrected-protocol | `12c0ec4b08c9…` | 0.01–0.99/0.01 | 42 | 0.0005 | 112.87821151853076 |
| V2-MEANPOOL-TS6-CTRL | mean_pool | 6 | V2.0-corrected-protocol | `663a2ad27eea…` | 0.01–0.99/0.01 | 42 | 0.0005 | 112.87821151853076 |

Shared train lock OK: **True** (`v2_p0_shared_20261103`)

## V1 vs V2 protocol selection

| Key | Loader | Stats | Geo | Dir-aug | Thr grid |
|-----|--------|-------|-----|---------|----------|
| `v1` / `v1_frozen` | phase4b V1 | fp_stats_matched.npz | independent crop | no remap | 0.05–0.75 |
| `v2` / `v2_corrected` | wtp.data.v2 | fp_stats_matched_v2.npz | FP→VIIRS | remap 11/17/23 | 0.01–0.99 |
| `v2_mask_gfs` | wtp.data.v2 + mask | same V2 stats | same | same | 0.01–0.99 |
