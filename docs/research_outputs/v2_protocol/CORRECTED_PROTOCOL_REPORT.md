# Corrected Protocol Report (V2 Preparation)

**Generated (UTC):** 2026-11-03T08:15:37.778264+00:00
**Protocol:** `V2.0-corrected-protocol`
**Retraining:** not launched

## ENV-0 / queue safety

- ENV-0 left running: **True**
- Queued ablations cancelled: **False** (found=False)
- Limitation: No additional environmental ablations are queued. Only ENV-0 is active. Nothing to prevent from starting without affecting ENV-0.

## Issues corrected in code (V2 package)

| ID | Correction | Module |
|----|------------|--------|
| GEO-MISREG | FP→VIIRS resample + shared crop | `wtp.data.v2.geo_align` / `windows` |
| AUG-DIRECTIONAL | Unit-vector angle remap under flip/rot90 | `wtp.data.v2.directional_aug` |
| STATS-SUBSAMPLE | Matched-train-only geo-aligned FP stats | `wtp.data.v2.normalize` |
| TEMP-FORECAST (docs) | Issue/valid-time + availability flags | `wtp.data.v2.timestamps` |
| THR-EXT-LIMITATION | Val re-inference sweep 0.01–0.99 | `scripts/v2_extended_threshold_sweep.py` |

## Issues remaining unresolved / deferred

| ID | Status | Notes |
|----|--------|-------|
| FP-OMIT-BAND | methodological | Still skipped; comment corrected in channel_map; not re-added |
| CAL-GAPS | methodological | Documented; optional gap==1 filter not applied yet |
| PROT-DROP-LAST | methodological | Unchanged default |
| COMPARE-POPULATIONS | methodological | Phase3A ≠ matched; labeling only |
| SPLIT-GEO-NEAR-SAME | methodological | Within-split residual dependence |
| GFS dir range | partial | Remap applied via unit vectors; signed [-90,90] convention still dataset-odd |
| Land-cover z-score | unchanged | Still continuous z-score of class ids (not one-hot) |

## Population preservation

- Verify OK: **True**
- Counts: `{"train": 983, "val": 135, "test24": 561, "test18": 553, "test16": 531}` (expected `{"train": 983, "val": 135, "test24": 561, "test18": 553, "test16": 531}`)
- Geo-mismatch train fires retained: ['20778186', '20899673', '20899813', '21038333']

## Geospatial

- Confirmed mismatches probed: 4
- V2 strategy: reproject FirePred→VIIRS (bilinear continuous / nearest landcover) then shared crop

## Temporal / availability

- Window contract probes OK: **True**
- Forecast model channels flagged: [22, 23, 24, 25]
- Policy: Keep GFS bands as declared forecast features for V2 training; do not describe them as past observations. Omitted precip remains out of model.

## Normalization

- V2 stats meta: `{"protocol": "V2.0-matched-train-only", "ts_for_day_union": 6, "unique_days": true, "n_train_rows_used": 983, "n_train_rows_total": 983, "n_day_crops": 1523, "n_missing_fp_days": 0, "n_train_fires": 82, "train_fire_ids": ["20778186", "20899673", "20899813", "21038333", "21748801", "21751305", "21751309", "21804582", "21804884", "21804985", "21889717", "21889750", "21889763", "21889779", "21889943", "21889992", "21889994", "21890003", "21890009", "21890013", "21890024", "21890056", "21890072", "2...`
- Δ vs V1: {"l2_mean": 246.87749947468686, "l2_std": 190.59818381640326, "max_abs_mean": 172.45166015625, "max_abs_std": 172.832275390625, "per_slot_mean_delta": [172.45166015625, -102.46337890625, -0.05245161056518555, -5.9920806884765625, -2.465972900390625, -4.104705810546875, 0.7905807495117188, -0.0005827285349369049, 1.8602333068847656, 6.8994598388671875, 142.7880859375, 0.3061954975128174, -0.6915760040283203, -9.756706237792969, -0.501030683517456, 10.524560928344727, -2.1059303283691406, -0.0006206966936588287], "per_slot_std_delta": [-76.703857421875, -172.832275390625, 0.06782150268554688, 7.942527770996094, 5.915290832519531, 6.046924591064453, 3.0787734985351562, 0.00023604813031852245, 0.30097103118896484, -1.9011764526367188, 1.96136474609375, -0.1857466697692871, -0.012912750244140625, -20.51039695739746, -0.334861159324646, 0.14923858642578125, 0.8184709548950195, 9.285192936658859e-05], "v1_stats_path": "/aul/homes/mmazi007/Desktop/Source Code (Research)/DeepFire-Forecaster/phase4b_matched_ts_experiment/runs/matched_seed42_parallel_preload_20261101_200211/shared_populations/fp_stats_matched.npz", "v2_stats_path": "/a/bear.cs.fiu.edu./disk/bear-b/users/mmazi007/Desktop/Source Code (Research)/Wildfire-Temporal-Prediction/outputs/v2_protocol/fp_stats_matched_v2.npz"}

## Extended validation threshold sweep (eval-only)

Historical headline metrics **unchanged**. Re-inference on frozen checkpoints, val-only grid 0.01–0.99:

| Arm | Historical thr / val F1 | Extended best thr / F1 | F1@0.75 (re-infer) |
|-----|-------------------------|------------------------|--------------------|
| E1 | 0.75 / 0.64615 | **0.92** / 0.65375 | 0.64615 |
| Phase4B TS6 | 0.75 / 0.35139 | **0.89** / 0.36022 | 0.35134 |
| Phase4B TS2 | 0.75 / 0.45820 | **0.78** / 0.45887 | 0.45821 |

Source: `outputs/v2_protocol/threshold_extended/summary.json`.

## Ready for controlled retraining?

- **Protocol-ready: Yes** — V2 preprocessor, matched population verify, train-only stats, regression tests, and retraining matrix are in place; no train jobs launched.
- **Train harness: not yet wired** — need a `v2_train_eval` entrypoint that calls `wtp.data.v2` before launching P0 arms.
- Blocking for *protocol*: none.
- Non-blocking follow-ups: wire train entrypoint; optional gap==1 sensitivity; optional precip-band ablation.

## Artifacts

- `outputs/v2_protocol/` (this report, comparison JSON, stats, probes, threshold_extended/)
- `docs/V1_VS_V2_PROTOCOL.md`
- `tests/v2/` regression suite (24 passed)
- `src/wtp/data/v2/` corrected preprocessor (does not overwrite V1)
