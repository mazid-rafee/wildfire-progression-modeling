# Threshold-grid erratum (SpaSE TS=6 seed-42)

## Reported inconsistency
Audit prose stated SpaSE TS=6 selected threshold **0.95** was a **grid ceiling** of a documented max **0.99**.

## Resolution (frozen files inspected)
File: `runs/v2_matched_spase_ts6_seed42_20261103_162551/selected_threshold.json`

| Field | Value |
|---|---|
| `n_thresholds` | **99** (0.01 … 0.99 step 0.01) |
| `selected_threshold` | **0.95** |
| `selected_is_grid_ceiling` | **`false`** |
| Val F1 @0.95 | 0.390665 (maximum in sweep) |
| Val F1 @0.96 … @0.99 | 0.3902 → 0.3489 (**declining**) |

**Conclusion:** There is no protocol mismatch. The grid correctly extends to **0.99**. Threshold **0.95** is a genuine interior optimum on validation. The audit report’s “grid ceiling” wording was incorrect; the JSON flag was already `false`.

## Action for multi-seed
- Keep locked thr grid `{min: 0.01, max: 0.99, step: 0.01}`, validation-only.
- Do **not** retune seed-42 headlines.
- Do **not** change the grid based on test performance.
- Treat audit ceiling language as erratum only (`outputs/v2_audit/AUDIT_REPORT.md` footnote †).

## Frozen seed-42 results
Unchanged. Checkpoints and metrics preserved.
