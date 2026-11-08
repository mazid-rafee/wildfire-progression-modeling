# Publication Tables (Markdown view)

CSVs are authoritative.

## Table 1 — Main comparison (full BA-valid 21 fires / 561 windows)

| method | arm | n_seeds | n_fires | n_windows | micro_f1_mean | micro_f1_std | micro_f1_nested_fire_bootstrap_ci95_lo | micro_f1_nested_fire_bootstrap_ci95_hi | iou_mean | iou_std | macro_f1_mean | macro_f1_std | precision_mean | precision_std | recall_mean | recall_std | population_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RecencyFusion E1 (TS=6, learnable fusion) | E1 | 5 | 21 | 561 | 0.6118 | 0.0234 | 0.5000 | 0.6806 | 0.4410 | 0.0243 | 0.4048 | 0.0069 | 0.5407 | 0.0594 | 0.7141 | 0.0486 | full matched BA-valid test (21 fires / 561 windows); exploratory reused benchmark |
| ENV-0 (VIIRS+BA only, RecencyFusion TS=6) | ENV0 | 5 | 21 | 561 | 0.5405 | 0.0189 | 0.4676 | 0.5870 | 0.3706 | 0.0177 | 0.3535 | 0.0173 | 0.5440 | 0.0410 | 0.5408 | 0.0406 | full matched BA-valid test (21 fires / 561 windows); exploratory reused benchmark |
| SpaSE mean-pool (TS=2) | TS2 | 5 | 21 | 561 | 0.4041 | 0.0251 | 0.3345 | 0.4480 | 0.2534 | 0.0196 | 0.2693 | 0.0218 | 0.2965 | 0.0190 | 0.6361 | 0.0559 | full matched BA-valid test (21 fires / 561 windows); exploratory reused benchmark |
| Frozen-uniform RecencyFusion (TS=6) | UNIFORM | 5 | 21 | 561 | 0.2354 | 0.0501 | 0.1871 | 0.2851 | 0.1341 | 0.0320 | 0.2006 | 0.0207 | 0.2879 | 0.0276 | 0.2089 | 0.0703 | full matched BA-valid test (21 fires / 561 windows); exploratory reused benchmark |

## Table 2 — Paired contrasts (micro F1)

| contrast | seeds | n | mean_delta_micro_f1 | std_delta | nested_fire_bootstrap_ci95_lo | nested_fire_bootstrap_ci95_hi | init_pairing | role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 − TS2 | 7,17,29,42,101 | 5 | 0.2077 | 0.0323 | 0.1637 | 0.2411 | N/A (architecture/TS differs) | secondary contrast |
| E1 − ENV0 | 7,17,29,42,101 | 5 | 0.0712 | 0.0290 | 0.0070 | 0.1158 | N/A (channel mask differs; same RecencyFusion class) | secondary contrast |
| E1 − UNIFORM (all 5) | 7,17,29,42,101 | 5 | 0.3763 | 0.0494 | 0.2525 | 0.4590 | mixed (seed 42 unpaired) | descriptive all-seed |
| E1 − UNIFORM (init-paired) | 7,17,29,101 | 4 | 0.3857 | 0.0517 | 0.2540 | 0.4669 | PAIRED identical backbone init | PRIMARY causal temporal-fusion ablation |
| E1 − UNIFORM (seed 42 only) | 42 | 1 | 0.3388 | n/a | n/a | n/a | UNPAIRED (reconstructed set_seed(42) init) | supplementary unpaired point |

## Table 3 — E1 vs UNIFORM init pairing

| seed | e1_init_pairing | uniform_init_pairing | causal_status | e1_micro_f1 | uniform_micro_f1 | delta_micro_f1 | e1_macro_f1 | uniform_macro_f1 | e1_selected_threshold | uniform_selected_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | PAIRED_exporter | PAIRED_via_exported_backbone | PAIRED | 0.5785 | 0.1702 | 0.4083 | 0.3994 | 0.1679 | 0.73 | 0.97 |
| 17 | PAIRED_exporter | PAIRED_via_exported_backbone | PAIRED | 0.6215 | 0.2888 | 0.3327 | 0.4029 | 0.2123 | 0.93 | 0.97 |
| 29 | PAIRED_exporter | PAIRED_via_exported_backbone | PAIRED | 0.6425 | 0.1956 | 0.4469 | 0.4008 | 0.1923 | 0.79 | 0.9400000000000001 |
| 42 | HISTORICAL_no_export | UNPAIRED | UNPAIRED | 0.6044 | 0.2656 | 0.3388 | 0.4166 | 0.2149 | 0.8 | 0.93 |
| 101 | PAIRED_exporter | PAIRED_via_exported_backbone | PAIRED | 0.6119 | 0.2569 | 0.3550 | 0.4040 | 0.2158 | 0.6 | 0.9400000000000001 |

## Table 4 — Micro vs macro

| arm | micro_f1_mean_sd | macro_f1_mean_sd | micro_minus_macro | weighting_note |
| --- | --- | --- | --- | --- |
| E1 | 0.6118 ± 0.0234 | 0.4048 ± 0.0069 | 0.2070 | micro=pixel-pooled; macro=unweighted mean of 21 per-fire F1 |
| ENV0 | 0.5405 ± 0.0189 | 0.3535 ± 0.0173 | 0.1871 | micro=pixel-pooled; macro=unweighted mean of 21 per-fire F1 |
| TS2 | 0.4041 ± 0.0251 | 0.2693 ± 0.0218 | 0.1348 | micro=pixel-pooled; macro=unweighted mean of 21 per-fire F1 |
| UNIFORM | 0.2354 ± 0.0501 | 0.2006 ± 0.0207 | 0.0348 | micro=pixel-pooled; macro=unweighted mean of 21 per-fire F1 |

## Table S1 — Empty-target supplement

| arm | macro_f1_full21_primary | macro_f1_positive_target_only_n19 | n_empty_target_fires | empty_target_mean_fp_sum | empty_target_mean_pred_positive_sum | zero_division_convention | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.4048 ± 0.0069 | 0.4474 ± 0.0076 | 2 | 2.40 | 2.40 | F1 = 2TP/(2TP+FP+FN+1e-8); TP=FP=FN=0 yields F1≈0 | Supplementary only; does not replace full-21 headline macro |
| ENV0 | 0.3535 ± 0.0173 | 0.3907 ± 0.0191 | 2 | 1.20 | 1.20 | F1 = 2TP/(2TP+FP+FN+1e-8); TP=FP=FN=0 yields F1≈0 | Supplementary only; does not replace full-21 headline macro |
| TS2 | 0.2693 ± 0.0218 | 0.2976 ± 0.0241 | 2 | 1.60 | 1.60 | F1 = 2TP/(2TP+FP+FN+1e-8); TP=FP=FN=0 yields F1≈0 | Supplementary only; does not replace full-21 headline macro |
| UNIFORM | 0.2006 ± 0.0207 | 0.2218 ± 0.0229 | 2 | 88.60 | 88.60 | F1 = 2TP/(2TP+FP+FN+1e-8); TP=FP=FN=0 yields F1≈0 | Supplementary only; does not replace full-21 headline macro |
