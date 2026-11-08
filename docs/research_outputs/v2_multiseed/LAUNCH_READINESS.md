# V2 Multi-Seed Launch-Readiness Report

**Status:** PREPARED — **do not train until explicitly approved**  
**Protocol:** locked `V2.0-corrected-protocol` / shared lock `v2_p0_shared_20261103`  
**Benchmark label:** exploratory reused test population (see `REPORTING_PLAN.md`)

---

## 1. Threshold inconsistency — resolved

SpaSE TS=6 seed-42 selected thr **0.95** on a complete **0.01–0.99** grid; `selected_is_grid_ceiling=false`; F1 peaks at 0.95 then falls. Not a max=0.99 bug. Details: `THRESHOLD_ERRATUM.md`. Frozen headlines unchanged.

---

## 2. Frozen seed-42 assets (preserved; do not retrain)

| Arm | Run dir | Test24 F1 | Thr | Action |
|---|---|---:|---:|---|
| E1 | `runs/v2_matched_e1_ts6_gfs_mask_seed42_20261103_162551` | 0.6044 | 0.80 | **REUSE** |
| TS2 | `runs/v2_matched_spase_ts2_seed42_20261103_162551` | 0.4264 | 0.89 | **REUSE** |
| ENV0 | `runs/v2_matched_env0_ts6_seed42_20261103_162551` | 0.5121 | 0.90 | **REUSE** |
| SpaSE TS6 | `runs/v2_matched_spase_ts6_seed42_20261103_162551` | 0.1616 | 0.95 | **ARCHIVAL** (not multi-seeded) |

Checkpoints, configs, metrics, logs: untouched.

---

## 3. Multi-seed arms (publication matrix)

| Key | Arm ID | Temporal op | Notes |
|---|---|---|---|
| **E1** | `V2-E1-TS6` | Learnable RecencyFusion | Primary claim |
| **TS2** | `V2-P4B-TS2` | Mean-pool T=2 | Short-horizon baseline |
| **UNIFORM** | `V2-E1-UNIFORM-TS6` | RecencyFusion **frozen uniform 1/T** | Causal control vs E1 |
| **ENV0** | `V2-ENV0-TS6` | Learnable + mask ch 8–25 | Ablation |

**Seeds:** `42, 7, 17, 29, 101`  
**Cells:** 20 total · **3 reuse** · **17 to train**

Identical locked settings: matched pops (983/135/561/553/531), V2 FP stats, GFS `mask_zero`, loss/opt/OneCycle, 60 epochs, bs=8, lr=5e-4, pos_weight, ckpt=val F1@0.5, thr val-only 0.01–0.99. **No HP/arch tuning from test.**

### Frozen-uniform control (implemented, untrained)
- Config: `configs/experiments/v2/e1_frozen_uniform_ts6.yaml`
- `freeze_temporal_fusion_uniform()` → logits=0, scale=0, `requires_grad=False`
- Trainable params = **24,279,561** (= MeanPool)
- Unit tests: `tests/v2/test_frozen_uniform_fusion.py` (**2 passed**)
- Harness: same `scripts/v2_train_eval.py` as E1; AdamW skips frozen params
- CLI: `--seed`, `--export-init-backbone`, `--load-init-backbone`
- Aug RNG now follows experiment seed (`aug_seed` in `make_datasets`)

### Paired init (E1 ↔ UNIFORM)
For seeds **7, 17, 29, 101**: E1 exports `outputs/v2_multiseed/inits/backbone_init_seed{S}.pt` before/at train start; UNIFORM loads it then re-freezes fusion.  
For seed **42**: UNIFORM uses `set_seed(42)` reconstruction (historical E1 init was not exported) — documented caveat, not a blocker for launching UNIFORM_seed42.

---

## 4. Resource estimate (inspect-only; GPUs not occupied)

Snapshot at prep time:

| GPU | Free MiB | Notes |
|---:|---:|---|
| 0 | ~77k | Preferred |
| 1 | ~53k | Preferred |
| 4 | ~67k | Preferred (some util) |
| 7 | ~61k | Preferred (some util) |
| 2,3,5 | lower free | Avoid for V2 |

| Arm | Prior wall (seed42) |
|---|---:|
| TS2 | 86.5 min |
| E1 | 257 min |
| ENV0 | 224 min |
| UNIFORM (est.) | ~250 min |
| Disk free | ~30 TB |

**17 new jobs ≈ 58.7 GPU-hours.**  
**4-GPU wall:** ~15 h optimistic · ~18 h typical · ~24 h conservative.  
Peak host RAM: avoid stacking >2 concurrent T=6 float16 preloads (~12–15 GiB each).

Full matrix/schedule: `matrix/experiment_matrix.json`, `matrix/schedule_plan.json`.  
Example commands (not executed): `scripts/example_launch_commands.json`.

---

## 5. Proposed scheduling (resource-safe)

1. **Wave 1:** `UNIFORM_seed42` alone (reuse other seed-42 arms).  
2. **Waves 2–5 (per seed 7→17→29→101):**  
   - Parallel: E1 (export init) + TS2 + ENV0 on GPUs 0/1/4  
   - Then: UNIFORM on freed GPU with `--load-init-backbone`  
3. Stagger preload 0/90/180/270 s; `--min-free-vram-gib 14`.  
4. Do not use GPUs 2/3/5 unless free memory recovers.

---

## 6. Reporting (planned, post-train)

See `REPORTING_PLAN.md`: micro + macro F1, IoU, P/R, paired per-fire Δ, **fire-event cluster bootstrap** CIs (not pixel/window independence). Always label test24 as exploratory reused benchmark; recommend independent final events if obtainable.

---

## 7. Integrity blockers / caveats

| ID | Severity | Item |
|---|---|---|
| B0 | None | Thr 0.95 vs 0.99 — **resolved** (erratum) |
| B1 | Low | Seed-42 E1↔UNIFORM init not file-verified (seed reconstruction) |
| B2 | Low | CUDA nondeterminism across GPUs; multi-seed absorbs variance |
| B3 | Process | Training gated until explicit approval (`--i-understand-full-train`) |
| B4 | Science | test24 is exploratory reused set — label required |
| B5 | None | No DATA_ROOT / external-repo / frozen-ckpt mutations in this prep |

**No blockers that prevent a future approved launch** of the prepared matrix.

---

## 8. Prep deliverables

| Path | Purpose |
|---|---|
| `THRESHOLD_ERRATUM.md` | 0.95 vs 0.99 resolution |
| `LAUNCH_READINESS.md` | This document |
| `REPORTING_PLAN.md` | Metrics + fire bootstrap + exploratory label |
| `matrix/experiment_matrix.json` | Full 20-cell matrix |
| `matrix/schedule_plan.json` | GPU waves |
| `scripts/example_launch_commands.json` | Commands (not run) |
| `configs/experiments/v2/e1_frozen_uniform_ts6.yaml` | New arm |
| Code: freeze helpers, `--seed`, init export/load, `aug_seed` | Launch wiring |

---

## 9. Explicit non-actions this session

- No GPU training started  
- No frozen checkpoint / DATA_ROOT / external-repo edits  
- No git commit or push  
- No hyperparameter or architecture changes based on test scores  
