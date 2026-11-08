# V2 Multi-Seed Launch Status (immediate)

**UTC:** 2026-11-04T04:53Z  
**Training jobs launched:** **0**  
**Pending in queue:** **17**  
**Queue controller:** **alive** (PID **1919990**, detached)

## GPU safety checks

| GPU | Decision | Why |
|---:|---|---|
| **0** | **Unsafe** | Foreign processes (`ashir018`, `tvo013`) hold ~5.3 GiB; no share permission. Free ~76 GiB / 0% util is **not** treated as available. |
| **1** | **Unsafe** | Qwen serving (`bdas004`) + other foreign mem ~27.8 GiB. Blocked by policy. |
| **4** | **Unsafe** | Busy (~90–100% sustained util); free ~15.9 GiB &lt; 22 GiB floor. |
| **7** | **Unsafe** | Busy (~99% sustained util); free ~9.8 GiB. |
| 2,3,5,6 | Forbidden | Not in approved set. |

**Policy:** allowed {0,1,4,7}; min free 22 GiB; sustained util &lt;10%; foreign mem ≤2 GiB for share; hard-block Qwen on GPU 1. No processes terminated.

## Jobs

| Status | Count |
|---|---:|
| Reused seed-42 (E1/TS2/ENV0) | 3 (not retrained) |
| Newly launched | **0** |
| Pending | **17** |
| Running | 0 |
| Failed | 0 |

No `v2_train_eval.py` training PIDs exist. Controllers will launch only after a fresh recheck shows a safe GPU.

## Queue controller

| Field | Value |
|---|---|
| PID | `1919990` |
| Log | `outputs/v2_multiseed/logs/queue_controller.log` |
| State | `outputs/v2_multiseed/queue/ACTIVE.json` |
| Live status | `outputs/v2_multiseed/queue/STATUS.json` |
| Safety snapshot | `outputs/v2_multiseed/queue/gpu_safety.json` |
| Poll interval | 120 s |

## Preflight passed (non-GPU)

- Critical unit tests: **18 passed** (incl. frozen-uniform)
- Freeze stays exactly 1/6 across Adam steps; scale not learnable
- Mean-pool equivalence OK at atol 1e-4
- Seed-42 reconstructed init SHA256 saved; **E1↔UNIFORM seed42 labeled UNPAIRED**
- Details: `outputs/v2_multiseed/preflight/freeze_and_init_verification.json`

## Integrity / resource blockers (open)

1. **No approved GPU currently safe** → all 17 runs remain queued.  
2. Seed-42 uniform vs historical E1 is **unpaired** for init (documented).  
3. When GPUs 4/7 free or foreign tenants leave 0/1, controller will auto-launch with recheck; E1 still exports init before UNIFORM for seeds ≠42.

## Watch

```bash
tail -f outputs/v2_multiseed/logs/queue_controller.log
cat outputs/v2_multiseed/queue/STATUS.json
```
