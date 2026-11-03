# Attribution and license review

**Policy:** Do not copy upstream source into this repository until the license is confirmed and this file records the decision. Prefer absolute-path references to frozen external artifacts.

Reviewed at project setup (2026-11-02). No upstream source was copied into this tree.

---

## 1. TS-SatFire dataset

| Item | Detail |
|------|--------|
| Role | Read-only GeoTIFF event tree used via `DATA_ROOT` |
| Local path | `/a/bear.cs.fiu.edu./disk/bear-b/users/mmazi007/Desktop/datasets/ts-satfire` |
| Paper | Zhao et al., *TS-SatFire: A Multi-Task Satellite Image Time-Series Dataset for Wildfire Detection and Prediction* (Nature Scientific Data; arXiv:2412.11555) |
| Dataset distribution | Kaggle / Hugging Face mirrors |
| Dataset license | **CC BY 4.0** (Attribution 4.0 International) as stated on Kaggle and Hugging Face dataset cards |
| Action in this repo | Reference only; **do not duplicate** the dataset tree |

Citation (dataset / paper):

```text
Zhao, Y., Gerard, S. & Ban, Y. TS-SatFire: A Multi-Task Satellite Image
Time-Series Dataset for Wildfire Detection and Prediction.
```

---

## 2. TS-SatFire code repository

| Item | Detail |
|------|--------|
| Upstream | https://github.com/zhaoyutim/TS-SatFire |
| Local clone | `/aul/homes/mmazi007/Desktop/Source Code (Research)/TS-SatFire` |
| LICENSE file in local tree | **Not present** at setup review |
| GitHub license metadata | No SPDX license detected in public listing at review time |
| Action in this repo | **Do not copy** scripts/models from TS-SatFire until a clear OSS license is confirmed. Reimplement loaders/protocols as needed; cite authors. |

Authors (from upstream README): Yu Zhao, Sebastian Gerard, Yifang Ban (KTH).

---

## 3. DeepFire-Forecaster (upstream baseline tree)

| Item | Detail |
|------|--------|
| Upstream | DeepFire-Forecaster |
| Local clone | `/aul/homes/mmazi007/Desktop/Source Code (Research)/DeepFire-Forecaster` |
| License file | `LICENSE` — **MIT License** |
| Copyright | Copyright (c) 2026 Nikoloas Mavros |
| Frozen commit referenced for baselines | `82c985654509abe802a760445b95d4baa843627f` |
| Action in this repo | Frozen baseline artifacts are **referenced by absolute path**. This project includes a MIT-attributed **reimplementation** of the spatial backbone in `src/wtp/models/spase_backbone.py` (`MeanPoolUNet3D`) plus an original recency-fusion head in `src/wtp/models/recency_fusion.py` (`RecencyFusionUNet3D`). Diagnostics may import upstream helpers via absolute `sys.path` adapters (`src/wtp/utils/external.py`) without modifying those trees. |

MIT permission summary: use, copy, modify, and distribute are allowed provided the copyright notice and permission notice are included in substantial portions.

---

## 4. Local reproduction phases (not upstream)

Independent reproduction / audit code living under `DeepFire-Forecaster/phase3a_*`, `phase4a_*`, and `phase4b_*` remains outside this project. This repository must not modify those directories. Reference them as frozen baselines and manifests only.

---

## 5. Mean-pool backbone reimplementation

| Item | Detail |
|------|--------|
| File | `src/wtp/models/spase_backbone.py` (`MeanPoolUNet3D`) |
| Based on | DeepFire-Forecaster `phase3a_fp_model.py` (`SEUNet3DPred`) |
| License | MIT — Copyright (c) 2026 Nikoloas Mavros |
| Modifications | Factored `encode_decode_5d` / `temporal_pool` hooks for fusion variants; no change to spatial operator semantics |

The MIT permission notice from DeepFire-Forecaster/`LICENSE` continues to apply to substantial portions derived from that work.

## 6. This project

| Item | Detail |
|------|--------|
| Name | Wildfire-Temporal-Prediction / wildfire-progression-modeling |
| Intent | Original wildfire progression modeling research |
| License | Not yet declared |
| Third-party code | Mean-pool backbone reimplementation as above; no dataset files vendored |
