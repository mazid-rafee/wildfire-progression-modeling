# Reproducibility policy

## Non-interference

- Do not modify, move, delete, or overwrite:
  - `TS-SatFire/`
  - `DeepFire-Forecaster/` (including phase3a / phase4a / phase4b trees)
  - `TS-SatFire-Phase1-EventAudit/`, `TS-SatFire-Phase2-SpaSE-Readiness/`
  - Anything under the verified `DATA_ROOT`
- Preserve all Phase 3A and Phase 4B checkpoints, logs, and results in place.

## Data

- Resolve dataset location from `configs/paths.yaml` → `data.root`.
- Canonical `DATA_ROOT`: `/a/bear.cs.fiu.edu./disk/bear-b/users/mmazi007/Desktop/datasets/ts-satfire`
- Read-only open flags; no writes, caches, or derived GeoTIFFs under `DATA_ROOT`.
- Do not copy or regenerate the TS-SatFire dataset into this project.

## Baselines and manifests

- Load Phase 3A / 4A / 4B references via absolute paths in `configs/paths.yaml`.
- Prefer reading manifests and metrics; do not rewrite them.
- If a local derived view is needed (e.g., filtered key list), write it under this project’s `outputs/` or `runs/` only.

## Experiments

- Config-driven: every run should record the resolved config snapshot in its run directory.
- Seed, TS length, population ID (published vs matched), and eval protocol version are mandatory metadata (`src/wtp/tracking`).
- Checkpoints, logs, and large arrays stay under gitignored `checkpoints/`, `runs/`, `logs/`, `outputs/`.

## Code provenance

- See `docs/ATTRIBUTION.md`.
- No upstream source was vendored at setup. Future copies require license confirmation + attribution update in the same PR/commit.

## Launch gate

- Training / GPU inference entrypoints must refuse to run unless an explicit approval flag is provided (to be implemented before first Phase 4C launch).
- Project setup intentionally does **not** start training or experiments.
