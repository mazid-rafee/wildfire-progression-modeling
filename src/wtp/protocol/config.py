"""Load and hash versioned protocol + arm configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from wtp.utils.paths import PROJECT_ROOT

PROTOCOL_DIR = PROJECT_ROOT / "configs" / "protocol"
ARM_DIR = PROJECT_ROOT / "configs" / "experiments" / "v2"
SHARED_TRAIN = ARM_DIR / "_shared_train.yaml"

PROTOCOLS = {
    "v1": "v1_frozen.yaml",
    "v1_frozen": "v1_frozen.yaml",
    "v2": "v2_corrected.yaml",
    "v2_corrected": "v2_corrected.yaml",
    "v2_mask_gfs": "v2_corrected_mask_gfs_forecast.yaml",
    "v2_corrected_mask_gfs_forecast": "v2_corrected_mask_gfs_forecast.yaml",
}

ARMS = {
    "V2-P4B-TS2": "spase_meanpool_ts2.yaml",
    "V2-P4B-TS6": "spase_meanpool_ts6.yaml",
    "V2-E1": "e1_recency_ts6.yaml",
    "V2-E1-TS6": "e1_recency_ts6.yaml",
    "V2-ENV0-TS6": "env0_recency_ts6.yaml",
    "V2-MEANPOOL-TS6-CTRL": "meanpool_ts6_e1_harness.yaml",
    "V2-E1-UNIFORM-TS6": "e1_frozen_uniform_ts6.yaml",
}

# Fields that must be identical across all P0 arms
TRAIN_LOCK_KEYS = (
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "amp",
    "drop_last",
    "grad_clip",
    "pos_weight",
    "optimizer",
    "scheduler",
    "loss",
    "checkpoint",
    "threshold",
    "population",
    "expected_counts",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_shared_train() -> dict[str, Any]:
    return _read_yaml(SHARED_TRAIN)


def load_protocol(name: str) -> dict[str, Any]:
    key = name.strip()
    if key not in PROTOCOLS:
        raise KeyError(f"Unknown protocol '{name}'. Choose from: {sorted(PROTOCOLS)}")
    path = PROTOCOL_DIR / PROTOCOLS[key]
    cfg = _read_yaml(path)
    # Resolve inherits_protocol for conservative variants
    parent = cfg.pop("inherits_protocol", None)
    if parent:
        base = _read_yaml(PROTOCOL_DIR / parent)
        merged = {**base, **cfg}
        # deep-merge data / train / eval lightly
        for section in ("data", "train", "eval"):
            if section in base or section in cfg:
                merged[section] = {**(base.get(section) or {}), **(cfg.get(section) or {})}
        cfg = merged
    cfg["_protocol_yaml"] = str(path)
    cfg["_protocol_key"] = key
    return cfg


def load_arm(arm_id: str) -> dict[str, Any]:
    if arm_id not in ARMS:
        raise KeyError(f"Unknown arm '{arm_id}'. Choose from: {sorted(ARMS)}")
    path = ARM_DIR / ARMS[arm_id]
    cfg = _read_yaml(path)
    cfg["_arm_yaml"] = str(path)
    return cfg


def config_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_resolved_config(
    arm_id: str,
    protocol_name: str | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge shared train locks + protocol + arm into one resolved config.

    ``protocol_name`` defaults to the arm's ``protocol`` field. Passing an
    explicit protocol never silently swaps V1↔V2.
    """
    arm = load_arm(arm_id)
    proto_key = protocol_name or arm.get("protocol") or "v2_corrected"
    if protocol_name is None and arm.get("protocol"):
        proto_key = arm["protocol"]
    # Map arm shorthand
    if proto_key == "v2_corrected":
        proto_key = "v2"
    protocol = load_protocol(proto_key)
    shared = load_shared_train()

    resolved = {
        "arm_id": arm_id,
        "arm": arm,
        "protocol": protocol,
        "shared_train": shared,
        "protocol_key": proto_key,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_version": protocol.get("protocol_version"),
        "preprocessing_version": protocol.get("protocol_version"),
        "ts_length": int(arm["ts_length"]),
        "model": dict(arm["model"]),
        "forecast_policy": protocol.get("forecast_policy"),
        "forecast_model_channels": list(protocol.get("forecast_model_channels") or []),
        "forecast_availability_established": bool(
            protocol.get("forecast_availability_established", False)
        ),
        "train": {
            **{k: shared[k] for k in TRAIN_LOCK_KEYS if k in shared},
            "enabled": False,  # hard default; full train requires explicit flag
        },
        "data": dict(protocol.get("data") or {}),
        "eval": {
            **(protocol.get("eval") or {}),
            "threshold_min": shared["threshold"]["min"],
            "threshold_max": shared["threshold"]["max"],
            "threshold_step": shared["threshold"]["step"],
            "tune_on_test": shared["threshold"]["tune_on_test"],
        },
    }
    if overrides:
        resolved["overrides"] = overrides
        # shallow apply known override keys
        if "forecast_policy" in overrides:
            resolved["forecast_policy"] = overrides["forecast_policy"]
            resolved["protocol"]["forecast_policy"] = overrides["forecast_policy"]
        if "train_enabled" in overrides:
            resolved["train"]["enabled"] = bool(overrides["train_enabled"])

    resolved["config_sha256"] = config_sha256(
        {
            "arm_id": arm_id,
            "protocol_id": resolved["protocol_id"],
            "protocol_version": resolved["protocol_version"],
            "ts_length": resolved["ts_length"],
            "model": resolved["model"],
            "train_locks": resolved["train"],
            "data": resolved["data"],
            "eval": resolved["eval"],
            "forecast_policy": resolved["forecast_policy"],
        }
    )
    resolved["shared_train_lock_id"] = shared.get("shared_train_lock_id")
    return resolved


def compare_train_locks(arm_ids: list[str] | None = None) -> dict[str, Any]:
    """Ensure all listed arms resolve to identical shared train locks."""
    arm_ids = arm_ids or list(ARMS.keys())
    shared = load_shared_train()
    mismatches = []
    snapshots = {}
    for arm_id in arm_ids:
        resolved = build_resolved_config(arm_id)
        snap = {k: resolved["train"].get(k) for k in TRAIN_LOCK_KEYS}
        snapshots[arm_id] = snap
        for k in TRAIN_LOCK_KEYS:
            if snap.get(k) != shared.get(k) and k in shared:
                # train.enabled is forced False in resolved — skip
                if k == "enabled":
                    continue
                if json.dumps(snap.get(k), sort_keys=True, default=str) != json.dumps(
                    shared.get(k), sort_keys=True, default=str
                ):
                    mismatches.append({"arm": arm_id, "key": k, "arm_val": snap.get(k), "shared": shared.get(k)})
    # pairwise equality
    ref = snapshots[arm_ids[0]]
    for arm_id in arm_ids[1:]:
        if snapshots[arm_id] != ref:
            mismatches.append({"arm": arm_id, "error": "train_lock_differs_from_first_arm"})
    return {
        "ok": len(mismatches) == 0,
        "shared_train_lock_id": shared.get("shared_train_lock_id"),
        "n_arms": len(arm_ids),
        "mismatches": mismatches,
        "snapshots": snapshots,
    }
