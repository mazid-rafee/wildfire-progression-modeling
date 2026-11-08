"""Protocol selection and shared train-lock tests."""

from __future__ import annotations

from wtp.protocol.config import (
    ARMS,
    build_resolved_config,
    compare_train_locks,
    config_sha256,
    load_protocol,
)


def test_v1_and_v2_are_distinct():
    v1 = load_protocol("v1")
    v2 = load_protocol("v2")
    assert v1["protocol_id"] != v2["protocol_id"]
    assert v1["data"]["loader"] == "v1_phase4b"
    assert v2["data"]["loader"] == "v2_corrected"
    assert v2["data"]["directional_aug_correct"] is True
    assert v1["data"]["directional_aug_correct"] is False


def test_v2_not_silent_default_of_v1_loader():
    # Arms declare protocol explicitly
    for arm in ARMS:
        r = build_resolved_config(arm)
        assert r["protocol_id"].startswith("V2")
        assert r["data"]["loader"] == "v2_corrected"


def test_explicit_v1_override_on_arm():
    r = build_resolved_config("V2-E1", protocol_name="v1")
    assert r["protocol_id"].startswith("V1")
    assert r["data"]["loader"] == "v1_phase4b"


def test_shared_train_locks_identical():
    report = compare_train_locks()
    assert report["ok"] is True


def test_config_hash_stable():
    a = build_resolved_config("V2-E1")
    b = build_resolved_config("V2-E1")
    assert a["config_sha256"] == b["config_sha256"]
    assert len(a["config_sha256"]) == 64


def test_threshold_grid_v2_extended():
    r = build_resolved_config("V2-P4B-TS2")
    assert r["eval"]["threshold_min"] == 0.01
    assert r["eval"]["threshold_max"] == 0.99
    assert r["eval"]["tune_on_test"] is False


def test_mask_gfs_protocol():
    p = load_protocol("v2_mask_gfs")
    assert p["forecast_policy"] == "mask_zero"
    assert p["forecast_model_channels"] == [22, 23, 24, 25]
    assert p["forecast_availability_established"] is False


def test_four_arms_present():
    required = {
        "V2-P4B-TS2",
        "V2-P4B-TS6",
        "V2-E1",
        "V2-E1-TS6",
        "V2-ENV0-TS6",
        "V2-MEANPOOL-TS6-CTRL",
    }
    assert required.issubset(set(ARMS))


def test_env0_arm_masks_firepred():
    from wtp.protocol.config import load_arm

    arm = load_arm("V2-ENV0-TS6")
    assert arm["channel_ablation"] == "ENV-0"
    assert arm["mask_channels"] == list(range(8, 26))
    assert arm["model"]["name"] == "recency_fusion"
    assert arm["ts_length"] == 6
