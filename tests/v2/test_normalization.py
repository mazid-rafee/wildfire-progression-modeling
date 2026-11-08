"""Regression: V2 norm formula + train-only policy markers."""

from __future__ import annotations

import numpy as np

from wtp.data.v2.normalize import compare_to_v1, normalize_firepred


def test_normalize_firepred_clip_and_std_floor():
    fp = np.zeros((18, 4, 4), dtype=np.float32)
    fp[0] = 100.0
    mean = np.zeros(18, dtype=np.float32)
    std = np.full(18, 0.01, dtype=np.float32)  # below floor 0.1
    out = normalize_firepred(fp, mean, std)
    # (100 - 0) / 0.1 = 1000 → clipped to 5
    assert np.allclose(out[0], 5.0)


def test_compare_to_v1_l2():
    a = np.arange(18, dtype=np.float32)
    b = a + 1.0
    d = compare_to_v1(a, a, b, b)
    assert abs(d["l2_mean"] - float(np.sqrt(18.0))) < 1e-5
    assert abs(d["l2_std"] - float(np.sqrt(18.0))) < 1e-5


def test_unique_day_key_helper_dedups():
    from wtp.data.v2.normalize import _unique_train_day_keys

    rows = [
        {"fire_id": "1", "label_idx": 3, "target_key": "a"},
        {"fire_id": "1", "label_idx": 4, "target_key": "b"},
    ]
    fire_by_id = {
        "1": {
            "fire_id": "1",
            "fire_dir": "/f/1",
            "day_files": [f"/f/1/d{i}.tif" for i in range(6)],
        }
    }
    keys = _unique_train_day_keys(rows, fire_by_id, ts=2)
    # windows use days[1:3] and days[2:4] → unique {1,2,3}
    paths = sorted(k[2] for k in keys)
    assert paths == ["/f/1/d1.tif", "/f/1/d2.tif", "/f/1/d3.tif"]
