"""Regression: BA progression label construction."""

from __future__ import annotations

import numpy as np

from wtp.data.v2.geo_align import center_crop_hw


def test_progression_clip_nonnegative():
    ba_last = np.zeros((64, 64), dtype=np.float32)
    ba_next = np.zeros((64, 64), dtype=np.float32)
    ba_last[10:20, 10:20] = 1.0
    ba_next[10:25, 10:20] = 1.0  # grew south
    lab = np.clip(ba_next - ba_last, 0, 1)
    assert lab.sum() == 5 * 10
    assert lab.min() >= 0 and lab.max() <= 1


def test_label_crop_uses_viirs_hw():
    H, W, cs = 300, 300, 256
    lab = np.zeros((H, W), dtype=np.float32)
    lab[100, 100] = 1.0
    crop = center_crop_hw(lab, cs, H, W)
    assert crop.shape == (cs, cs)
    r0 = (H - cs) // 2
    c0 = (W - cs) // 2
    assert crop[100 - r0, 100 - c0] == 1.0
