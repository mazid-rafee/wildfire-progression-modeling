"""Regression: forecast cutoff / day-L exclusion / availability flags."""

from __future__ import annotations

from wtp.data.v2.timestamps import (
    feature_availability_table,
    parse_viirs_day_date,
    summarize_availability,
    window_temporal_contract,
)


def test_parse_viirs_day_date():
    dt = parse_viirs_day_date("/data/fire/VIIRS_Day/2018-07-04_VIIRS_Day.tif")
    assert dt is not None
    assert dt.strftime("%Y-%m-%d") == "2018-07-04"


def test_window_excludes_label_day():
    inputs = [
        "/x/2018-07-01_VIIRS_Day.tif",
        "/x/2018-07-02_VIIRS_Day.tif",
    ]
    label = "/x/2018-07-03_VIIRS_Day.tif"
    c = window_temporal_contract(inputs, label, ts=2)
    assert c["ok"] is True
    assert c["label_excluded_from_inputs"] is True
    assert c["newest_to_label_horizon_days"] == 1


def test_forecast_channels_flagged_not_demonstrably_past_obs():
    rows = feature_availability_table()
    summary = summarize_availability(rows)
    assert summary["forecast_model_channels"] == [22, 23, 24, 25]
    for ch in summary["forecast_model_channels"]:
        row = next(r for r in rows if r.get("model_channel") == ch)
        assert row["demonstrably_available_at_cutoff"] is False
        assert row["availability"] == "declared_forecast_lookahead"


def test_omitted_precip_flagged():
    rows = feature_availability_table()
    omit = next(r for r in rows if r.get("source_band_index_0based") == 2)
    assert omit["in_model"] is False
    assert "omitted" in omit["availability"]
