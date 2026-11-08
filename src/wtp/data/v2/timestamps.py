"""Forecast issue / valid-time verification and feature availability flags.

Cutoff policy (matched protocol):
  - Input days: files at indices [L-TS, ..., L-1]  (day L excluded from X)
  - Label day: file index L; target = clip(BA_L - BA_{L-1})
  - BA channel on newest frame uses BA at L-1

GFS forecast FirePred bands (source 15–18 → model 22–25) are issued at the
end of the calendar day of each input frame and encode next-day weather —
declared forecast features relative to that frame's issue time, not past obs.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from wtp.data.channel_map import FIREPRED_SOURCE_NAMES, FP_BAND_IDX, build_model_channel_table

_DAY_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_VIIRS_Day")


def parse_viirs_day_date(path: str) -> Optional[datetime]:
    name = Path(path).name
    m = _DAY_RE.search(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)


def window_temporal_contract(
    input_days: Sequence[str],
    label_day: str,
    ts: int,
) -> dict[str, Any]:
    """Verify day-L exclusion and report calendar gaps / forecast horizons."""
    if len(input_days) != ts:
        return {"ok": False, "error": f"expected {ts} input days, got {len(input_days)}"}
    dates = [parse_viirs_day_date(p) for p in input_days]
    label_dt = parse_viirs_day_date(label_day)
    if any(d is None for d in dates) or label_dt is None:
        return {
            "ok": False,
            "error": "unparseable VIIRS_Day filename date",
            "input_days": list(input_days),
            "label_day": label_day,
        }
    # Day L must not appear in inputs
    label_in_inputs = any(d.date() == label_dt.date() for d in dates)  # type: ignore[union-attr]
    gaps = []
    for a, b in zip(dates, dates[1:]):
        gaps.append(int((b - a).days))  # type: ignore[operator]
    newest = dates[-1]
    horizon_to_label = int((label_dt - newest).days)  # type: ignore[operator]
    return {
        "ok": not label_in_inputs and horizon_to_label >= 1,
        "label_excluded_from_inputs": not label_in_inputs,
        "input_dates": [d.strftime("%Y-%m-%d") for d in dates],  # type: ignore[union-attr]
        "label_date": label_dt.strftime("%Y-%m-%d"),
        "calendar_gaps_between_inputs": gaps,
        "max_input_gap_days": max(gaps) if gaps else 0,
        "newest_to_label_horizon_days": horizon_to_label,
        "ba_channel_newest_uses": "L-1 (newest input day)",
        "forecast_issue_time_newest_frame": {
            "issue_date": newest.strftime("%Y-%m-%d"),  # type: ignore[union-attr]
            "issue_note": (
                "GFS bands on frame t are treated as forecast issued end of that "
                "calendar day (dataset contract / channel_map)."
            ),
            "valid_date_approx": (newest + timedelta(days=1)).strftime("%Y-%m-%d"),  # type: ignore[operator]
        },
    }


def feature_availability_table(forecast_cutoff: str = "end_of_day_L-1") -> list[dict[str, Any]]:
    """Per-model-channel availability relative to the forecasting cutoff.

    Cutoff = end of newest input day (L-1). Features that cannot be shown to be
    available at that cutoff are flagged ``availability=uncertain_or_lookahead``.
    """
    rows = []
    for row in build_model_channel_table():
        kind = row.get("kind")
        ch = row["model_channel"]
        name = row["name"]
        if kind == "forecast":
            avail = "declared_forecast_lookahead"
            demonstrably_available = False
            note = (
                f"GFS forecast feature: valid after issue at {forecast_cutoff}; "
                "must be labelled as forecast, not past observation."
            )
        elif kind in ("observation", "observation_derived", "observation_label_derived"):
            avail = "available_at_or_before_cutoff"
            demonstrably_available = True
            note = "Tied to VIIRS/GRIDMET observation day ≤ L-1 for inputs."
        elif kind == "static_or_slow":
            avail = "available_at_or_before_cutoff"
            demonstrably_available = True
            note = "Static/slow terrain or land-cover; treated as known at cutoff."
        elif kind == "categorical_as_float":
            avail = "available_at_or_before_cutoff"
            demonstrably_available = True
            note = "Land-cover class id; static map assumed known at cutoff."
        else:
            avail = "uncertain"
            demonstrably_available = False
            note = f"Unhandled kind={kind}"
        rows.append(
            {
                "model_channel": ch,
                "name": name,
                "kind": kind,
                "availability": avail,
                "demonstrably_available_at_cutoff": demonstrably_available,
                "forecast_cutoff": forecast_cutoff,
                "note": note,
                "stream": row.get("stream"),
            }
        )

    # Omitted Total Precipitation (source idx 2) — available in rasters but not in model
    omit = FIREPRED_SOURCE_NAMES[2]
    rows.append(
        {
            "model_channel": None,
            "source_band_index_0based": 2,
            "name": omit["name"],
            "kind": omit["kind"],
            "availability": "available_in_source_but_omitted_from_model",
            "demonstrably_available_at_cutoff": True,
            "forecast_cutoff": forecast_cutoff,
            "note": omit.get("skip_reason"),
            "in_model": False,
        }
    )
    return rows


def summarize_availability(rows: Optional[list[dict]] = None) -> dict[str, Any]:
    rows = rows or feature_availability_table()
    flagged = [
        r
        for r in rows
        if not r.get("demonstrably_available_at_cutoff", False)
        or r.get("availability")
        in (
            "declared_forecast_lookahead",
            "uncertain",
            "available_in_source_but_omitted_from_model",
        )
    ]
    return {
        "forecast_cutoff": "end_of_day_L-1",
        "n_model_channels": sum(1 for r in rows if r.get("model_channel") is not None),
        "n_flagged": len(flagged),
        "flagged": flagged,
        "forecast_model_channels": [
            r["model_channel"]
            for r in rows
            if r.get("kind") == "forecast" and r.get("model_channel") is not None
        ],
        "policy": (
            "Keep GFS bands as declared forecast features for V2 training; "
            "do not describe them as past observations. Omitted precip remains out of model."
        ),
    }
