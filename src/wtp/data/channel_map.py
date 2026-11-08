"""Verified 27-channel model input dictionary (matched SpaSE / E1 layout).

Sources:
  - TS-SatFire Scientific Data Table 1 (27-channel publication sequence)
  - DeepFire phase3a_fp_config.FP_BAND_IDX (skips FirePred source index 2)
  - phase3a_fp_data.load_frame_sat / load_firepred_raw / build_window_tensor_ts
  - Empirical GeoTIFF probes under DATA_ROOT (this project audit)

Model tensor layout per timestep: (C=27, H, W) with
  [0:8]   VIIRS sat (6 day + 2 night), z-scored with CFG.SAT_MEAN/STD
  [8:26]  FirePred 18 bands (19 raw minus skipped index 2), z-scored with
          matched train fp_stats mean/std, clipped to [-5, 5]
  [26:27] BA cumulative mask at that day (0/1; newest frame uses ba at L-1)

Publication FirePred band 3 (0-based index 2, Total Precipitation / GRIDMET)
is **not present** in the 27-channel model — Phase 3A/4B/E1 skip it.
"""

from __future__ import annotations

from typing import Any

# FirePred raw GeoTIFF band index (0-based) → keep order used by CFG.FP_BAND_IDX
FP_BAND_IDX: list[int] = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# Publication names for all 19 FirePred bands (1-based FirePred band = index+1)
FIREPRED_SOURCE_NAMES: dict[int, dict[str, Any]] = {
    0: {
        "name": "NDVI",
        "group": "vegetation_landcover",
        "source": "VIIRS VNP13A1 (derived)",
        "kind": "observation_derived",
        "units": "scaled index (≈NDVI×10000 in rasters)",
        "time_varying": True,
    },
    1: {
        "name": "EVI",
        "group": "vegetation_landcover",
        "source": "VIIRS VNP13A1 (derived)",
        "kind": "observation_derived",
        "units": "scaled index (≈EVI×10000 in rasters)",
        "time_varying": True,
    },
    2: {
        "name": "Total precipitation (GRIDMET)",
        "group": "meteorology",
        "source": "GRIDMET",
        "kind": "observation",
        "units": "mm (often sparse/zero)",
        "time_varying": True,
        "in_model": False,
        "skip_reason": (
            "Excluded by CFG.FP_BAND_IDX. Code comment says 'all-zeros probe band'; "
            "publication names it Total Precipitation. Probe: often but not always zero "
            "across fires — treated as skipped precip observation, not a true probe."
        ),
    },
    3: {
        "name": "Wind speed",
        "group": "wind",
        "source": "GRIDMET",
        "kind": "observation",
        "units": "m/s (approx.)",
        "time_varying": True,
    },
    4: {
        "name": "Wind direction",
        "group": "wind",
        "source": "GRIDMET",
        "kind": "observation",
        "units": "degrees [0, 360]",
        "time_varying": True,
    },
    5: {
        "name": "Minimum temperature",
        "group": "meteorology",
        "source": "GRIDMET",
        "kind": "observation",
        "units": "K (raster values ~260–305)",
        "time_varying": True,
    },
    6: {
        "name": "Maximum temperature",
        "group": "meteorology",
        "source": "GRIDMET",
        "kind": "observation",
        "units": "K (raster values ~268–319)",
        "time_varying": True,
    },
    7: {
        "name": "Energy Release Component (ERC)",
        "group": "drought_fire_danger",
        "source": "NFDRS / GRIDMET ecosystem",
        "kind": "observation_derived",
        "units": "NFDRS ERC index",
        "time_varying": True,
    },
    8: {
        "name": "Specific humidity",
        "group": "meteorology",
        "source": "GRIDMET",
        "kind": "observation",
        "units": "kg/kg (small positive)",
        "time_varying": True,
    },
    9: {
        "name": "Slope",
        "group": "terrain",
        "source": "DEM-derived",
        "kind": "static_or_slow",
        "units": "degrees (approx.)",
        "time_varying": False,
    },
    10: {
        "name": "Aspect",
        "group": "terrain",
        "source": "DEM-derived",
        "kind": "static_or_slow",
        "units": "degrees [0, 360]",
        "time_varying": False,
    },
    11: {
        "name": "Elevation",
        "group": "terrain",
        "source": "DEM",
        "kind": "static_or_slow",
        "units": "m",
        "time_varying": False,
    },
    12: {
        "name": "Palmer Drought Severity Index (PDSI)",
        "group": "drought_fire_danger",
        "source": "GRIDMET / drought product",
        "kind": "observation_derived",
        "units": "PDSI index",
        "time_varying": True,
    },
    13: {
        "name": "Land cover",
        "group": "vegetation_landcover",
        "source": "land-cover map",
        "kind": "categorical_as_float",
        "units": "class id (≈1–17); z-scored as continuous in this pipeline (not one-hot)",
        "time_varying": False,
    },
    14: {
        "name": "Total precipitation, surface",
        "group": "meteorology",
        "source": "GRIDMET surface precip",
        "kind": "observation",
        "units": "mm (often sparse/zero)",
        "time_varying": True,
    },
    15: {
        "name": "Forecast wind speed",
        "group": "wind",
        "source": "GFS (24h mean from forecast issued end of current day)",
        "kind": "forecast",
        "units": "m/s (approx.)",
        "time_varying": True,
    },
    16: {
        "name": "Forecast wind direction",
        "group": "wind",
        "source": "GFS",
        "kind": "forecast",
        "units": "degrees (raster range observed ≈[-90, 90], not identical to GRIDMET 0–360)",
        "time_varying": True,
    },
    17: {
        "name": "Forecast temperature",
        "group": "meteorology",
        "source": "GFS",
        "kind": "forecast",
        "units": "°C-like in rasters (observed ≈[-2, 31]); not same scale as GRIDMET K",
        "time_varying": True,
    },
    18: {
        "name": "Forecast specific humidity",
        "group": "meteorology",
        "source": "GFS",
        "kind": "forecast",
        "units": "kg/kg (small positive)",
        "time_varying": True,
    },
}

SAT_DAY = [
    ("I1", "VIIRS_Day band 1", "reflectance"),
    ("I2", "VIIRS_Day band 2", "reflectance"),
    ("I3", "VIIRS_Day band 3", "reflectance"),
    ("I4", "VIIRS_Day band 4", "brightness/temp-like"),
    ("I5", "VIIRS_Day band 5", "brightness/temp-like"),
    ("M11", "VIIRS_Day band 6", "reflectance"),
]
SAT_NIGHT = [
    ("I4_Night", "VIIRS_Night band 1", "brightness/temp-like"),
    ("I5_Night", "VIIRS_Night band 2", "brightness/temp-like"),
]


def build_model_channel_table(fp_mean=None, fp_std=None) -> list[dict[str, Any]]:
    """Return one row per model channel 0..26."""
    rows: list[dict[str, Any]] = []
    for i, (name, src, units) in enumerate(SAT_DAY):
        rows.append(
            {
                "model_channel": i,
                "stream": "VIIRS_Day",
                "source_band_index_1based": i + 1,
                "name": name,
                "units": units,
                "normalization": "SAT_MEAN/SAT_STD then nan_to_num→0",
                "missing_handling": "nan/inf → 0 after z-score",
                "time_varying": True,
                "kind": "observation",
                "ablation_group": None,
                "is_firepred": False,
                "in_env0_retained": True,
            }
        )
    for j, (name, src, units) in enumerate(SAT_NIGHT):
        rows.append(
            {
                "model_channel": 6 + j,
                "stream": "VIIRS_Night",
                "source_band_index_1based": j + 1,
                "name": name,
                "units": units,
                "normalization": "SAT_MEAN/SAT_STD then nan_to_num→0",
                "missing_handling": "missing night file → zeros before norm",
                "time_varying": True,
                "kind": "observation",
                "ablation_group": None,
                "is_firepred": False,
                "in_env0_retained": True,
            }
        )
    for slot, src_i in enumerate(FP_BAND_IDX):
        meta = FIREPRED_SOURCE_NAMES[src_i]
        row = {
            "model_channel": 8 + slot,
            "stream": "FirePred",
            "source_band_index_0based": src_i,
            "source_band_index_1based": src_i + 1,
            "firepred_slot": slot,
            "name": meta["name"],
            "units": meta["units"],
            "normalization": "train-only fp_stats z-score; std floor 0.1; clip [-5,5]; nan→0",
            "missing_handling": "missing FirePred file → zero FP crop; nan_to_num in norm",
            "time_varying": meta["time_varying"],
            "kind": meta["kind"],
            "ablation_group": meta["group"],
            "is_firepred": True,
            "in_env0_retained": False,
            "source_product": meta["source"],
        }
        if fp_mean is not None:
            row["fp_train_mean"] = float(fp_mean[slot])
        if fp_std is not None:
            row["fp_train_std"] = float(fp_std[slot])
        rows.append(row)
    rows.append(
        {
            "model_channel": 26,
            "stream": "VIIRS_Day_BA",
            "source_band_index_1based": 8,
            "name": "Burned area cumulative mask",
            "units": "binary {0,1} from isfinite(band8)",
            "normalization": "none (0/1)",
            "missing_handling": "missing BA → RuntimeError at build; older non-newest frames may zero-fill",
            "time_varying": True,
            "kind": "observation_label_derived",
            "ablation_group": None,
            "is_firepred": False,
            "in_env0_retained": True,
            "notes": (
                "Newest frame uses ba_last at day L-1. Target y uses ba(L)-ba(L-1); "
                "day L never appears in X."
            ),
        }
    )
    return rows


# Disjoint ablation groups over FirePred model channels only
ABLATION_GROUPS: dict[str, dict[str, Any]] = {
    "ENV-0": {
        "title": "No FirePred (all 18 auxiliary channels)",
        "model_channels": list(range(8, 26)),
        "description": "Retain VIIRS day/night + BA only.",
    },
    "ENV-WIND": {
        "title": "Wind (obs + forecast)",
        "model_channels": [8 + FP_BAND_IDX.index(i) for i in (3, 4, 15, 16)],
        "description": "Wind speed/direction + forecast wind speed/direction.",
    },
    "ENV-TERRAIN": {
        "title": "Terrain",
        "model_channels": [8 + FP_BAND_IDX.index(i) for i in (9, 10, 11)],
        "description": "Slope, aspect, elevation.",
    },
    "ENV-VEG": {
        "title": "Vegetation and land cover",
        "model_channels": [8 + FP_BAND_IDX.index(i) for i in (0, 1, 13)],
        "description": "NDVI, EVI, land-cover class id (numeric).",
    },
    "ENV-METEO": {
        "title": "Meteorology (non-wind)",
        "model_channels": [8 + FP_BAND_IDX.index(i) for i in (5, 6, 8, 14, 17, 18)],
        "description": (
            "Min/max temp, specific humidity, surface precip, forecast temp/humidity. "
            "GRIDMET Total Precipitation (source idx 2) is already absent from the model."
        ),
    },
    "ENV-DROUGHT": {
        "title": "Drought / fire-danger",
        "model_channels": [8 + FP_BAND_IDX.index(i) for i in (7, 12)],
        "description": "ERC + PDSI.",
    },
}


def validate_groups_disjoint() -> dict[str, Any]:
    """Ensure group ablations (excluding ENV-0) partition the 18 FirePred channels."""
    groups = {k: v for k, v in ABLATION_GROUPS.items() if k != "ENV-0"}
    seen: dict[int, str] = {}
    overlaps = []
    for g, spec in groups.items():
        for ch in spec["model_channels"]:
            if ch in seen:
                overlaps.append((ch, seen[ch], g))
            else:
                seen[ch] = g
    expected = set(range(8, 26))
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    return {
        "ok": not overlaps and not missing and not extra,
        "overlaps": overlaps,
        "missing_firepred_channels": missing,
        "extra_non_firepred": extra,
        "n_covered": len(seen),
    }
