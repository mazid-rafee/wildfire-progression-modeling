"""Immutable V2 protocol constants (matched 27-channel layout preserved)."""

from __future__ import annotations

PROTOCOL_VERSION = "V2.0-corrected-protocol"

CROP_SIZE = 256
N_CHANNELS = 27
TS_LENGTH_DEFAULT = 2

# Same keep-list as V1 / CFG.FP_BAND_IDX (skips source index 2 = Total Precipitation)
FP_BAND_IDX: list[int] = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# FirePred slot within the 18 kept bands → model channel = 8 + slot
FP_SLOT_WIND_SPEED = FP_BAND_IDX.index(3)  # 2 → model 10
FP_SLOT_WIND_DIR = FP_BAND_IDX.index(4)  # 3 → model 11
FP_SLOT_ASPECT = FP_BAND_IDX.index(10)  # 9 → model 17
FP_SLOT_LANDCOVER = FP_BAND_IDX.index(13)  # 12 → model 20
FP_SLOT_FCST_WIND_SPEED = FP_BAND_IDX.index(15)  # 14 → model 22
FP_SLOT_FCST_WIND_DIR = FP_BAND_IDX.index(16)  # 15 → model 23

LANDCOVER_FP_SLOT = FP_SLOT_LANDCOVER

# Model-channel indices for angle-valued FirePred bands (post VIIRS||FP||BA concat)
DIRECTIONAL_MODEL_CHANNELS: dict[int, dict] = {
    11: {
        "name": "wind_direction_gridmet",
        "fp_slot": FP_SLOT_WIND_DIR,
        "convention": "meteorological_from_deg",
        "observed_range": [0.0, 360.0],
        "wrap": 360.0,
    },
    17: {
        "name": "aspect",
        "fp_slot": FP_SLOT_ASPECT,
        "convention": "aspect_facing_deg",
        "observed_range": [0.0, 360.0],
        "wrap": 360.0,
    },
    23: {
        "name": "forecast_wind_direction_gfs",
        "fp_slot": FP_SLOT_FCST_WIND_DIR,
        "convention": "meteorological_from_deg_gfs_signed",
        "observed_range": [-90.0, 90.0],
        "wrap": 360.0,
        "notes": (
            "GFS forecast dir rasters observed ≈[-90, 90], not GRIDMET [0,360]. "
            "Geometric remapping still applied via unit-vector transform then atan2."
        ),
    },
}

# Continuous vs categorical FirePred slots for resampling
CATEGORICAL_FP_SLOTS: set[int] = {LANDCOVER_FP_SLOT}
# All other FP slots treated as continuous (bilinear)

SAT_MEAN = [
    18.76488,
    27.441864,
    20.584806,
    305.99478,
    294.31738,
    14.625097,
    276.4207,
    275.16766,
]
SAT_STD = [
    15.911591,
    14.879259,
    10.832616,
    21.761852,
    24.703484,
    9.878246,
    40.64329,
    40.7657,
]

EXPECTED_COUNTS = {
    "train": 983,
    "val": 135,
    "test24": 561,
    "test18": 553,
    "test16": 531,
}

# Audit-confirmed train fires with VIIRS/FirePred grid mismatch under V1
GEO_MISMATCH_TRAIN_FIRES = ("20778186", "20899673", "20899813", "21038333")
