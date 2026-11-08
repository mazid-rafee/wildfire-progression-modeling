"""V2 scientifically corrected preprocessing protocol.

Separate from frozen V1 Phase 3A / 4B / E1 loaders. Does not overwrite
DeepFire helpers or mutate DATA_ROOT GeoTIFFs.
"""

from wtp.data.v2.constants import (
    CROP_SIZE,
    DIRECTIONAL_MODEL_CHANNELS,
    FP_BAND_IDX,
    LANDCOVER_FP_SLOT,
    PROTOCOL_VERSION,
)
from wtp.data.v2.directional_aug import apply_geometric_aug
from wtp.data.v2.geo_align import align_firepred_to_viirs, read_georef
from wtp.data.v2.normalize import compute_matched_train_fp_stats, normalize_firepred
from wtp.data.v2.windows import build_window_tensor_v2, label_crop_v2

__all__ = [
    "PROTOCOL_VERSION",
    "CROP_SIZE",
    "FP_BAND_IDX",
    "DIRECTIONAL_MODEL_CHANNELS",
    "LANDCOVER_FP_SLOT",
    "align_firepred_to_viirs",
    "read_georef",
    "apply_geometric_aug",
    "compute_matched_train_fp_stats",
    "normalize_firepred",
    "build_window_tensor_v2",
    "label_crop_v2",
]
