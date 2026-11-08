"""Proposed wildfire progression architectures."""

from wtp.models import recency_fusion as recency_fusion  # noqa: F401
from wtp.models import spase_backbone as spase_backbone  # noqa: F401
from wtp.models.recency_fusion import (
    RecencyAwareTemporalFusion,
    RecencyFusionUNet3D,
    backbone_state_dict,
    freeze_temporal_fusion_uniform,
    load_backbone_state_dict,
)
from wtp.models.registry import available, build, register
from wtp.models.spase_backbone import MeanPoolUNet3D, count_params

__all__ = [
    "MeanPoolUNet3D",
    "RecencyFusionUNet3D",
    "RecencyAwareTemporalFusion",
    "freeze_temporal_fusion_uniform",
    "backbone_state_dict",
    "load_backbone_state_dict",
    "count_params",
    "available",
    "build",
    "register",
]
