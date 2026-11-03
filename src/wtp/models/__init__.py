"""Proposed wildfire progression architectures."""

from wtp.models import recency_fusion as recency_fusion  # noqa: F401
from wtp.models import spase_backbone as spase_backbone  # noqa: F401
from wtp.models.recency_fusion import RecencyAwareTemporalFusion, RecencyFusionUNet3D
from wtp.models.registry import available, build, register
from wtp.models.spase_backbone import MeanPoolUNet3D, count_params

__all__ = [
    "MeanPoolUNet3D",
    "RecencyFusionUNet3D",
    "RecencyAwareTemporalFusion",
    "count_params",
    "available",
    "build",
    "register",
]
