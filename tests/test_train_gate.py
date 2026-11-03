"""Training must remain gated in the scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.train.gate import TrainingNotEnabledError
from wtp.train.loop import run_training


def test_run_training_blocked_by_default():
    with pytest.raises(TrainingNotEnabledError):
        run_training({"train": {"enabled": False}})
