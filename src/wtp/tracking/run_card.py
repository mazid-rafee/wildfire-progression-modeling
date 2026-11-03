"""Write a local run card under runs/ for experiment tracking."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def create_run_dir(runs_root: Path, experiment_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{experiment_id}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_card(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = run_dir / "run_card.json"
    card = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
