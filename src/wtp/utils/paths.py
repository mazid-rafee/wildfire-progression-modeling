"""Load absolute external paths from configs/paths.yaml (read-only references)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"


def load_paths(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_PATHS_CONFIG
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_root(paths: dict[str, Any] | None = None) -> Path:
    cfg = paths or load_paths()
    root = Path(cfg["data"]["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT not found or not a directory: {root}")
    return root
