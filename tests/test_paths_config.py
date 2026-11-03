"""Lightweight tests for path config resolution (no dataset I/O beyond exists)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.utils.paths import data_root, load_paths


def test_paths_yaml_loads_and_data_root_exists():
    paths = load_paths()
    assert paths["data"]["read_only"] is True
    root = data_root(paths)
    assert root.is_dir()
    assert Path(paths["phase3a"]["freeze_report"]).is_file()
    assert Path(paths["phase4a"]["matched_windows"]).is_file()
    assert Path(paths["phase4b"]["comparison"]).is_file()
