"""Pointers to frozen Phase 4A matched manifests (no mutation)."""

from __future__ import annotations

from pathlib import Path

from wtp.utils.paths import load_paths


def matched_windows_csv(paths: dict | None = None) -> Path:
    cfg = paths or load_paths()
    path = Path(cfg["phase4a"]["matched_windows"])
    if not path.is_file():
        raise FileNotFoundError(f"Phase 4A matched windows missing: {path}")
    return path


def temporal_window_manifest(paths: dict | None = None) -> Path:
    cfg = paths or load_paths()
    path = Path(cfg["phase4a"]["temporal_window_manifest"])
    if not path.is_file():
        raise FileNotFoundError(f"Phase 4A temporal window manifest missing: {path}")
    return path
