#!/usr/bin/env python3
"""CPU-only sanity check that external DATA_ROOT / baseline paths exist (read-only)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.utils.paths import data_root, load_paths  # noqa: E402


def main() -> int:
    paths = load_paths()
    root = data_root(paths)
    checks = {
        "DATA_ROOT": root,
        "phase3a_freeze": Path(paths["phase3a"]["freeze_report"]),
        "phase3a_ckpt": Path(paths["phase3a"]["checkpoint"]),
        "phase4a_matched": Path(paths["phase4a"]["matched_windows"]),
        "phase4b_comparison": Path(paths["phase4b"]["comparison"]),
        "phase4b_ckpt_ts2": Path(paths["phase4b"]["checkpoint_ts2"]),
        "phase4b_ckpt_ts6": Path(paths["phase4b"]["checkpoint_ts6"]),
        "ts_satfire_repo": Path(paths["external_repos"]["ts_satfire"]),
        "deepfire_repo": Path(paths["external_repos"]["deepfire_forecaster"]),
    }
    ok = True
    for name, path in checks.items():
        exists = path.exists()
        print(f"{'OK' if exists else 'MISSING':7} {name}: {path}")
        ok = ok and exists
    if not paths["data"].get("read_only", False):
        print("WARN: data.read_only is not true in paths.yaml")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
