#!/usr/bin/env python3
"""Confirm the training gate blocks accidental launches (no GPU work)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wtp.train.gate import TrainingNotEnabledError, require_train_approval  # noqa: E402
from wtp.train.loop import run_training  # noqa: E402


def main() -> int:
    try:
        require_train_approval(enabled=False, i_understand_full_train=False)
    except TrainingNotEnabledError:
        pass
    else:
        print("FAIL: gate did not block enabled=False")
        return 1

    try:
        run_training({"train": {"enabled": False}})
    except TrainingNotEnabledError:
        print("OK: training gate blocks scaffold launches")
        return 0
    except NotImplementedError:
        print("FAIL: gate bypassed into NotImplementedError")
        return 1
    print("FAIL: training unexpectedly returned")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
