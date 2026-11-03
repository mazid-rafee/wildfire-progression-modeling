"""Training loop stub."""

from __future__ import annotations

from wtp.train.gate import require_train_approval


def run_training(config: dict, *, i_understand_full_train: bool = False) -> None:
    require_train_approval(
        enabled=bool(config.get("train", {}).get("enabled", False)),
        i_understand_full_train=i_understand_full_train,
    )
    raise NotImplementedError("Training loop not implemented yet.")
