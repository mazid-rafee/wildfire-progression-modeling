"""Hard gate preventing accidental GPU training from the scaffold."""

from __future__ import annotations


class TrainingNotEnabledError(RuntimeError):
    pass


def require_train_approval(*, enabled: bool, i_understand_full_train: bool) -> None:
    if not enabled or not i_understand_full_train:
        raise TrainingNotEnabledError(
            "Training is disabled. Set train.enabled in the config and pass "
            "--i-understand-full-train only after explicit Phase 4C launch approval. "
            "No experiments were started during project setup."
        )
