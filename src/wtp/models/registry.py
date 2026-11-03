"""Model registry for architectures owned by this project.

Frozen baseline checkpoints remain external; register only in-repo builders.
"""

from __future__ import annotations

from typing import Any, Callable

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(f"Model already registered: {name}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def build(name: str, **kwargs: Any) -> Any:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Registered: {sorted(_REGISTRY) or '(none yet)'}"
        )
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
