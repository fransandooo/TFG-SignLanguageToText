"""Model registry: maps string names to nn.Module constructors."""

from __future__ import annotations

from typing import Any, Callable, Type

import torch.nn as nn

MODEL_REGISTRY: dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Class decorator that registers a model under *name*."""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered model by *name* with the given kwargs."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{name}'. Available: {available}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def ensure_registry_populated() -> None:
    """Import all model modules so that decorators run."""
    import slr.models.gru_attention  # noqa: F401
    import slr.models.gru_baseline   # noqa: F401
    import slr.models.transformer    # noqa: F401
