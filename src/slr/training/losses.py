"""Loss function builders."""

from __future__ import annotations

import torch.nn as nn


def build_criterion(label_smoothing: float = 0.1) -> nn.Module:
    """Return a CrossEntropyLoss with optional label smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
