"""Plain GRU baseline (no attention) for comparison."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from slr.models.registry import register_model


@register_model("gru_baseline")
class PoseClassifierGRU(nn.Module):
    """Bidirectional GRU with last-hidden-state classification."""

    def __init__(
        self,
        input_size: int = 63,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 30,
        dropout: float = 0.4,
        bidirectional: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        rnn_hidden = hidden_size * self.num_directions

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        rnn_out, _ = self.gru(x)
        last_hidden = rnn_out[:, -1, :]
        return self.classifier(last_hidden)
