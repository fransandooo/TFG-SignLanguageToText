"""GRU + Temporal Attention model for landmark-sequence classification."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from slr.models.registry import register_model


class TemporalAttention(nn.Module):
    """Learns a scalar importance weight per time step."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(
        self, rnn_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(rnn_output).squeeze(-1)        # (B, T)
        weights = F.softmax(scores, dim=1)                     # (B, T)
        context = torch.bmm(weights.unsqueeze(1), rnn_output)  # (B, 1, H)
        return context.squeeze(1), weights


@register_model("gru_attention")
class PoseClassifierAttention(nn.Module):
    """Bidirectional GRU with Temporal Attention classifier."""

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
        rnn_hidden = hidden_size * (2 if bidirectional else 1)

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
        self.attention = TemporalAttention(rnn_hidden)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.input_projection(x)
        rnn_out, _ = self.gru(x)
        context, attn_weights = self.attention(rnn_out)
        logits = self.classifier(context)
        if return_attention:
            return logits, attn_weights
        return logits
