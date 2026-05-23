"""Transformer baseline for landmark-sequence classification."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from slr.models.registry import register_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 65, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@register_model("transformer")
class PoseTransformer(nn.Module):
    """Transformer encoder with CLS token for sequence classification."""

    def __init__(
        self,
        input_size: int = 63,
        num_classes: int = 30,
        dropout: float = 0.4,
        d_model: int = 256,
        nhead: int = 8,
        num_transformer_layers: int = 4,
        max_frames: int = 64,
        **kwargs: Any,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_frames + 1, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.classifier(x[:, 0])
