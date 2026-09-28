"""Classical Transformer Encoder baseline for battery SOH estimation."""

from __future__ import annotations

import torch
from torch import nn
from models.proposed.te_q_transformer import PositionalEncoding


class TransformerModel(nn.Module):
    """Pure classical Transformer encoder baseline without quantum embedding."""

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 3,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        head_hidden_dim: int = 64,
        use_cls_token: bool = True,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        self.use_cls_token = use_cls_token
        self.pooling = pooling
        self.input_projection = nn.Linear(input_dim, d_model)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=1024)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, 4]
        h = self.input_projection(x)
        if self.use_cls_token and self.cls_token is not None:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            h = torch.cat([cls, h], dim=1)
        h = self.positional_encoding(h)
        encoded = self.encoder(h)
        if self.use_cls_token and self.pooling == "cls":
            pooled = encoded[:, 0, :]
        else:
            pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)


__all__ = ["TransformerModel"]
