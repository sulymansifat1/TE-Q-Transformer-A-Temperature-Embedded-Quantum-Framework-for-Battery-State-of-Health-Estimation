"""Standard GRU baseline model for battery SOH estimation."""

from __future__ import annotations

import torch
from torch import nn


class GRUModel(nn.Module):
    """GRU sequence model with input projection and regression head."""

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        head_hidden_dim: int = 64,
        use_projection: bool = True,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.input_projection = nn.Linear(input_dim, d_model) if use_projection else nn.Identity()
        gru_in = d_model if use_projection else input_dim
        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, 4]
        x = self.input_projection(x)
        outputs, _ = self.gru(x)
        if self.pooling == "mean":
            pooled = outputs.mean(dim=1)
        else:
            pooled = outputs[:, -1, :]
        return self.head(pooled).squeeze(-1)


__all__ = ["GRUModel"]
