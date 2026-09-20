"""1D Temporal Convolutional baseline model for battery SOH estimation."""

from __future__ import annotations

import torch
from torch import nn


class CNN1DModel(nn.Module):
    """1D CNN sequence model with multi-scale temporal convolutions and global pooling."""

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 64,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.0,
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        layers = []
        in_ch = input_dim
        for _ in range(num_layers):
            out_ch = d_model
            layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.conv_net = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, 4] -> permute to [B, 4, L]
        x_conv = x.transpose(1, 2)
        features = self.conv_net(x_conv)  # [B, d_model, L]
        pooled = features.mean(dim=-1)     # Global average pooling -> [B, d_model]
        return self.head(pooled).squeeze(-1)


__all__ = ["CNN1DModel"]
