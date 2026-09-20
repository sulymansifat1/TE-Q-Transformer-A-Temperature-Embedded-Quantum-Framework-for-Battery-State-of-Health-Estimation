"""Temporal Convolutional Network (TCN) baseline model for battery SOH estimation."""

from __future__ import annotations

import torch
from torch import nn


class ChausalDilatedConv1DBlock(nn.Module):
    """Causal dilated conv block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal trim: drop trailing padding
        res = self.residual(x)
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.drop1(self.act1(out))
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.drop2(self.act2(out))
        return out + res


class TCNModel(nn.Module):
    """Deep Temporal Convolutional Network with exponential dilations."""

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 64,
        kernel_size: int = 3,
        num_levels: int = 4,
        dropout: float = 0.0,
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(
                ChausalDilatedConv1DBlock(
                    in_channels=in_ch,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = d_model

        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, 4] -> [B, 4, L]
        x_in = x.transpose(1, 2)
        feat = self.network(x_in)
        pooled = feat[:, :, -1]  # Last causal step
        return self.head(pooled).squeeze(-1)
