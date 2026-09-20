"""Informer architecture with distillation for battery sequence modeling."""

from __future__ import annotations

import math
import torch
from torch import nn
from Experiment.Proposed_Model.model import PositionalEncoding


class InformerDistillBlock(nn.Module):
    """Halving temporal sequence length via max-pooling distillation."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model] -> [B, d_model, L]
        x_c = self.conv(x.transpose(1, 2))
        x_c = self.act(self.norm(x_c))
        x_p = self.pool(x_c)
        return x_p.transpose(1, 2)


class InformerModel(nn.Module):
    """Informer baseline with self-attention and temporal distilling layers."""

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 64,
        n_heads: int = 2,
        dropout: float = 0.0,
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=1024)

        # Stage 1: Attention + Distill
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.distill1 = InformerDistillBlock(d_model)

        # Stage 2: Attention + Distill
        self.attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.distill2 = InformerDistillBlock(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 512, 4] -> [B, 512, 64]
        h = self.input_projection(x)
        h = self.pos_enc(h)

        # Stage 1: 512 -> 256
        h_att, _ = self.attn1(h, h, h)
        h = h + h_att
        h = self.distill1(h)

        # Stage 2: 256 -> 128
        h_att2, _ = self.attn2(h, h, h)
        h = h + h_att2
        h = self.distill2(h)

        # Global average pooling
        pooled = h.mean(dim=1)
        return self.head(pooled).squeeze(-1)
