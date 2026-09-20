"""iTransformer baseline adapted for battery SOH seq-to-one regression.

Provenance (official upstream reference):
- Paper: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024 Spotlight)
- Design: Inverted tokenization (variates as tokens) + Transformer encoder over variates.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class ITransformerConfig:
    seq_len: int = 512
    enc_in: int = 4
    d_model: int = 64
    n_heads: int = 2
    e_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.0
    use_norm: bool = True
    pooling: str = "mean"


class _DataEmbeddingInverted(nn.Module):
    """Inverts dimensions and projects time dimension into d_model per variate token."""

    def __init__(self, c_in: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N] -> [B, N, L] -> [B, N, d_model]
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        return self.dropout(x)


class ITransformerSOHModel(nn.Module):
    """iTransformer-style inverted Transformer encoder for SOH regression: [B, 512, 4] -> [B]."""

    def __init__(
        self,
        seq_len: int = 512,
        enc_in: int = 4,
        d_model: int = 64,
        n_heads: int = 2,
        e_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.0,
        use_norm: bool = True,
        pooling: str = "mean",
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cfg = ITransformerConfig(
            seq_len=seq_len,
            enc_in=enc_in,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_norm=use_norm,
            pooling=pooling,
        )

        self.enc_embedding = _DataEmbeddingInverted(
            c_in=seq_len, d_model=d_model, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.cfg.seq_len or x.shape[2] != self.cfg.enc_in:
            raise ValueError(
                f"Expected [B, {self.cfg.seq_len}, {self.cfg.enc_in}], got {tuple(x.shape)}"
            )

        if self.cfg.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x0 = x - means
            stdev = torch.sqrt(torch.var(x0, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x0 = x0 / stdev
        else:
            x0 = x

        enc_in = self.enc_embedding(x0)  # [B, N, d_model]
        enc_out = self.encoder(enc_in)   # [B, N, d_model]

        if self.cfg.pooling == "mean":
            pooled = enc_out.mean(dim=1)
        elif self.cfg.pooling == "cls":
            pooled = enc_out[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling='{self.cfg.pooling}'.")

        y = self.head(pooled)
        return y.squeeze(-1)


__all__ = ["ITransformerSOHModel", "ITransformerConfig"]
