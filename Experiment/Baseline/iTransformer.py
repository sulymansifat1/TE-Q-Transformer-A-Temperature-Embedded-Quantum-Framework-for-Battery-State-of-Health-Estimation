"""iTransformer baseline adapted for NASA SOH seq-to-one regression.

Provenance (official upstream reference):
- Paper: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024 Spotlight)
  - Paper PDF: https://proceedings.iclr.cc/paper_files/paper/2024/file/2ea18fdc667e0ef2ad82b2b4d65147ad-Paper-Conference.pdf
- Official repo: https://github.com/thuml/iTransformer
- Upstream commit (HEAD verified 2026-09-15): c2426e68ca13f74aaec08045c5c724d8ad328124
- License: MIT (per upstream repo)

Core architecture preserved? YES:
- **Inverted tokenization**: variates are tokens (N tokens), time points are token features.
- **Encoder-only Transformer**: native Transformer modules operate over variate tokens.

Task adaptation (minimal):
- Forecasting head replaced with a **seq-to-one regression head** for SOH.
- No timestamp covariates (`x_mark`) are used in this project; we follow upstream behavior for `x_mark=None`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ITransformerConfig:
    seq_len: int = 512
    enc_in: int = 4          # number of variates/tokens
    d_model: int = 64
    n_heads: int = 2
    e_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.0
    use_norm: bool = True    # upstream-style per-sample normalization
    pooling: str = "mean"    # token pooling over variates


class _DataEmbeddingInverted(nn.Module):
    """Upstream DataEmbedding_inverted (simplified): linear map Time->d_model per variate token."""

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

        # Embedding: invert and linearly embed per variate token
        self.enc_embedding = _DataEmbeddingInverted(c_in=seq_len, d_model=d_model, dropout=dropout)

        # Encoder-only Transformer over variate tokens (token length = enc_in = 4)
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
        # x: [B, L, N] where N=4
        if x.ndim != 3 or x.shape[1] != self.cfg.seq_len or x.shape[2] != self.cfg.enc_in:
            raise ValueError(f"Expected [B, {self.cfg.seq_len}, {self.cfg.enc_in}], got {tuple(x.shape)}")

        if self.cfg.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x0 = x - means
            stdev = torch.sqrt(torch.var(x0, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x0 = x0 / stdev
        else:
            x0 = x

        # Embed and encode over variate tokens
        # embedding expects [B, L, N] but internally inverts to [B, N, L]
        enc_in = self.enc_embedding(x0)          # [B, N, d_model]
        enc_out = self.encoder(enc_in)           # [B, N, d_model]

        # Pool over tokens (variates)
        if self.cfg.pooling == "mean":
            pooled = enc_out.mean(dim=1)
        elif self.cfg.pooling == "cls":
            # optional: treat the first variate token as a representative token
            pooled = enc_out[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling='{self.cfg.pooling}'.")

        y = self.head(pooled)  # [B, 1]
        return y.squeeze(-1)


__all__ = ["ITransformerSOHModel"]

