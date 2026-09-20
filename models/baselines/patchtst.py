"""PatchTST baseline adapted for battery SOH seq-to-one regression.

Provenance (official upstream reference):
- Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)
- Design: patching + channel-independence + Transformer encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


class _LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, length: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, length, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class _RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN)."""

    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        subtract_last: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.affine = affine
        self.subtract_last = subtract_last
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.gamma = None
            self.beta = None
        self._last = None
        self._mean = None
        self._stdev = None

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_last:
            self._last = x[:, -1:, :].detach()
            x = x - self._last
        self._mean = x.mean(dim=1, keepdim=True).detach()
        x = x - self._mean
        self._stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x = x / self._stdev
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.beta) / (self.gamma + self.eps)
        x = x * self._stdev + self._mean
        if self.subtract_last:
            x = x + self._last
        return x


@dataclass(frozen=True)
class PatchTSTConfig:
    seq_len: int = 512
    enc_in: int = 4
    patch_len: int = 16
    stride: int = 8
    d_model: int = 64
    n_heads: int = 2
    e_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.0
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False


class PatchTSTSOHModel(nn.Module):
    """PatchTST (channel-independent) adapted to SOH regression: [B, 512, 4] -> [B]."""

    def __init__(
        self,
        seq_len: int = 512,
        enc_in: int = 4,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 2,
        e_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.0,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cfg = PatchTSTConfig(
            seq_len=seq_len,
            enc_in=enc_in,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
        )

        patch_num = int((seq_len - patch_len) / stride + 1)
        self.revin = _RevIN(enc_in, affine=affine, subtract_last=subtract_last) if revin else None

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_enc = _LearnablePositionalEncoding(length=patch_num, d_model=d_model)
        self.dropout = nn.Dropout(dropout)

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

        # Per-channel head (pred_len = 1)
        self.channel_head = nn.Linear(d_model * patch_num, 1)

        # Channel fusion to scalar SOH
        self.scalar_head = nn.Sequential(
            nn.Linear(enc_in, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.cfg.seq_len or x.shape[2] != self.cfg.enc_in:
            raise ValueError(
                f"Expected [B, {self.cfg.seq_len}, {self.cfg.enc_in}], got {tuple(x.shape)}"
            )

        if self.revin is not None:
            x = self.revin.norm(x)

        # [B, L, C] -> [B, C, L] -> patches [B, C, P, PL]
        z = x.permute(0, 2, 1)
        patches = z.unfold(dimension=-1, size=self.cfg.patch_len, step=self.cfg.stride)
        B, C, P, PL = patches.shape
        tokens = patches.reshape(B * C, P, PL)  # channel-independent batch

        h = self.patch_embed(tokens)
        h = self.pos_enc(h)
        h = self.dropout(h)
        h = self.encoder(h)

        h_flat = h.reshape(B * C, -1)
        y_ch = self.channel_head(h_flat).reshape(B, C)  # [B, C]

        y = self.scalar_head(y_ch)  # [B, 1]
        return y.squeeze(-1)


__all__ = ["PatchTSTSOHModel", "PatchTSTConfig"]
