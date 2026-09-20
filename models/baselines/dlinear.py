"""DLinear baseline (LTSF-Linear family) adapted for battery SOH seq-to-one regression.

Provenance (official upstream):
- Paper: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
- Architecture: series decomposition + linear seasonal/trend heads.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


class _MovingAvg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)  # [B, L + pad, C]
        x_avg = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, C]
        return x_avg


class _SeriesDecomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


@dataclass(frozen=True)
class DLinearConfig:
    seq_len: int = 512
    pred_len: int = 1
    enc_in: int = 4
    individual: bool = False
    kernel_size: int = 25


class _DLinearCore(nn.Module):
    """DLinear forward core: [B, seq_len, C] -> [B, pred_len, C]."""

    def __init__(self, cfg: DLinearConfig) -> None:
        super().__init__()
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.channels = cfg.enc_in
        self.individual = cfg.individual

        self.decomposition = _SeriesDecomp(cfg.kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, C, L]
        trend_init = trend_init.permute(0, 2, 1)        # [B, C, L]

        if self.individual:
            seasonal_output = torch.zeros(
                (seasonal_init.size(0), seasonal_init.size(1), self.pred_len),
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)  # [B, C, pred_len]
            trend_output = self.Linear_Trend(trend_init)          # [B, C, pred_len]

        out = seasonal_output + trend_output  # [B, C, pred_len]
        return out.permute(0, 2, 1)  # [B, pred_len, C]


class DLinearSOHModel(nn.Module):
    """DLinear adapted to SOH regression: [B, 512, 4] -> [B]."""

    def __init__(
        self,
        seq_len: int = 512,
        enc_in: int = 4,
        kernel_size: int = 25,
        individual: bool = False,
        head: str = "linear",
    ) -> None:
        super().__init__()
        cfg = DLinearConfig(
            seq_len=seq_len,
            pred_len=1,
            enc_in=enc_in,
            individual=individual,
            kernel_size=kernel_size,
        )
        self.core = _DLinearCore(cfg)

        if head == "mean":
            self.scalar_head = None
            self.head_mode = "mean"
        elif head == "linear":
            self.scalar_head = nn.Linear(enc_in, 1)
            self.head_mode = "linear"
        else:
            raise ValueError(f"Unknown head='{head}'. Use 'linear' or 'mean'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 512 or x.shape[2] != 4:
            raise ValueError(f"Expected [B, 512, 4], got {tuple(x.shape)}")
        y_seq = self.core(x)              # [B, 1, 4]
        y_vec = y_seq[:, 0, :]            # [B, 4]
        if self.head_mode == "mean":
            y = y_vec.mean(dim=1, keepdim=True)
        else:
            y = self.scalar_head(y_vec)   # [B, 1]
        return y.squeeze(-1)


__all__ = ["DLinearSOHModel", "DLinearConfig"]
