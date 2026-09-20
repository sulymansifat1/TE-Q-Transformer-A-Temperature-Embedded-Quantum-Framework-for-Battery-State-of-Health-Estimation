"""Baseline benchmark models for battery SOH estimation (lazy loaded).

E05 LOCKED ACTIVE SET (2026 revision):
  Classical (8): LSTM, GRU, CNN1D, TCN, DLinear, Transformer, PatchTST, iTransformer
  Quantum (2):   QLSTM, QGRU

Notes:
- Informer is intentionally excluded from the locked E05 set.
- BiLSTM is intentionally excluded from the locked E05 set to reduce redundancy.
"""

from __future__ import annotations
from typing import Type
import torch.nn as nn

BASELINE_NAMES = [
    "LSTM",
    "GRU",
    "CNN1D",
    "TCN",
    "DLinear",
    "Transformer",
    "PatchTST",
    "iTransformer",
    "QLSTM",
    "QGRU",
]


def get_baseline_model_class(name: str) -> Type[nn.Module]:
    """Dynamically loads and returns the requested baseline model class."""
    if name == "LSTM":
        from Experiment.Baseline.LSTM import LSTMModel
        return LSTMModel
    elif name == "GRU":
        from Experiment.Baseline.GRU import GRUModel
        return GRUModel
    elif name == "CNN1D":
        from Experiment.Baseline.CNN1D import CNN1DModel
        return CNN1DModel
    elif name == "TCN":
        from Experiment.Baseline.TCN import TCNModel
        return TCNModel
    elif name == "DLinear":
        from Experiment.Baseline.DLinear import DLinearSOHModel
        return DLinearSOHModel
    elif name == "Transformer":
        from Experiment.Baseline.Transformer import TransformerModel
        return TransformerModel
    elif name == "PatchTST":
        from Experiment.Baseline.PatchTST import PatchTSTSOHModel
        return PatchTSTSOHModel
    elif name == "iTransformer":
        from Experiment.Baseline.iTransformer import ITransformerSOHModel
        return ITransformerSOHModel
    elif name == "QLSTM":
        from Experiment.Baseline.QLSTM import QLSTMSOHModel
        return QLSTMSOHModel
    elif name == "QGRU":
        from Experiment.Baseline.QGRU import QGRUSOHModel
        return QGRUSOHModel
    else:
        raise ValueError(f"Unknown baseline '{name}'. Available: {BASELINE_NAMES}")


__all__ = ["BASELINE_NAMES", "get_baseline_model_class"]

