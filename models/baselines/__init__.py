"""Baseline benchmark models package for battery SOH estimation.

Exposes:
- CNN1DModel: 1D Temporal Convolutional baseline
- TCNModel: Temporal Convolutional Network with causal dilations
- DLinearSOHModel: LTSF-Linear series decomposition baseline
- TransformerModel: Classical Multi-Head Self-Attention baseline
- PatchTSTSOHModel: Channel-independent patch Transformer baseline
- ITransformerSOHModel: Inverted variate token Transformer baseline
- QLSTMSOHModel: Gate-level Quantum LSTM (PennyLane)
- QNNGRUModel: Quantum Neural Network + GRU (PennyLane)
- LSTMModel: Standard LSTM baseline
- GRUModel: Standard GRU baseline
"""

from __future__ import annotations

from typing import Type
import torch.nn as nn

from models.baselines.cnn1d import CNN1DModel
from models.baselines.tcn import TCNModel, CausalDilatedConv1DBlock, ChausalDilatedConv1DBlock
from models.baselines.dlinear import DLinearSOHModel, DLinearConfig
from models.baselines.transformer import TransformerModel
from models.baselines.patchtst import PatchTSTSOHModel, PatchTSTConfig
from models.baselines.itransformer import ITransformerSOHModel, ITransformerConfig
from models.baselines.qlstm import QLSTMSOHModel, QLSTMConfig
from models.baselines.qnn_gru import QNNGRUModel
from models.baselines.lstm import LSTMModel
from models.baselines.gru import GRUModel

BASELINE_REGISTRY = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "CNN1D": CNN1DModel,
    "TCN": TCNModel,
    "DLinear": DLinearSOHModel,
    "Transformer": TransformerModel,
    "PatchTST": PatchTSTSOHModel,
    "iTransformer": ITransformerSOHModel,
    "QLSTM": QLSTMSOHModel,
    "QNN_GRU": QNNGRUModel,
    "QNN-GRU": QNNGRUModel,
}


def get_baseline_model(name: str, **kwargs) -> nn.Module:
    """Factory to instantiate a baseline model by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{name}'. Available models: {list(BASELINE_REGISTRY.keys())}"
        )
    model_cls = BASELINE_REGISTRY[name]
    return model_cls(**kwargs)


def list_baselines() -> list[str]:
    """Returns the list of canonical baseline model names."""
    return [
        "LSTM",
        "GRU",
        "CNN1D",
        "TCN",
        "DLinear",
        "Transformer",
        "PatchTST",
        "iTransformer",
        "QLSTM",
        "QNN_GRU",
    ]


__all__ = [
    "BASELINE_REGISTRY",
    "get_baseline_model",
    "list_baselines",
    "CNN1DModel",
    "TCNModel",
    "CausalDilatedConv1DBlock",
    "ChausalDilatedConv1DBlock",
    "DLinearSOHModel",
    "DLinearConfig",
    "TransformerModel",
    "PatchTSTSOHModel",
    "PatchTSTConfig",
    "ITransformerSOHModel",
    "ITransformerConfig",
    "QLSTMSOHModel",
    "QLSTMConfig",
    "QNNGRUModel",
    "LSTMModel",
    "GRUModel",
]
