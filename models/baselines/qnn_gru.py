"""Quantum Neural Network combined with GRU (QNN-GRU) baseline."""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn


def _build_qnn_layer(
    n_qubits: int = 4, n_q_layers: int = 1, q_device: str = "default.qubit"
) -> qml.qnn.TorchLayer:
    """Constructs PennyLane variational quantum circuit layer."""
    dev = qml.device(q_device, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[:, i], wires=i)
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_q_layers, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QNNGRUModel(nn.Module):
    """Hybrid Quantum Neural Network + GRU sequence model."""

    def __init__(
        self,
        input_dim: int = 4,
        n_qubits: int = 4,
        n_q_layers: int = 1,
        d_model: int = 64,
        num_gru_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_dim: int = 64,
        q_device: str = "default.qubit",
        use_projection: bool = True,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.qnn_device = torch.device("cpu")
        self.input_projection = (
            nn.Linear(input_dim, n_qubits) if use_projection else nn.Identity()
        )
        self.qnn_layer = _build_qnn_layer(
            n_qubits=n_qubits, n_q_layers=n_q_layers, q_device=q_device
        )
        self.qnn_layer.to(self.qnn_device)
        self.qnn_out_projection = nn.Linear(n_qubits, d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_device = x.device
        b, l, _ = x.shape
        x = self.input_projection(x)
        # Quantum circuit evaluates on CPU
        x_cpu = x.reshape(b * l, -1).to(self.qnn_device)
        q_feat = self.qnn_layer(x_cpu)
        q_feat = q_feat.to(orig_device).reshape(b, l, -1)
        h = self.qnn_out_projection(q_feat)
        outputs, _ = self.gru(h)
        if self.pooling == "mean":
            pooled = outputs.mean(dim=1)
        else:
            pooled = outputs[:, -1, :]
        return self.head(pooled).squeeze(-1)


__all__ = ["QNNGRUModel"]
