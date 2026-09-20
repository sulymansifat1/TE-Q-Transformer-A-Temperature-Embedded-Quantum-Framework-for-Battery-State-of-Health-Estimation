"""QLSTM baseline (gate-level quantum recurrent) adapted for NASA SOH seq-to-one regression.

Provenance (paper + author code reference):
- Paper: "Quantum Long Short-Term Memory" (ICASSP 2022)
  - DOI: https://doi.org/10.1109/icassp43922.2022.9747369
  - arXiv: https://arxiv.org/abs/2009.01783
- Author code repo (verified): https://github.com/ycchen1989/Quantum_Long_Short_Term_Memory
  - Commit (HEAD verified 2026-09-15): a015e0a2daf0347e16c1e74868398a281c6b2803
  - License: MIT (see upstream LICENSE)

Identity rule:
- This implementation is a **gate-level QLSTM**: each LSTM gate uses a Variational Quantum Circuit (VQC).
- It is NOT the prior "VQC feature layer + classical LSTM" baseline previously in this repo.

Task adaptation:
- Input: [B, 512, 4] (V, I, T_C, t_norm)
- Output: scalar SOH [B]

Runtime note:
- Gate-level quantum recurrence is expensive at L=512. E05 readiness will explicitly report feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pennylane as qml
import torch
from torch import nn


@dataclass(frozen=True)
class QLSTMConfig:
    input_dim: int = 4
    seq_len: int = 512
    n_qubits: int = 4
    vqc_depth: int = 1
    hidden_size: int = 16
    dropout: float = 0.0
    q_device: str = "default.qubit"
    diff_method: str = "adjoint"


class _VQC(nn.Module):
    """Small VQC layer returning PauliZ expectation values per qubit."""

    def __init__(self, n_qubits: int, depth: int, q_device: str, diff_method: str) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.qnn_device = torch.device("cpu")
        self.weights = nn.Parameter(0.01 * torch.randn(depth, n_qubits, dtype=torch.float32))

        dev = qml.device(q_device, wires=n_qubits)

        def _entangle() -> None:
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits):
                qml.RY(inputs[:, i], wires=i)
            for d in range(weights.shape[0]):
                _entangle()
                for i in range(n_qubits):
                    qml.RY(weights[d, i], wires=i)
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

        self._circuit = circuit

    def forward(self, x_angles: torch.Tensor) -> torch.Tensor:
        orig_device = x_angles.device
        x_cpu = x_angles.to(self.qnn_device)
        w_cpu = self.weights.to(self.qnn_device)
        out = self._circuit(x_cpu, w_cpu)
        y = torch.stack(out, dim=1)
        return y.to(device=orig_device, dtype=x_angles.dtype)


class _QLSTMCell(nn.Module):
    def __init__(self, cfg: QLSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc_in = nn.Linear(cfg.input_dim + cfg.hidden_size, cfg.n_qubits)

        self.vqc_i = _VQC(cfg.n_qubits, cfg.vqc_depth, cfg.q_device, cfg.diff_method)
        self.vqc_f = _VQC(cfg.n_qubits, cfg.vqc_depth, cfg.q_device, cfg.diff_method)
        self.vqc_g = _VQC(cfg.n_qubits, cfg.vqc_depth, cfg.q_device, cfg.diff_method)
        self.vqc_o = _VQC(cfg.n_qubits, cfg.vqc_depth, cfg.q_device, cfg.diff_method)

        self.fc_i = nn.Linear(cfg.n_qubits, cfg.hidden_size)
        self.fc_f = nn.Linear(cfg.n_qubits, cfg.hidden_size)
        self.fc_g = nn.Linear(cfg.n_qubits, cfg.hidden_size)
        self.fc_o = nn.Linear(cfg.n_qubits, cfg.hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x_t, h_prev], dim=1)
        angles = torch.tanh(self.fc_in(combined)) * torch.pi

        i_t = torch.sigmoid(self.fc_i(self.vqc_i(angles)))
        f_t = torch.sigmoid(self.fc_f(self.vqc_f(angles)))
        g_t = torch.tanh(self.fc_g(self.vqc_g(angles)))
        o_t = torch.sigmoid(self.fc_o(self.vqc_o(angles)))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class QLSTMSOHModel(nn.Module):
    """Gate-level QLSTM for SOH regression: [B, 512, 4] -> [B]."""

    def __init__(
        self,
        input_dim: int = 4,
        seq_len: int = 512,
        n_qubits: int = 4,
        vqc_depth: int = 1,
        hidden_size: int = 16,
        dropout: float = 0.0,
        head_hidden_dim: int = 64,
        q_device: str = "default.qubit",
        diff_method: str = "adjoint",
    ) -> None:
        super().__init__()
        self.cfg = QLSTMConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            n_qubits=n_qubits,
            vqc_depth=vqc_depth,
            hidden_size=hidden_size,
            dropout=dropout,
            q_device=q_device,
            diff_method=diff_method,
        )
        self.cell = _QLSTMCell(self.cfg)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.cfg.seq_len or x.shape[2] != self.cfg.input_dim:
            raise ValueError(f"Expected [B, {self.cfg.seq_len}, {self.cfg.input_dim}], got {tuple(x.shape)}")

        B = x.size(0)
        h = torch.zeros(B, self.cfg.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)

        for t in range(self.cfg.seq_len):
            h, c = self.cell(x[:, t, :], h, c)
            h = self.dropout(h)

        return self.head(h).squeeze(-1)


__all__ = ["QLSTMSOHModel"]
