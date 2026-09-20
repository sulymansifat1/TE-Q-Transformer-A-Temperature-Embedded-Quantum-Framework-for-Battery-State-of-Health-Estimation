"""QGRU baseline (gate-level quantum recurrent) adapted for NASA SOH seq-to-one regression.

Provenance (authoritative paper reference):
- Paper: "A variational approach to quantum gated recurrent units"
- Authors: Andrea Ceschini, Antonello Rosato, Massimo Panella
- Journal: Journal of Physics Communications, Vol. 8, Issue 8, Article 085004 (August 2024)
- DOI: https://doi.org/10.1088/2399-6528/ad6db7
- Publisher URL: https://iopscience.iop.org/article/10.1088/2399-6528/ad6db7
- Open Access: Yes (CC BY 4.0, IOP Publishing). Full text inspected online.

Official implementation status:
- No public author repository released.
- The paper provides exact closed-form architectural equations (Eqs. 8-11), circuit diagram (Fig. 6),
  shared classical layer design (FC_in, FC_out), circular CNOT entanglement, and parameter formulas.
- This file faithfully implements the authoritative architecture from Section 4 of Ceschini et al. (2024).

Identity rule:
- This implementation is a true **gate-level QGRU**: variational quantum circuits reside inside each recurrent gate.
- Classical FC_in and FC_out layers are shared across all 3 gates (reset, update, candidate) exactly as proved in Section 4.1 & 4.3.
- Parameter count strictly matches: n * (3*l + 2*d_hid + d_in + 1) + d_hid.
- It is NOT a "VQC feature layer + classical GRU" model.

Task adaptation:
- Input: [B, 512, 4] (Voltage_V, Current_A, Temperature_C, Time_norm)
- Output: scalar SOH [B] via MLP regression head on the final hidden state h_T.

Runtime note:
- Gate-level quantum recurrence evaluates 3 VQCs x 512 time steps = 1,536 quantum circuit executions per sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pennylane as qml
import torch
from torch import nn


@dataclass(frozen=True)
class QGRUConfig:
    input_dim: int = 4
    seq_len: int = 512
    n_qubits: int = 4
    vqc_depth: int = 1
    hidden_size: int = 16
    dropout: float = 0.0
    q_device: str = "default.qubit"
    diff_method: str = "adjoint"


class _VQC(nn.Module):
    """Variational Quantum Circuit per Ceschini et al. (2024) Figure 6.

    Architecture:
    - Rx rotation data encoding of input angles
    - Ansatz of depth l: parametrized Rx rotation gates followed by circular CNOT entanglement
    - Final measurement: Pauli-Z expectation values on all n qubits.
    """

    def __init__(self, n_qubits: int, depth: int, q_device: str, diff_method: str) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.qnn_device = torch.device("cpu")
        self.weights = nn.Parameter(0.01 * torch.randn(depth, n_qubits, dtype=torch.float32))

        dev = qml.device(q_device, wires=n_qubits)

        def _circular_entanglement() -> None:
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            # Rx data encoding (Ceschini et al. Section 4.1)
            for i in range(n_qubits):
                qml.RX(inputs[:, i], wires=i)
            # Ansatz layers: Rx parametrized rotations + circular CNOTs
            for d in range(weights.shape[0]):
                for i in range(n_qubits):
                    qml.RX(weights[d, i], wires=i)
                _circular_entanglement()
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

        self._circuit = circuit

    def forward(self, x_angles: torch.Tensor) -> torch.Tensor:
        orig_device = x_angles.device
        x_cpu = x_angles.to(self.qnn_device)
        w_cpu = self.weights.to(self.qnn_device)
        out = self._circuit(x_cpu, w_cpu)
        y = torch.stack(out, dim=1)
        return y.to(device=orig_device, dtype=x_angles.dtype)


class _QGRUCell(nn.Module):
    """QGRU Cell implementing Equations (8)-(11) of Ceschini et al. (2024).

    Equations:
      r_t = sigma(FC_out(VQC_reset(FC_in([h_{t-1}, x_t]))))            -- Eq. (8)
      z_t = sigma(FC_out(VQC_update(FC_in([h_{t-1}, x_t]))))           -- Eq. (9)
      h_tilde = tanh(FC_out(VQC_candidate(FC_in([r_t * h_{t-1}, x_t])))) -- Eq. (10)
      h_t = (1 - z_t) * h_tilde + z_t * h_{t-1}                         -- Eq. (11)

    Classical parameter sharing:
      FC_in and FC_out are shared across all gates (Section 4.1 & 4.3).
    """

    def __init__(self, cfg: QGRUConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Shared classical linear layers (Ceschini et al. Section 4.1 & 4.3)
        # FC_in: maps [h_{t-1}, x_t] of dim (hidden_size + input_dim) to n_qubits
        self.fc_in = nn.Linear(cfg.hidden_size + cfg.input_dim, cfg.n_qubits)
        # FC_out: maps n_qubits expectation values to hidden_size
        self.fc_out = nn.Linear(cfg.n_qubits, cfg.hidden_size)

        # 3 separate VQC layers: reset, update, candidate (out)
        self.vqc_reset = _VQC(cfg.n_qubits, cfg.vqc_depth, cfg.q_device, cfg.diff_method)
        self.vqc_update = _VQC(cfg.n_qubits, cfg.vqc_depth, cfg.q_device, cfg.diff_method)
        self.vqc_candidate = _VQC(cfg.n_qubits, cfg.vqc_depth, cfg.q_device, cfg.diff_method)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # Concatenate [h_{t-1}, x_t] per paper Eq. (8) and (9)
        comb_hx = torch.cat([h_prev, x_t], dim=1)

        # Eq. (8): reset gate
        ang_r = self.fc_in(comb_hx)
        r_t = torch.sigmoid(self.fc_out(self.vqc_reset(ang_r)))

        # Eq. (9): update gate
        ang_z = self.fc_in(comb_hx)
        z_t = torch.sigmoid(self.fc_out(self.vqc_update(ang_z)))

        # Eq. (10): candidate hidden state with reset gate elementwise applied to h_{t-1}
        comb_rx = torch.cat([r_t * h_prev, x_t], dim=1)
        ang_c = self.fc_in(comb_rx)
        h_tilde = torch.tanh(self.fc_out(self.vqc_candidate(ang_c)))

        # Eq. (11): final hidden state update
        h_t = (1.0 - z_t) * h_tilde + z_t * h_prev
        return h_t


class QGRUSOHModel(nn.Module):
    """Gate-level QGRU for battery SOH regression: [B, 512, 4] -> [B]."""

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
        self.cfg = QGRUConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            n_qubits=n_qubits,
            vqc_depth=vqc_depth,
            hidden_size=hidden_size,
            dropout=dropout,
            q_device=q_device,
            diff_method=diff_method,
        )
        self.cell = _QGRUCell(self.cfg)
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

        for t in range(self.cfg.seq_len):
            h = self.cell(x[:, t, :], h)
            h = self.dropout(h)

        return self.head(h).squeeze(-1)


__all__ = ["QGRUSOHModel"]
