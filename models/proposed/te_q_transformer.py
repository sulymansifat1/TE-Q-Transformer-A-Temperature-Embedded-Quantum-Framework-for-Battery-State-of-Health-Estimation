"""TE-Q-Transformer: A Temperature-Embedded Quantum Framework for Battery State-of-Health Estimation.

This module provides the authoritative implementation of the proposed hybrid architecture:
1. Physics-guided Arrhenius temperature encoding (SEI growth + Lithium plating mechanisms)
   with trainable empirical temperature parameters.
2. Four-qubit simulated quantum feature mapping with parameterized rich entangling circuit
   and Pauli-Z expectation value readout.
3. Latent linear projection into transformer embedding space.
4. Conv1D temporal smoothing layer.
5. Learnable CLS token and sinusoidal positional encoding.
6. Multi-layer Transformer encoder backbone with GELU activations.
7. Multi-layer perceptron regression head outputting cycle-level SOH.

Model parameter count: exactly 92,554 trainable parameters under default rich_entangler_config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pennylane as qml
import torch
from torch import nn


@dataclass(frozen=True)
class TEQTransformerConfig:
    """Architectural configuration for TE-Q-Transformer."""
    input_dim: int = 4
    seq_len: int = 512
    quantum_dim: int = 4
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 3
    dim_feedforward: int = 64
    dropout: float = 0.0
    q_device: str = "default.qubit"
    entangler_layers: int = 1
    entangler_type: str = "rich"  # "rich" is the authoritative proposed setting
    use_pauli_feature_map: bool = False
    feature_map_reps: int = 1
    feature_map_entangle: bool = True
    use_cls_token: bool = True
    pooling: str = "cls"
    use_positional_encoding: bool = True
    use_temporal_smooth: bool = True
    temporal_kernel_size: int = 3
    use_gru_smoother: bool = False
    gru_num_layers: int = 1
    gru_dropout: float = 0.0
    head_hidden_dim: int = 64
    use_residual_mlp: bool = False
    residual_mlp_dim: int = 128


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for temporal sequence representation."""

    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class QuantumEmbeddingLayer(nn.Module):
    """Hybrid Physics-Guided Temperature Encoding and Parameterized Quantum Feature Map.

    Integrates:
    1. Physics-guided Arrhenius kinetics:
       - High-temperature SEI growth: exp((Ea_sei / R) * (1/T_ref - 1/T))
       - Low-temperature Lithium plating: exp((Ea_pl / R) * (1/T - 1/T_ref))
       where Ea_sei and Ea_pl are trainable empirical parameters.
    2. Four-qubit PennyLane simulated variational quantum circuit with rich entangler:
       RZ rotations, CNOT ladder, IsingZZ couplings, multi-qubit Toffoli gates, and Hadamard layers.
    3. Pauli-Z expectation value readout on all 4 qubits.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        q_device: str = "default.qubit",
        entangler_layers: int = 1,
        entangler_type: str = "rich",
        use_pauli_feature_map: bool = False,
        feature_map_reps: int = 1,
        feature_map_entangle: bool = True,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.R = 8.314462618  # Universal gas constant, J/(mol*K)
        self.T_ref = 298.15   # Reference temperature (25 °C), Kelvin
        self.entangler_type = entangler_type
        self.use_pauli_feature_map = use_pauli_feature_map
        self.feature_map_reps = feature_map_reps
        self.feature_map_entangle = feature_map_entangle

        # Trainable empirical activation energy parameters initialized at 3.0 (representing 30 kJ/mol)
        self.Ea_sei = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        self.Ea_pl = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))

        # Variational entangler circuit parameters
        self.entangler_weights = nn.Parameter(
            0.01 * torch.randn(entangler_layers, n_qubits, dtype=torch.float32)
        )
        self.entangler_rzz = nn.Parameter(
            0.01 * torch.randn(entangler_layers, max(1, n_qubits - 1), dtype=torch.float32)
        )

        # Classical simulation quantum device
        dev = qml.device(q_device, wires=n_qubits)

        def _apply_pauli_feature_map(inputs: torch.Tensor) -> None:
            for _ in range(self.feature_map_reps):
                for i in range(n_qubits):
                    qml.RX(inputs[:, i], wires=i)
                    qml.RY(inputs[:, i], wires=i)
                if self.feature_map_entangle:
                    for i in range(n_qubits - 1):
                        qml.CZ(wires=[i, i + 1])

        def _apply_basic_entangler(entangler_weights: torch.Tensor) -> None:
            qml.BasicEntanglerLayers(entangler_weights, wires=range(n_qubits))

        def _apply_rich_entangler(
            entangler_weights: torch.Tensor, entangler_rzz: torch.Tensor
        ) -> None:
            for layer in range(entangler_weights.shape[0]):
                for q in range(n_qubits):
                    qml.RZ(entangler_weights[layer, q], wires=q)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(n_qubits - 1):
                    qml.IsingZZ(entangler_rzz[layer, q], wires=[q, q + 1])
                if n_qubits >= 3:
                    qml.Toffoli(wires=[0, 1, 2])
                    if n_qubits >= 4:
                        qml.Toffoli(wires=[1, 2, 3])
                for q in range(n_qubits):
                    qml.Hadamard(wires=q)
                    qml.RZ(entangler_weights[layer, q], wires=q)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(
            inputs: torch.Tensor,
            entangler_weights: torch.Tensor,
            entangler_rzz: torch.Tensor,
        ):
            if self.use_pauli_feature_map:
                _apply_pauli_feature_map(inputs)
            qml.RY(inputs[:, 0], wires=0)
            qml.RY(inputs[:, 1], wires=1)
            qml.RY(inputs[:, 2], wires=2)
            qml.RY(inputs[:, 3], wires=3)
            if self.entangler_type == "rich":
                _apply_rich_entangler(entangler_weights, entangler_rzz)
            else:
                _apply_basic_entangler(entangler_weights)
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms batch of flat [B_flat, 4] features through physics encoding and quantum circuit."""
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError(f"Expected [B_flat, 4], got {tuple(x.shape)}")

        out_device = x.device
        out_dtype = x.dtype

        # 1. Classical feature angles (channels 0, 1, 3: Voltage, Current, Normalized Time)
        voltage_angle = x[:, 0] * torch.pi
        current_angle = x[:, 1] * torch.pi
        time_angle = x[:, 3] * torch.pi

        # 2. Physics-guided Arrhenius temperature encoding (channel 2: Temperature in Celsius)
        temp_c = x[:, 2]
        temp_k = torch.clamp(temp_c + 273.15, min=1.0)
        inv_t = 1.0 / temp_k
        inv_t_ref = 1.0 / self.T_ref
        Ea_sei_actual = self.Ea_sei * 10000.0
        Ea_pl_actual = self.Ea_pl * 10000.0

        sei_term = torch.exp((Ea_sei_actual / self.R) * (inv_t_ref - inv_t))
        plating_term = torch.exp((Ea_pl_actual / self.R) * (inv_t - inv_t_ref))
        phi = sei_term + plating_term
        theta_temp = torch.pi * phi / 4.0

        # Stack into 4-channel quantum state rotation angles
        angles = torch.stack([voltage_angle, current_angle, time_angle, theta_temp], dim=1)

        # Evaluate quantum circuit
        angles_cpu = angles.to("cpu")
        entangler_cpu = self.entangler_weights.to("cpu")
        entangler_rzz_cpu = self.entangler_rzz.to("cpu")
        q_out = self.circuit(angles_cpu, entangler_cpu, entangler_rzz_cpu)
        q_tensor = torch.stack(q_out, dim=1).to(device=out_device, dtype=out_dtype)
        return q_tensor


class TEQTransformer(nn.Module):
    """TE-Q-Transformer model for battery sequence-to-one SOH estimation."""

    def __init__(self, cfg: Optional[TEQTransformerConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or TEQTransformerConfig()
        if self.cfg.input_dim != 4:
            raise ValueError("This model expects exactly 4 input features.")
        if self.cfg.quantum_dim != 4:
            raise ValueError("quantum_dim must be 4 to match the 4 input channels.")
        if self.cfg.d_model % self.cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.cfg.d_model <= self.cfg.quantum_dim:
            raise ValueError("d_model should be larger than quantum_dim after projection.")
        if self.cfg.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'.")
        if self.cfg.use_cls_token and self.cfg.pooling != "cls":
            raise ValueError("use_cls_token=True requires pooling='cls'.")
        if not self.cfg.use_cls_token and self.cfg.pooling != "mean":
            raise ValueError("use_cls_token=False requires pooling='mean'.")
        if self.cfg.use_temporal_smooth and self.cfg.temporal_kernel_size % 2 == 0:
            raise ValueError(
                "temporal_kernel_size should be odd so the sequence length is preserved."
            )
        if self.cfg.use_gru_smoother and self.cfg.use_temporal_smooth:
            raise ValueError("Enable only one temporal module: GRU smoother or Conv1D smoother.")
        if self.cfg.entangler_type not in {"basic", "rich"}:
            raise ValueError("entangler_type must be 'basic' or 'rich'.")
        if self.cfg.head_hidden_dim <= 0:
            raise ValueError("head_hidden_dim must be positive.")

        # 1. Quantum feature embedding layer
        self.quantum_embed = QuantumEmbeddingLayer(
            n_qubits=self.cfg.quantum_dim,
            q_device=self.cfg.q_device,
            entangler_layers=self.cfg.entangler_layers,
            entangler_type=self.cfg.entangler_type,
            use_pauli_feature_map=self.cfg.use_pauli_feature_map,
            feature_map_reps=self.cfg.feature_map_reps,
            feature_map_entangle=self.cfg.feature_map_entangle,
        )

        # 2. Linear projection into d_model dimensional latent space
        self.quantum_proj = nn.Linear(self.cfg.quantum_dim, self.cfg.d_model)

        # 3. Learnable classification token
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, self.cfg.d_model))
            if self.cfg.use_cls_token
            else None
        )

        # 4. Positional encoding
        self.pos_encoder = (
            PositionalEncoding(self.cfg.d_model, max_len=self.cfg.seq_len + 2)
            if self.cfg.use_positional_encoding
            else None
        )

        # 5. Temporal smoothing layer
        if self.cfg.use_temporal_smooth:
            self.temporal_smooth = nn.Conv1d(
                in_channels=self.cfg.d_model,
                out_channels=self.cfg.d_model,
                kernel_size=self.cfg.temporal_kernel_size,
                padding=self.cfg.temporal_kernel_size // 2,
                bias=False,
            )
        else:
            self.temporal_smooth = nn.Identity()

        if self.cfg.use_gru_smoother:
            self.gru_smoother = nn.GRU(
                input_size=self.cfg.d_model,
                hidden_size=self.cfg.d_model,
                num_layers=self.cfg.gru_num_layers,
                dropout=self.cfg.gru_dropout if self.cfg.gru_num_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            self.gru_smoother = None

        self.residual_mlp = (
            nn.Sequential(
                nn.Linear(self.cfg.d_model, self.cfg.residual_mlp_dim),
                nn.GELU(),
                nn.Linear(self.cfg.residual_mlp_dim, self.cfg.d_model),
            )
            if self.cfg.use_residual_mlp
            else None
        )

        # 6. Transformer Encoder backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.n_heads,
            dim_feedforward=self.cfg.dim_feedforward,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.n_layers)

        # 7. SOH regression head
        self.head = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, 512, 4] -> [B]."""
        if x.ndim != 3 or x.shape[-1] != 4:
            raise ValueError(f"Expected [B, L, 4], got {tuple(x.shape)}")
        batch_size, seq_len, _ = x.shape
        if seq_len != self.cfg.seq_len:
            raise ValueError(f"Expected sequence length {self.cfg.seq_len}, got {seq_len}.")

        # Flatten sequence for parallel quantum gate evaluation
        x_flat = x.reshape(batch_size * seq_len, 4)
        q_features = self.quantum_embed(x_flat)
        q_features = self.quantum_proj(q_features)
        q_sequence = q_features.reshape(batch_size, seq_len, self.cfg.d_model)

        # Optional GRU smoother
        if self.gru_smoother is not None:
            q_sequence, _ = self.gru_smoother(q_sequence)

        # Conv1D temporal smoothing along sequence length
        q_sequence = q_sequence.transpose(1, 2)
        q_sequence = self.temporal_smooth(q_sequence)
        q_sequence = q_sequence.transpose(1, 2)

        if self.residual_mlp is not None:
            q_sequence = q_sequence + self.residual_mlp(q_sequence)

        # Prepend CLS token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            q_sequence = torch.cat([cls_tokens, q_sequence], dim=1)

        # Add sinusoidal positional encoding
        if self.pos_encoder is not None:
            q_sequence = self.pos_encoder(q_sequence)

        # Transformer encoder self-attention
        transformed = self.transformer(q_sequence)

        # Pooling: extract CLS token representation
        if self.cfg.pooling == "cls":
            pooled = transformed[:, 0]
        else:
            pooled = transformed.mean(dim=1)

        # Regression to scalar SOH
        soh = self.head(pooled)
        return soh.squeeze(-1)


def rich_entangler_config() -> TEQTransformerConfig:
    """Standard configuration of the proposed TE-Q-Transformer with rich entangler."""
    return TEQTransformerConfig(entangler_type="rich")


__all__ = [
    "TEQTransformerConfig",
    "PositionalEncoding",
    "QuantumEmbeddingLayer",
    "TEQTransformer",
    "rich_entangler_config",
]
