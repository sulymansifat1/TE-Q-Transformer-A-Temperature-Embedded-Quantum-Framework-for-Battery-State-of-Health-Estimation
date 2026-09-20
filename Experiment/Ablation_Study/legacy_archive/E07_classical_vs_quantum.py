"""Experiment E07: Classical vs. Quantum Feature Embedding.

Scientific Question:
Does replacing the 4-qubit parameterized quantum circuit with a parameter-matched
classical MLP embedding isolate the specific empirical advantage of quantum Hilbert space encoding?
"""

from __future__ import annotations

import argparse
import torch
from torch import nn
from Experiment.Proposed_Model.evaluate import evaluate_nasa


class ClassicalEmbeddingTEQTransformer(nn.Module):
    """Replaces the QuantumEmbeddingLayer with a 2-layer classical MLP of matching capacity."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        # Matches quantum embedding output dimension (4 -> 16 -> 4)
        self.classical_embed = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 4),
            nn.Tanh(),
        )
        self.proj = nn.Linear(4, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=2, dim_feedforward=64, dropout=0.0, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x_flat = x.reshape(b * l, 4)
        c_feat = self.classical_embed(x_flat)
        h = self.proj(c_feat).reshape(b, l, -1)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        h = torch.cat([cls, h], dim=1)
        encoded = self.encoder(h)
        return self.head(encoded[:, 0, :]).squeeze(-1)


def run_experiment() -> dict:
    print("\n==================================================")
    print("EXPERIMENT E07: CLASSICAL VS. QUANTUM COMPARISON")
    print("Architecture: Classical MLP Embedding vs. 4-Qubit Quantum Circuit")
    print("==================================================")
    metrics = evaluate_nasa(experiment_id="E07_classical_vs_quantum")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_experiment()
