"""Authoritative TE-Q-Transformer model definition.

Re-exports the core PennyLane + PyTorch implementation from models.proposed.te_q_transformer.
DO NOT create duplicate or diverging architectures here.
"""

from __future__ import annotations

from models.proposed.te_q_transformer import (
    TEQTransformerConfig,
    PositionalEncoding,
    QuantumEmbeddingLayer,
    TEQTransformer,
    rich_entangler_config,
)

__all__ = [
    "TEQTransformerConfig",
    "PositionalEncoding",
    "QuantumEmbeddingLayer",
    "TEQTransformer",
    "rich_entangler_config",
]
