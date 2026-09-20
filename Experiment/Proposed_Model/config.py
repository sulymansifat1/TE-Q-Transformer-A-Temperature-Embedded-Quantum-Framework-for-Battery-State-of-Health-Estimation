"""Configuration for TE-Q-Transformer proposed model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from src.models.teq_transformer import TEQTransformerConfig, rich_entangler_config


def get_default_config() -> TEQTransformerConfig:
    """Returns the primary TE-Q-Transformer configuration with rich entangler."""
    return rich_entangler_config()


def get_training_hyperparameters() -> dict[str, object]:
    """Canonical training hyperparameters matching the NASA ablation baseline."""
    return {
        "batch_size": 8,
        "num_epochs": 80,
        "patience": 20,
        "learning_rate": 1e-3,
        "weight_decay": 5e-2,
        "scheduler_patience": 10,
        "scheduler_factor": 0.5,
        "grad_clip_norm": 1.0,
        "seed": 42,
    }
