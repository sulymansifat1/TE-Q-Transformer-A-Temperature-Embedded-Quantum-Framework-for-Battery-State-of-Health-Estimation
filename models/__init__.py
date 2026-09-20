"""Models package for TE-Q-Transformer research repository.

Subpackages:
- `models.proposed`: Proposed TE-Q-Transformer model and configurations.
- `models.baselines`: Benchmark baseline models (CNN1D, TCN, DLinear, Transformer,
  PatchTST, iTransformer, QLSTM, QNN-GRU, LSTM, GRU).
"""

from models.proposed.te_q_transformer import (
    TEQTransformer,
    TEQTransformerConfig,
    rich_entangler_config,
)
from models.baselines import get_baseline_model, list_baselines

__all__ = [
    "TEQTransformer",
    "TEQTransformerConfig",
    "rich_entangler_config",
    "get_baseline_model",
    "list_baselines",
]
