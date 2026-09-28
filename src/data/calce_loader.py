"""CALCE CS2 battery dataset loader and zero-shot preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

from src.data.nasa_loader import FEATURE_IDX_TO_SCALE, transform_with_scaler

CALCE_CELL_IDS = ("CS2_35", "CS2_36", "CS2_37", "CS2_38")


class CALCEBatteryDataset(Dataset):
    """PyTorch Dataset for CALCE discharge windows and SOH target."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X.float()
        self.y = y.float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_calce_cell(data_dir: Path, cell_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads unscaled X, SOH, and cycle index for a CALCE cell."""
    x_path = data_dir / f"{cell_id}_X_unscaled.npy"
    y_path = data_dir / f"{cell_id}_soh.npy"
    c_path = data_dir / f"{cell_id}_cycle.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing arrays for CALCE cell {cell_id} in {data_dir}")
    X = np.load(x_path)
    y = np.load(y_path)
    cycle = np.load(c_path) if c_path.exists() else np.arange(len(y))
    return X.astype(np.float32), y.astype(np.float32), cycle


def get_calce_dataloaders(
    data_dir: Path,
    nasa_scaler: MinMaxScaler,
    batch_size: int = 8,
) -> Dict[str, DataLoader]:
    """Prepares zero-shot CALCE DataLoaders using the frozen NASA train scaler."""
    loaders: Dict[str, DataLoader] = {}
    for cell_id in CALCE_CELL_IDS:
        X_raw, y, _ = load_calce_cell(data_dir, cell_id)
        X_scaled = transform_with_scaler(X_raw, nasa_scaler, clip=True)
        loaders[cell_id] = DataLoader(
            CALCEBatteryDataset(X_scaled, torch.from_numpy(y).float()),
            batch_size=batch_size,
            shuffle=False,
        )
    return loaders


__all__ = [
    "CALCE_CELL_IDS",
    "CALCEBatteryDataset",
    "load_calce_cell",
    "get_calce_dataloaders",
]
