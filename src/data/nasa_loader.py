"""NASA battery dataset loader, multi-temperature splitting, and zero-leakage preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

NASA_TRAIN_CELLS = ("B0005", "B0006", "B0007", "B0029", "B0030", "B0031")
NASA_TEST_CELLS = ("B0018", "B0032")
NASA_SPLIT_CELL = "B0053"
NASA_SPLIT_RATIO = 0.70  # First 70% train (37 cycles), last 30% test (16 cycles)
FEATURE_IDX_TO_SCALE = (0, 1, 3)  # Voltage(0), Current(1), Time_norm(3); Temp(2) left in Celsius
SEQUENCE_LENGTH = 512


class NASABatteryDataset(Dataset):
    """PyTorch Dataset for battery discharge windows and SOH target."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X.float()
        self.y = y.float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def load_cell_arrays(data_dir: Path, cell_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads X and soh arrays for a given NASA cell."""
    x_path = data_dir / f"{cell_id}_X.npy"
    y_path = data_dir / f"{cell_id}_soh.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing data arrays for cell {cell_id} in {data_dir}")
    X = np.load(x_path)
    y = np.load(y_path)
    return X.astype(np.float32, copy=True), y.astype(np.float32, copy=False)


def normalize_soh_per_cell(y: np.ndarray) -> np.ndarray:
    """Normalizes SOH as discharge capacity relative to initial cycle C0."""
    c0 = float(y[0])
    return (y / np.float32(c0)).astype(np.float32, copy=False)


def load_full_cell(data_dir: Path, cell_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads full cell profile and normalizes SOH."""
    X, y = load_cell_arrays(data_dir, cell_id)
    y = normalize_soh_per_cell(y)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def split_cell_70_30(
    data_dir: Path, cell_id: str
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Splits a cell into 70% train and 30% test segments chronologically."""
    X, y = load_cell_arrays(data_dir, cell_id)
    y = normalize_soh_per_cell(y)
    split_idx = int(len(y) * NASA_SPLIT_RATIO)
    train_part = (
        torch.from_numpy(X[:split_idx].copy()).float(),
        torch.from_numpy(y[:split_idx].copy()).float(),
    )
    test_part = (
        torch.from_numpy(X[split_idx:].copy()).float(),
        torch.from_numpy(y[split_idx:].copy()).float(),
    )
    return train_part, test_part


def fit_train_scaler(train_X: torch.Tensor | np.ndarray) -> MinMaxScaler:
    """Fits MinMaxScaler exclusively on training split for features (0, 1, 3)."""
    if isinstance(train_X, torch.Tensor):
        train_X = train_X.cpu().numpy()
    flat_scaled = train_X[:, :, FEATURE_IDX_TO_SCALE].reshape(-1, len(FEATURE_IDX_TO_SCALE))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(flat_scaled)
    return scaler


def transform_with_scaler(
    X: torch.Tensor | np.ndarray, scaler: MinMaxScaler, clip: bool = True
) -> torch.Tensor:
    """Applies fitted scaler to features (0, 1, 3), keeping temperature unscaled."""
    X_tensor = torch.from_numpy(X.copy()).float() if isinstance(X, np.ndarray) else X.clone().float()
    n_samples, seq_len, _ = X_tensor.shape
    flat_scaled = (
        X_tensor[:, :, FEATURE_IDX_TO_SCALE]
        .reshape(-1, len(FEATURE_IDX_TO_SCALE))
        .cpu()
        .numpy()
    )
    transformed = scaler.transform(flat_scaled)
    if clip:
        transformed = np.clip(transformed, 0.0, 1.0)
    reshaped = torch.from_numpy(
        transformed.reshape(n_samples, seq_len, len(FEATURE_IDX_TO_SCALE))
    ).float()
    for i, orig_idx in enumerate(FEATURE_IDX_TO_SCALE):
        X_tensor[:, :, orig_idx] = reshaped[:, :, i]
    return X_tensor


def get_nasa_dataloaders(
    data_dir: Path,
    batch_size: int = 8,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[DataLoader, Dict[str, DataLoader], MinMaxScaler]:
    """Constructs train DataLoader and test DataLoaders for all test cells."""
    train_X_list, train_y_list = [], []
    for cell_id in NASA_TRAIN_CELLS:
        cx, cy = load_full_cell(data_dir, cell_id)
        train_X_list.append(cx)
        train_y_list.append(cy)

    (split_tr_x, split_tr_y), (split_te_x, split_te_y) = split_cell_70_30(
        data_dir, NASA_SPLIT_CELL
    )
    train_X_list.append(split_tr_x)
    train_y_list.append(split_tr_y)

    train_X_raw = torch.cat(train_X_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    if scaler is None:
        scaler = fit_train_scaler(train_X_raw)

    train_X = transform_with_scaler(train_X_raw, scaler, clip=True)
    train_ds = NASABatteryDataset(train_X, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_loaders: Dict[str, DataLoader] = {}
    for cell_id in NASA_TEST_CELLS:
        te_x, te_y = load_full_cell(data_dir, cell_id)
        te_x_scaled = transform_with_scaler(te_x, scaler, clip=True)
        test_loaders[cell_id] = DataLoader(
            NASABatteryDataset(te_x_scaled, te_y), batch_size=batch_size, shuffle=False
        )

    split_te_x_scaled = transform_with_scaler(split_te_x, scaler, clip=True)
    test_loaders[f"{NASA_SPLIT_CELL}_test"] = DataLoader(
        NASABatteryDataset(split_te_x_scaled, split_te_y),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loaders, scaler


__all__ = [
    "NASA_TRAIN_CELLS",
    "NASA_TEST_CELLS",
    "NASA_SPLIT_CELL",
    "NASA_SPLIT_RATIO",
    "FEATURE_IDX_TO_SCALE",
    "SEQUENCE_LENGTH",
    "NASABatteryDataset",
    "load_cell_arrays",
    "normalize_soh_per_cell",
    "load_full_cell",
    "split_cell_70_30",
    "fit_train_scaler",
    "transform_with_scaler",
    "get_nasa_dataloaders",
]
