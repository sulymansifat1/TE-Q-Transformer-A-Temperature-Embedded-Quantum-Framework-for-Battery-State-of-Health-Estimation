"""Evaluation metrics for battery State-of-Health (SOH) regression."""

from __future__ import annotations

from typing import Dict, List
import numpy as np


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard evaluation metrics: RMSE, MAE, MAPE (%), R2, and Max Error."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(abs_errors))

    # MAPE: avoid division by zero with small eps
    eps = 1e-8
    mape = float(np.mean(abs_errors / np.clip(np.abs(y_true), a_min=eps, a_max=None)) * 100.0)

    # R2 Score
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Max Error
    max_error = float(np.max(abs_errors))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "R2": r2,
        "MaxE": max_error,
    }


def calculate_macro_metrics(cell_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculates macro-average across multiple cells."""
    keys = ["RMSE", "MAE", "MAPE (%)", "R2", "MaxE"]
    macro = {}
    for k in keys:
        values = [m[k] for m in cell_metrics if k in m and not np.isnan(m[k])]
        macro[k] = float(np.mean(values)) if values else float("nan")
    return macro


__all__ = ["calculate_metrics", "calculate_macro_metrics"]
