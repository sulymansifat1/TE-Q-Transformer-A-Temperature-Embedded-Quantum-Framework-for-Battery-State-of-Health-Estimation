"""Standard battery SOH regression metrics."""

from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Computes standard SOH estimation metrics (RMSE, MAE, MAPE, R2, MaxE)."""
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape}, predicted {predicted.shape}.")

    abs_error = np.abs(actual - predicted)
    denom = np.clip(np.abs(actual), a_min=1e-8, a_max=None)

    # Calculate R2 safely
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MAPE (%)": float(np.mean(abs_error / denom) * 100.0),
        "R2": r2,
        "MaxE": float(np.max(abs_error)),
    }
