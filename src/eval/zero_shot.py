"""Zero-shot transfer evaluation from NASA model to CALCE CS2 battery cells."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from src.data.calce_loader import CALCE_CELL_IDS, load_calce_cell
from src.data.nasa_loader import transform_with_scaler
from src.eval.metrics import calculate_metrics, calculate_macro_metrics


def evaluate_zero_shot_calce(
    model: torch.nn.Module,
    calce_dir: Path,
    nasa_scaler: MinMaxScaler,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Evaluates a frozen NASA-trained model on CALCE CS2 cells without retraining."""
    model.eval()
    model.to(device)

    per_cell_results = {}
    all_actuals = []
    all_preds = []

    with torch.no_grad():
        for cell_id in CALCE_CELL_IDS:
            X_raw, y_true, cycles = load_calce_cell(calce_dir, cell_id)
            X_scaled = transform_with_scaler(X_raw, nasa_scaler, clip=True)

            # Batched inference
            preds = []
            for i in range(0, len(X_scaled), batch_size):
                batch_x = X_scaled[i : i + batch_size].to(device)
                batch_pred = model(batch_x)
                preds.append(batch_pred.cpu().numpy())

            y_pred = np.concatenate(preds)
            cell_metrics = calculate_metrics(y_true, y_pred)
            per_cell_results[cell_id] = {
                "metrics": cell_metrics,
                "predictions": y_pred,
                "actuals": y_true,
                "cycles": cycles,
            }
            all_actuals.append(y_true)
            all_preds.append(y_pred)

    macro_metrics = calculate_macro_metrics([res["metrics"] for res in per_cell_results.values()])
    pooled_actuals = np.concatenate(all_actuals)
    pooled_preds = np.concatenate(all_preds)
    pooled_metrics = calculate_metrics(pooled_actuals, pooled_preds)

    return {
        "per_cell": per_cell_results,
        "macro": macro_metrics,
        "pooled": pooled_metrics,
    }


__all__ = ["evaluate_zero_shot_calce"]
