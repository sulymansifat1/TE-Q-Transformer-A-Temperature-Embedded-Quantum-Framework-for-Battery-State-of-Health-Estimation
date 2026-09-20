"""Evaluation engine for TE-Q-Transformer on NASA and CALCE benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch

from Experiment.utils.paths import (
    NASA_DATA_DIR,
    CALCE_PROCESSED_DIR,
    NASA_FROZEN_CHECKPOINT_PATH,
    get_result_subdirs,
)
from Experiment.utils.metrics import compute_metrics
from Experiment.utils.plotting import plot_soh_trajectory, plot_parity
from Experiment.Dataset.nasa import get_nasa_dataloaders
from Experiment.Dataset.calce import get_calce_dataloaders
from Experiment.Proposed_Model.model import TEQTransformer, TEQTransformerConfig, rich_entangler_config


def evaluate_model_on_loader(
    model: TEQTransformer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs inference and returns concatenated ground truth and predicted SOH arrays."""
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            y_true_list.append(batch_y.cpu().numpy().reshape(-1))
            y_pred_list.append(preds.cpu().numpy().reshape(-1))

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


def evaluate_nasa(
    model: TEQTransformer | None = None,
    checkpoint_path: Path | None = None,
    data_dir: Path = NASA_DATA_DIR,
    experiment_id: str = "E01_TEQ",
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Evaluates TE-Q-Transformer on the NASA held-out test split."""
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    subdirs = get_result_subdirs("NASA")

    # Load model if not supplied directly
    if model is None:
        checkpoint_path = checkpoint_path or NASA_FROZEN_CHECKPOINT_PATH
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        model = TEQTransformer(rich_entangler_config()).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    _, test_loaders, _ = get_nasa_dataloaders(data_dir=data_dir, batch_size=8)

    results: Dict[str, Any] = {}
    csv_rows = []

    print(f"\n[Evaluation] Evaluating NASA on {device} (Experiment: {experiment_id})...")

    all_y_true, all_y_pred = [], []

    for cell_id, loader in test_loaders.items():
        y_true, y_pred = evaluate_model_on_loader(model, loader, device)
        cell_metrics = compute_metrics(y_true, y_pred)
        results[cell_id] = cell_metrics
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        # Save predictions
        np.save(subdirs["predictions"] / f"{experiment_id}_{cell_id}_y_true.npy", y_true)
        np.save(subdirs["predictions"] / f"{experiment_id}_{cell_id}_y_pred.npy", y_pred)

        # Generate trajectory plot
        cycles = np.arange(1, len(y_true) + 1)
        plot_soh_trajectory(cycles, y_true, y_pred, test_cell=f"{experiment_id}_{cell_id}", output_dir=subdirs["plots"])

        row = {"experiment": experiment_id, "dataset": "NASA", "cell": cell_id, **cell_metrics}
        csv_rows.append(row)
        print(f"  {cell_id:12s} | RMSE: {cell_metrics['RMSE']:.5f} | MAE: {cell_metrics['MAE']:.5f} | R2: {cell_metrics['R2']:.5f}")

    # Compute Macro metrics across test cells
    macro_metrics = {
        k: float(np.mean([results[c][k] for c in test_loaders]))
        for k in ["RMSE", "MAE", "MAPE (%)", "R2", "MaxE"]
    }
    results["macro"] = macro_metrics
    csv_rows.append({"experiment": experiment_id, "dataset": "NASA", "cell": "MACRO_AVG", **macro_metrics})
    print(f"  {'MACRO_AVG':12s} | RMSE: {macro_metrics['RMSE']:.5f} | MAE: {macro_metrics['MAE']:.5f} | R2: {macro_metrics['R2']:.5f}")

    # Save CSV and JSON reports
    csv_path = subdirs["metrics"] / f"{experiment_id}_NASA_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path = subdirs["reports"] / f"{experiment_id}_NASA_report.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Generate Parity Plot
    concat_true = np.concatenate(all_y_true)
    concat_pred = np.concatenate(all_y_pred)
    plot_parity(concat_true, concat_pred, f"{experiment_id} NASA Parity Plot", subdirs["plots"] / f"{experiment_id}_NASA_parity.png")

    return results


def evaluate_calce_zero_shot(
    model: TEQTransformer | None = None,
    checkpoint_path: Path | None = None,
    experiment_id: str = "E01_TEQ",
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Zero-shot evaluation of NASA-trained model on unseen CALCE CS2 cells."""
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    subdirs = get_result_subdirs("CALCE")

    if model is None:
        checkpoint_path = checkpoint_path or NASA_FROZEN_CHECKPOINT_PATH
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        model = TEQTransformer(rich_entangler_config()).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    calce_loaders, raw_calce, _ = get_calce_dataloaders(batch_size=8)

    results: Dict[str, Any] = {}
    csv_rows = []
    all_y_true, all_y_pred = [], []

    print(f"\n[Evaluation] Evaluating CALCE Zero-Shot on {device} (Experiment: {experiment_id})...")

    for cell_id, loader in calce_loaders.items():
        y_true, y_pred = evaluate_model_on_loader(model, loader, device)
        cell_metrics = compute_metrics(y_true, y_pred)
        results[cell_id] = cell_metrics
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        np.save(subdirs["predictions"] / f"{experiment_id}_{cell_id}_y_true.npy", y_true)
        np.save(subdirs["predictions"] / f"{experiment_id}_{cell_id}_y_pred.npy", y_pred)

        cycles = raw_calce[cell_id]["cycle"]
        plot_soh_trajectory(cycles, y_true, y_pred, test_cell=f"{experiment_id}_{cell_id}", output_dir=subdirs["plots"])

        row = {"experiment": experiment_id, "dataset": "CALCE", "cell": cell_id, **cell_metrics}
        csv_rows.append(row)
        print(f"  {cell_id:12s} | RMSE: {cell_metrics['RMSE']:.5f} | MAE: {cell_metrics['MAE']:.5f} | R2: {cell_metrics['R2']:.5f}")

    macro_metrics = {
        k: float(np.mean([results[c][k] for c in calce_loaders]))
        for k in ["RMSE", "MAE", "MAPE (%)", "R2", "MaxE"]
    }
    results["macro"] = macro_metrics
    csv_rows.append({"experiment": experiment_id, "dataset": "CALCE", "cell": "MACRO_AVG", **macro_metrics})
    print(f"  {'MACRO_AVG':12s} | RMSE: {macro_metrics['RMSE']:.5f} | MAE: {macro_metrics['MAE']:.5f} | R2: {macro_metrics['R2']:.5f}")

    csv_path = subdirs["metrics"] / f"{experiment_id}_CALCE_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path = subdirs["reports"] / f"{experiment_id}_CALCE_report.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    concat_true = np.concatenate(all_y_true)
    concat_pred = np.concatenate(all_y_pred)
    plot_parity(concat_true, concat_pred, f"{experiment_id} CALCE Parity Plot", subdirs["plots"] / f"{experiment_id}_CALCE_parity.png")

    return results
