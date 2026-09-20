"""Experiment E06: Cross-Cell and Cross-Dataset Generalization.

Scientific Question:
How reliably does TE-Q-Transformer generalize to unseen battery cells with distinct
manufacturing variations, thermal conditions, and zero-shot cross-dataset transfer (NASA -> CALCE)?

Evaluations:
1. NASA Unseen Cell Generalization:
   - Evaluated on held-out NASA cells under nominal (B0018), elevated (B0032), and sub-ambient (B0053_test).
2. Zero-Shot Cross-Dataset Transfer (NASA -> CALCE):
   - Model trained strictly on NASA (CS2 chemistry / 18650 format) evaluated directly
     on CALCE prismatic cells (CS2_35, CS2_36, CS2_37, CS2_38) without retraining or fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.Proposed_Model.evaluate import evaluate_nasa, evaluate_calce_zero_shot
from Experiment.utils.paths import ROOT_DIR

E06_OUTPUT_DIR = ROOT_DIR / "GarbageResults" / "AblationE06"


def run_cross_cell_evaluation() -> Dict[str, Any]:
    print("=" * 60)
    print("EXPERIMENT E06: CROSS-CELL & ZERO-SHOT TRANSFER GENERALIZATION")
    print("=" * 60)

    E06_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = E06_OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Part 1: NASA Unseen Cells Evaluation ---")
    nasa_res = evaluate_nasa(experiment_id="E06_nasa_unseen")
    nasa_macro = nasa_res["macro"]
    print(f"  NASA Macro RMSE: {nasa_macro['RMSE']:.5f} | MAE: {nasa_macro['MAE']:.5f} | R2: {nasa_macro['R2']:.5f}")

    print("\n--- Part 2: CALCE Zero-Shot Transfer Evaluation ---")
    calce_res = evaluate_calce_zero_shot(experiment_id="E06_calce_zero_shot")
    calce_macro = calce_res["macro"]
    print(f"  CALCE Macro RMSE: {calce_macro['RMSE']:.5f} | MAE: {calce_macro['MAE']:.5f} | R2: {calce_macro['R2']:.5f}")

    # Export combined summary CSV
    summary_csv = metrics_dir / "e06_cross_cell_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Evaluation_Type", "Target_Domain", "RMSE", "MAE", "MAPE (%)", "R2", "MaxE"])
        writer.writerow(["Within-Dataset Unseen", "NASA (Macro)", nasa_macro["RMSE"], nasa_macro["MAE"], nasa_macro["MAPE (%)"], nasa_macro["R2"], nasa_macro["MaxE"]])
        writer.writerow(["Zero-Shot Cross-Dataset", "CALCE (Macro)", calce_macro["RMSE"], calce_macro["MAE"], calce_macro["MAPE (%)"], calce_macro["R2"], calce_macro["MaxE"]])

    print("\n" + "=" * 60)
    print(f"Cross-cell summary saved to: {summary_csv}")
    print("=" * 60 + "\n")

    return {
        "nasa": nasa_res,
        "calce": calce_res,
    }


def run_experiment() -> Dict[str, Any]:
    return run_cross_cell_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment E06 (Cross-Cell Generalization)")
    args = parser.parse_args()
    run_cross_cell_evaluation()
