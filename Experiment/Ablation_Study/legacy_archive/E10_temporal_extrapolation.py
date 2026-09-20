"""Experiment E10: Temporal Extrapolation & Early-Life SOH Forecasting.

Scientific Question:
Can the model accurately predict the full battery degradation curve when trained
only on early cycling data (first 50%, 60%, or 70% of life)?
"""

from __future__ import annotations

import argparse
from Experiment.Proposed_Model.evaluate import evaluate_nasa


def run_experiment() -> dict:
    print("\n==================================================")
    print("EXPERIMENT E10: TEMPORAL EXTRAPOLATION")
    print("Evaluation: Early-to-Late Cycle Life Trajectory Forecasting")
    print("==================================================")
    metrics = evaluate_nasa(experiment_id="E10_temporal_extrapolation")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_experiment()
