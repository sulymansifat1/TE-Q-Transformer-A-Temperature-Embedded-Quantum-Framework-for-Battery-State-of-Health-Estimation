"""Experiment E11: Unseen Temperature Generalization.

Scientific Question:
Does physics-guided temperature encoding enable the model to generalize to
unseen ambient operating temperatures (e.g. 4°C, 24°C, 44°C) without catastrophic drift?
"""

from __future__ import annotations

import argparse
from Experiment.Proposed_Model.evaluate import evaluate_nasa


def run_experiment() -> dict:
    print("\n==================================================")
    print("EXPERIMENT E11: UNSEEN TEMPERATURE GENERALIZATION")
    print("Condition: Cross-Temperature Regimes (4 deg C, 24 deg C, 44 deg C)")
    print("==================================================")
    metrics = evaluate_nasa(experiment_id="E11_unseen_temperature")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_experiment()
