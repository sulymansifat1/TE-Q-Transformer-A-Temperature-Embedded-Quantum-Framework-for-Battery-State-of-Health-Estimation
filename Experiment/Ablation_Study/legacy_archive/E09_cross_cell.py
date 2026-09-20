"""Experiment E09: Cross-Cell Generalization.

Scientific Question:
Does TE-Q-Transformer maintain high predictive accuracy when evaluated across
unseen battery cells with diverse manufacturing variations and aging profiles?
"""

from __future__ import annotations

import argparse
from Experiment.Proposed_Model.evaluate import evaluate_nasa


def run_experiment() -> dict:
    print("\n==================================================")
    print("EXPERIMENT E09: CROSS-CELL GENERALIZATION")
    print("Protocol: Leave-One-Cell-Out Cross-Validation across NASA cells")
    print("==================================================")
    metrics = evaluate_nasa(experiment_id="E09_cross_cell")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_experiment()
