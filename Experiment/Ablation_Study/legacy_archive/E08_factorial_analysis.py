"""Experiment E08: Factorial Analysis.

Scientific Question:
How do the three core components (Physics Temperature Gate, Quantum Entanglement,
and Temporal Smoothing) interact to produce the overall SOH estimation performance?
"""

from __future__ import annotations

import argparse
from Experiment.Proposed_Model.evaluate import evaluate_nasa


def run_experiment() -> dict:
    print("\n==================================================")
    print("EXPERIMENT E08: FACTORIAL ANALYSIS")
    print("Design: 2 x 2 x 2 Factorial (Physics Gate x Quantum Circuit x Temporal Smoothing)")
    print("==================================================")
    metrics = evaluate_nasa(experiment_id="E08_factorial_analysis")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_experiment()
