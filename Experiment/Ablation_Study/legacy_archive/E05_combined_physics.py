"""Experiment E05: Combined Dual-Mechanism Physics (SEI + Plating).

Scientific Question:
Does coupling both high-temperature SEI layer growth and low-temperature Lithium plating
yield synergistic degradation tracking across multi-temperature operating regimes?
"""

from __future__ import annotations

import argparse
from Experiment.Proposed_Model.evaluate import evaluate_nasa


def run_experiment() -> dict:
    print("\n==================================================")
    print("EXPERIMENT E05: COMBINED DUAL-MECHANISM PHYSICS")
    print("Physics Gate: phi = sei_term + plating_term")
    print("==================================================")
    metrics = evaluate_nasa(experiment_id="E05_combined_physics")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_experiment()
