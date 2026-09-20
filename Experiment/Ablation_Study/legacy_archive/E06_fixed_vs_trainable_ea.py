"""Experiment E06: Fixed vs. Trainable Activation Energies (E_a).

Scientific Question:
Does allowing backpropagation to optimize Arrhenius activation energies (Ea_sei, Ea_pl)
outperform fixing them at static literature values?
"""

from __future__ import annotations

import argparse
from Experiment.Proposed_Model.evaluate import evaluate_nasa


def run_experiment() -> dict:
    print("\n==================================================")
    print("EXPERIMENT E06: FIXED VS. TRAINABLE ACTIVATION ENERGIES")
    print("Parameters: Ea_sei, Ea_pl (Trainable vs. Fixed)")
    print("==================================================")
    metrics = evaluate_nasa(experiment_id="E06_fixed_vs_trainable_ea")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_experiment()
