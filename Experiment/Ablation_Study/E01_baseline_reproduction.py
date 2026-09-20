"""Experiment E01: Baseline Reproduction.

Scientific Question:
Does the frozen primary TE-Q-Transformer checkpoint accurately reproduce the reported
NASA multi-temperature test performance and CALCE zero-shot domain transfer metrics?
"""

from __future__ import annotations

import argparse
from Experiment.Proposed_Model.evaluate import evaluate_nasa, evaluate_calce_zero_shot
from Experiment.utils.paths import NASA_FROZEN_CHECKPOINT_PATH


def run_experiment(evaluate_calce: bool = True) -> dict:
    print("\n==================================================")
    print("EXPERIMENT E01: BASELINE REPRODUCTION")
    print("Checkpoint: ", NASA_FROZEN_CHECKPOINT_PATH)
    print("==================================================")

    nasa_metrics = evaluate_nasa(
        checkpoint_path=NASA_FROZEN_CHECKPOINT_PATH,
        experiment_id="E01_baseline_reproduction",
    )

    calce_metrics = None
    if evaluate_calce:
        calce_metrics = evaluate_calce_zero_shot(
            checkpoint_path=NASA_FROZEN_CHECKPOINT_PATH,
            experiment_id="E01_baseline_reproduction",
        )

    return {"NASA": nasa_metrics, "CALCE": calce_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment E01: Baseline Reproduction.")
    parser.add_argument("--skip-calce", action="store_true", help="Skip CALCE zero-shot evaluation.")
    args = parser.parse_args()
    run_experiment(evaluate_calce=not args.skip_calce)
