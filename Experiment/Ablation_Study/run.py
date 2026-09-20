"""Central entry point for all ablation study experiments (E01 to E07)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.Ablation_Study import ABLATION_EXPERIMENT_NAMES, get_ablation_runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Central runner for Ablation Studies (E01-E07).")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", *ABLATION_EXPERIMENT_NAMES],
        help="Ablation experiment ID to execute (default: all).",
    )
    args = parser.parse_args()

    exps_to_run = ABLATION_EXPERIMENT_NAMES if args.experiment == "all" else [args.experiment]

    print("==================================================")
    print("ABLATION STUDY EXPERIMENT RUNNER")
    print(f"Experiments to execute: {exps_to_run}")
    print("==================================================")

    results = {}
    for exp_id in exps_to_run:
        fn = get_ablation_runner(exp_id)
        print(f"\n>>> Launching {exp_id}...")
        res = fn()
        results[exp_id] = res

    print("\n==================================================")
    print("ALL REQUESTED ABLATION EXPERIMENTS COMPLETED.")
    print("Outputs saved under GarbageResults/ and Experiment/Result/")
    print("==================================================")


if __name__ == "__main__":
    main()
