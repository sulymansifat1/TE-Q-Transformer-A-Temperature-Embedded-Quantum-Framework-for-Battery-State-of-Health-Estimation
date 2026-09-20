"""Experiment E03: SEI vs Plating vs Combined Physics-Guided Temperature Encoding.

Direct entry point forwarding to the authoritative 3-way ablation runner.
"""

from __future__ import annotations

import argparse
from Experiment.Ablation_Study.E03_physics_mechanisms import run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment E03: SEI vs Plating vs Combined.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    run_experiment(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, patience=args.patience)
