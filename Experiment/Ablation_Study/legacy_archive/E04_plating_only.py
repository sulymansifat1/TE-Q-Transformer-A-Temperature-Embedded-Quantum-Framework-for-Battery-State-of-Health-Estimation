"""Experiment E04: Matched Classical vs Quantum Representation.

Note: In the V2 publication plan, the SEI vs Plating mechanism comparison was unified
under E03 (GarbageResults/AblationE03/), and E04 is designated as the Matched Classical
vs. Quantum Representation Ablation under Physics-Guided Temperature Encoding.
"""

from __future__ import annotations

import argparse
from Experiment.Ablation_Study.E04_classical_vs_quantum import run_full_e04_experiment


def run_experiment() -> None:
    run_full_e04_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E04 Classical vs Quantum Ablation")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    run_full_e04_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        patience=args.patience,
    )
