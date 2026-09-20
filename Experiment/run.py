"""Top-level master experiment orchestrator for TE-Q-Transformer V2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASELINE_NAMES = [
    "LSTM",
    "GRU",
    "BiLSTM",
    "CNN1D",
    "TCN",
    "Transformer",
    "Informer",
    "PatchTST",
    "QNN_GRU",
    "QLSTM",
]

ABLATION_NAMES = [
    "E01",
    "E02",
    "E03",
    "E04",
    "E05",
    "E06",
    "E07",
]


def run_proposed_group(dataset: str = "all", evaluate_only: bool = True, train: bool = False) -> None:
    print("\n==================================================")
    print("EXECUTING GROUP: PROPOSED MODEL (TE-Q-Transformer)")
    print(f"Dataset: {dataset} | Mode: {'Train from scratch' if train else 'Evaluate checkpoint'}")
    print("==================================================")
    from Experiment.Proposed_Model.run import main as run_proposed_cli
    sys.argv = ["run.py", "--dataset", dataset]
    if train:
        sys.argv.append("--train")
    if evaluate_only and not train:
        sys.argv.append("--evaluate-only")
    run_proposed_cli()


def run_baseline_group(model_name: str = "all", epochs: int = 80, batch_size: int = 8) -> None:
    print("\n==================================================")
    print("EXECUTING GROUP: BASELINE MODELS")
    print(f"Model: {model_name} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("==================================================")
    from Experiment.Baseline.run import train_and_eval_single_baseline
    models = BASELINE_NAMES if model_name == "all" else [model_name]
    for m in models:
        train_and_eval_single_baseline(m, num_epochs=epochs, batch_size=batch_size)


def run_ablation_group(experiment_id: str = "all") -> None:
    print("\n==================================================")
    print("EXECUTING GROUP: ABLATION STUDIES")
    print(f"Experiment: {experiment_id}")
    print("==================================================")
    from Experiment.Ablation_Study import EXPERIMENT_REGISTRY
    exps = list(EXPERIMENT_REGISTRY.keys()) if experiment_id == "all" else [experiment_id]
    for exp_id in exps:
        print(f"\n>>> Running {exp_id}...")
        fn = EXPERIMENT_REGISTRY[exp_id]
        fn()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TE-Q-Transformer V2 Master Experiment Runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Experiment/run.py --help
  python Experiment/run.py --group proposed
  python Experiment/run.py --group proposed --dataset NASA
  python Experiment/run.py --group proposed --dataset CALCE
  python Experiment/run.py --group baseline --model LSTM
  python Experiment/run.py --group baseline --model all
  python Experiment/run.py --group ablation --experiment E01
  python Experiment/run.py --group ablation --experiment all
  python Experiment/run.py --group all
        """,
    )
    parser.add_argument(
        "--group",
        type=str,
        default="proposed",
        choices=["proposed", "baseline", "ablation", "all"],
        help="Experiment group to execute (default: proposed).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["NASA", "CALCE", "all"],
        help="Target dataset (for --group proposed).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", *BASELINE_NAMES],
        help="Specific baseline model (for --group baseline).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", *ABLATION_NAMES],
        help="Specific ablation study ID E01-E15 (for --group ablation).",
    )
    parser.add_argument("--train", action="store_true", help="Train from scratch instead of evaluating checkpoint.")
    parser.add_argument("--epochs", type=int, default=80, help="Epochs for training runs (default: 80).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8).")
    args = parser.parse_args()

    if args.group in ("proposed", "all"):
        run_proposed_group(dataset=args.dataset, train=args.train)

    if args.group in ("baseline", "all"):
        run_baseline_group(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size)

    if args.group in ("ablation", "all"):
        run_ablation_group(experiment_id=args.experiment)

    print("\n==================================================")
    print("ALL REQUESTED EXPERIMENT GROUPS COMPLETED SUCCESSFULLY.")
    print("Check Experiment/Result/ for generated metrics, plots, and tables.")
    print("==================================================")


if __name__ == "__main__":
    main()
