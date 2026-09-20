"""Central runner for the proposed TE-Q-Transformer model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.utils.paths import NASA_FROZEN_CHECKPOINT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Run proposed TE-Q-Transformer training or evaluation.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["NASA", "CALCE", "all"],
        default="all",
        help="Dataset benchmark to evaluate (default: all).",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        default=True,
        help="Run evaluation on pre-trained checkpoint without re-training (default: True).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Trigger training of TE-Q-Transformer from scratch.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=NASA_FROZEN_CHECKPOINT_PATH,
        help="Path to checkpoint .pth file (default: frozen rich entangler baseline).",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs (if --train).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8).")
    args = parser.parse_args()

    from Experiment.Proposed_Model.train import train_proposed_model
    from Experiment.Proposed_Model.evaluate import evaluate_nasa, evaluate_calce_zero_shot

    trained_model = None
    checkpoint_to_eval = args.checkpoint

    if args.train:
        print("[Mode] Training proposed model from scratch...")
        trained_model, history, saved_ckpt = train_proposed_model(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )
        checkpoint_to_eval = saved_ckpt

    if args.dataset in ("NASA", "all"):
        evaluate_nasa(
            model=trained_model,
            checkpoint_path=checkpoint_to_eval,
            experiment_id="E01_TEQ",
        )

    if args.dataset in ("CALCE", "all"):
        evaluate_calce_zero_shot(
            model=trained_model,
            checkpoint_path=checkpoint_to_eval,
            experiment_id="E01_TEQ",
        )

    print("\n[Proposed Model] Execution complete. Results saved in Experiment/Result/")


if __name__ == "__main__":
    main()
