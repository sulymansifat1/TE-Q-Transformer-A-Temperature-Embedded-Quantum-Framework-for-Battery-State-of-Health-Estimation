"""Central entry point for all baseline model training and evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.Baseline import BASELINE_NAMES, get_baseline_model_class


def train_and_eval_single_baseline(
    model_name: str,
    num_epochs: int = 80,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 20,
    seed: int = 42,
    device: object = None,
) -> Dict[str, Any]:
    """Trains a baseline model on NASA train split and evaluates on test split."""
    import numpy as np
    import torch
    from torch import nn, optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    from Experiment.utils.paths import NASA_DATA_DIR, get_result_subdirs, RESULT_TABLES_DIR
    from Experiment.utils.seed import seed_everything
    from Experiment.utils.metrics import compute_metrics
    from Experiment.utils.plotting import plot_soh_trajectory, plot_parity
    from Experiment.Dataset.nasa import get_nasa_dataloaders

    seed_everything(seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    subdirs = get_result_subdirs("NASA")

    model_cls = get_baseline_model_class(model_name)
    model = model_cls().to(device)

    train_loader, test_loaders, _ = get_nasa_dataloaders(batch_size=batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience_cnt = 0
    history: Dict[str, list] = {"epoch": [], "train_loss": [], "lr": []}
    ckpt_path = subdirs["checkpoints"] / f"{model_name}_best.pth"

    print(f"\n[{model_name}] Training on {device} (Epochs: {num_epochs}, Batch: {batch_size})...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        mean_l = float(np.mean(losses))
        history["epoch"].append(epoch)
        history["train_loss"].append(mean_l)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step(mean_l)

        if mean_l < best_loss:
            best_loss = mean_l
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d} | Train Loss: {mean_l:.6f}")

    # Load best weights for evaluation
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    cell_results: Dict[str, Any] = {}
    csv_rows = []
    all_true, all_pred = [], []

    print(f"[{model_name}] Evaluating on NASA held-out test cells...")
    with torch.no_grad():
        for cell_id, loader in test_loaders.items():
            y_t, y_p = [], []
            for bx, by in loader:
                bx = bx.to(device)
                pred = model(bx)
                y_t.append(by.cpu().numpy().reshape(-1))
                y_p.append(pred.cpu().numpy().reshape(-1))

            yt_arr = np.concatenate(y_t)
            yp_arr = np.concatenate(y_p)
            all_true.append(yt_arr)
            all_pred.append(yp_arr)

            m = compute_metrics(yt_arr, yp_arr)
            cell_results[cell_id] = m
            csv_rows.append({"model": model_name, "dataset": "NASA", "cell": cell_id, **m})

            np.save(subdirs["predictions"] / f"{model_name}_{cell_id}_y_true.npy", yt_arr)
            np.save(subdirs["predictions"] / f"{model_name}_{cell_id}_y_pred.npy", yp_arr)
            cycles = np.arange(1, len(yt_arr) + 1)
            plot_soh_trajectory(cycles, yt_arr, yp_arr, test_cell=f"{model_name}_{cell_id}", output_dir=subdirs["plots"])

            print(f"  {cell_id:12s} | RMSE: {m['RMSE']:.5f} | MAE: {m['MAE']:.5f} | R2: {m['R2']:.5f}")

    macro = {
        k: float(np.mean([cell_results[c][k] for c in test_loaders]))
        for k in ["RMSE", "MAE", "MAPE (%)", "R2", "MaxE"]
    }
    cell_results["macro"] = macro
    csv_rows.append({"model": model_name, "dataset": "NASA", "cell": "MACRO_AVG", **macro})
    print(f"  {'MACRO_AVG':12s} | RMSE: {macro['RMSE']:.5f} | MAE: {macro['MAE']:.5f} | R2: {macro['R2']:.5f}")

    csv_path = subdirs["metrics"] / f"{model_name}_NASA_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path = subdirs["reports"] / f"{model_name}_NASA_report.json"
    json_path.write_text(json.dumps(cell_results, indent=2), encoding="utf-8")

    concat_t = np.concatenate(all_true)
    concat_p = np.concatenate(all_pred)
    plot_parity(concat_t, concat_p, f"{model_name} NASA Parity Plot", subdirs["plots"] / f"{model_name}_NASA_parity.png")

    return cell_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Central runner for baseline models.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", *BASELINE_NAMES],
        help="Baseline model to run (default: all).",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs (default: 80).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    args = parser.parse_args()

    models_to_run = BASELINE_NAMES if args.model == "all" else [args.model]

    print("==================================================")
    print(f"BASELINE EXPERIMENT RUNNER")
    print(f"Models to evaluate: {models_to_run}")
    print("==================================================")

    from Experiment.utils.paths import RESULT_TABLES_DIR
    all_summary = []
    for model_name in models_to_run:
        res = train_and_eval_single_baseline(
            model_name=model_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        macro = res["macro"]
        all_summary.append({
            "Model": model_name,
            "Macro RMSE": macro["RMSE"],
            "Macro MAE": macro["MAE"],
            "Macro R2": macro["R2"],
            "Macro MAPE (%)": macro["MAPE (%)"],
            "Macro MaxE": macro["MaxE"],
        })

    summary_csv = RESULT_TABLES_DIR / "baseline_comparison_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summary[0].keys()))
        writer.writeheader()
        writer.writerows(all_summary)

    print("\n==================================================")
    print("BASELINE BENCHMARK LEADERBOARD (Macro RMSE):")
    for row in sorted(all_summary, key=lambda x: x["Macro RMSE"]):
        print(f"  {row['Model']:14s} | RMSE: {row['Macro RMSE']:.5f} | MAE: {row['Macro MAE']:.5f} | R2: {row['Macro R2']:.5f}")
    print(f"\nFull comparison table saved to: {summary_csv}")
    print("==================================================")


if __name__ == "__main__":
    main()
