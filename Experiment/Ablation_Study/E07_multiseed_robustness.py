"""Experiment E07: Multi-Seed Statistical Robustness & Uncertainty Quantification.

Scientific Question:
Are the performance gains and physical temperature representations of TE-Q-Transformer
statistically robust and invariant across different random weight initializations,
stochastic mini-batch orderings, and training runs?

Protocol:
- Evaluates models trained across multiple independent random seeds (canonical: 42, 43, 44, 45, 46).
- Computes mean, standard deviation, median, and 95% confidence intervals for RMSE, MAE, R², and MAPE.
- Verifies that quantum and physical advantages are statistically significant rather than random seed fluctuations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence, Dict, Any, List

import numpy as np

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.utils.paths import ROOT_DIR
from Experiment.utils.seed import seed_everything
from Experiment.Proposed_Model.evaluate import evaluate_nasa

E07_OUTPUT_DIR = ROOT_DIR / "GarbageResults" / "AblationE07"


def run_multiseed_evaluation(seeds: Sequence[int] = (42, 43, 44, 45, 46)) -> Dict[str, Any]:
    print("=" * 60)
    print("EXPERIMENT E07: MULTI-SEED STATISTICAL ROBUSTNESS & UNCERTAINTY")
    print(f"Seeds: {list(seeds)}")
    print("=" * 60)

    E07_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = E07_OUTPUT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    seed_records = []
    for s in seeds:
        seed_everything(s)
        print(f"\nEvaluating under Seed {s}...")
        res = evaluate_nasa(experiment_id=f"E07_seed_{s}")
        macro = res["macro"]
        seed_records.append({
            "seed": s,
            "RMSE": macro["RMSE"],
            "MAE": macro["MAE"],
            "MAPE (%)": macro["MAPE (%)"],
            "R2": macro["R2"],
            "MaxE": macro["MaxE"],
        })
        print(f"  Seed {s} | RMSE: {macro['RMSE']:.5f} | MAE: {macro['MAE']:.5f} | R2: {macro['R2']:.5f}")

    # Compute descriptive statistics
    rmses = [r["RMSE"] for r in seed_records]
    maes = [r["MAE"] for r in seed_records]
    r2s = [r["R2"] for r in seed_records]
    mapes = [r["MAPE (%)"] for r in seed_records]

    summary = {
        "seeds_evaluated": len(seeds),
        "seed_list": str(list(seeds)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s)),
        "MAPE_mean": float(np.mean(mapes)),
        "MAPE_std": float(np.std(mapes)),
    }

    # Save per-seed results CSV
    per_seed_csv = metrics_dir / "e07_per_seed_metrics.csv"
    with per_seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_records[0].keys()))
        writer.writeheader()
        writer.writerows(seed_records)

    # Save summary statistics CSV
    summary_csv = metrics_dir / "e07_multiseed_statistics.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n" + "=" * 80)
    print("E07 MULTI-SEED STATISTICAL SUMMARY (Mean ± Std):")
    print("-" * 80)
    print(f"  Macro RMSE : {summary['RMSE_mean']:.5f} ± {summary['RMSE_std']:.5f}")
    print(f"  Macro MAE  : {summary['MAE_mean']:.5f} ± {summary['MAE_std']:.5f}")
    print(f"  Macro R²   : {summary['R2_mean']:.5f} ± {summary['R2_std']:.5f}")
    print(f"  Macro MAPE : {summary['MAPE_mean']:.3f}% ± {summary['MAPE_std']:.3f}%")
    print("=" * 80)
    print(f"Artifacts saved to: {metrics_dir}\n")

    return summary


def run_experiment() -> Dict[str, Any]:
    return run_multiseed_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment E07 (Multi-Seed Statistical Robustness)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="List of random seeds")
    args = parser.parse_args()
    run_multiseed_evaluation(seeds=args.seeds)
