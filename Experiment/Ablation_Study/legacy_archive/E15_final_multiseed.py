"""Experiment E15: Multi-Seed Statistical Validation.

Scientific Question:
Are the performance gains statistically robust across different random weight
initializations and stochastic mini-batch orderings?
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Sequence
import numpy as np

from Experiment.utils.paths import RESULT_TABLES_DIR, RESULT_NASA_DIR
from Experiment.utils.seed import seed_everything
from Experiment.Proposed_Model.evaluate import evaluate_nasa


def run_experiment(seeds: Sequence[int] = (42, 43, 44, 45, 46)) -> dict:
    print("\n==================================================")
    print("EXPERIMENT E15: MULTI-SEED STATISTICAL VALIDATION")
    print(f"Random Seeds: {list(seeds)}")
    print("==================================================")

    seed_results = []
    for s in seeds:
        seed_everything(s)
        # Evaluates baseline under specified seed
        res = evaluate_nasa(experiment_id=f"E15_seed_{s}")
        seed_results.append(res["macro"])

    rmses = [r["RMSE"] for r in seed_results]
    maes = [r["MAE"] for r in seed_results]
    r2s = [r["R2"] for r in seed_results]

    summary = {
        "seeds": list(seeds),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "R2_mean": float(np.mean(r2s)),
        "R2_std": float(np.std(r2s)),
    }

    print("\n[E15 Multi-Seed Summary (Mean +/- Std)]")
    print(f"  RMSE: {summary['RMSE_mean']:.5f} +/- {summary['RMSE_std']:.5f}")
    print(f"  MAE : {summary['MAE_mean']:.5f} +/- {summary['MAE_std']:.5f}")
    print(f"  R2  : {summary['R2_mean']:.5f} +/- {summary['R2_std']:.5f}")

    out_csv = RESULT_TABLES_DIR / "E15_multiseed_statistical_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_experiment()
