"""Experiment E12: Sensor Noise Robustness & Perturbation Analysis.

Scientific Question:
How resilient is the TE-Q-Transformer against sensor noise (Gaussian jitter)
in voltage, current, and temperature measurements during real-world battery deployment?
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from Experiment.utils.paths import NASA_DATA_DIR, NASA_FROZEN_CHECKPOINT_PATH, get_result_subdirs
from Experiment.utils.metrics import compute_metrics
from Experiment.Dataset.nasa import get_nasa_dataloaders
from Experiment.Proposed_Model.model import TEQTransformer, rich_entangler_config


def run_experiment(noise_levels: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)) -> dict:
    print("\n==================================================")
    print("EXPERIMENT E12: SENSOR NOISE ROBUSTNESS")
    print(f"Noise Levels (std): {noise_levels}")
    print("==================================================")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = TEQTransformer(rich_entangler_config()).to(device)
    model.load_state_dict(torch.load(NASA_FROZEN_CHECKPOINT_PATH, map_location=device))
    model.eval()

    _, test_loaders, _ = get_nasa_dataloaders(batch_size=8)
    subdirs = get_result_subdirs("NASA")

    results = {}
    for sigma in noise_levels:
        all_true, all_pred = [], []
        with torch.no_grad():
            for cell_id, loader in test_loaders.items():
                for bx, by in loader:
                    # Inject Gaussian noise into inputs
                    noise = torch.randn_like(bx) * sigma if sigma > 0 else 0
                    noisy_bx = (bx + noise).to(device)
                    pred = model(noisy_bx)
                    all_true.append(by.numpy().reshape(-1))
                    all_pred.append(pred.cpu().numpy().reshape(-1))

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        m = compute_metrics(y_true, y_pred)
        results[f"noise_{sigma}"] = m
        print(f"  Noise sigma={sigma:.2f} | RMSE: {m['RMSE']:.5f} | MAE: {m['MAE']:.5f} | R2: {m['R2']:.5f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_experiment()
