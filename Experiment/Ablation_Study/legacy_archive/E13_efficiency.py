"""Experiment E13: Computational Efficiency & Latency Benchmarking.

Scientific Question:
What is the computational overhead (inference latency per window, parameter count,
memory footprint) of the hybrid quantum layer compared to pure classical models?
"""

from __future__ import annotations

import argparse
import time
import torch

from Experiment.Proposed_Model.model import TEQTransformer, rich_entangler_config
from Experiment.Baseline.Transformer import TransformerModel
from Experiment.Baseline.LSTM import LSTMModel


def run_experiment(num_runs: int = 20) -> dict:
    print("\n==================================================")
    print("EXPERIMENT E13: COMPUTATIONAL EFFICIENCY BENCHMARK")
    print("==================================================")

    dummy_input = torch.randn(1, 512, 4)
    models = {
        "TE-Q-Transformer": TEQTransformer(rich_entangler_config()),
        "Transformer (Classical)": TransformerModel(),
        "LSTM": LSTMModel(),
    }

    results = {}
    for name, model in models.items():
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)

        # Benchmark CPU inference latency
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                latencies.append(time.perf_counter() - t0)

        mean_lat_ms = float(np.mean(latencies) * 1000.0)
        std_lat_ms = float(np.std(latencies) * 1000.0)

        results[name] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "latency_ms_per_window": mean_lat_ms,
            "latency_std_ms": std_lat_ms,
        }
        print(f"  {name:24s} | Params: {trainable_params:,} | Latency: {mean_lat_ms:.2f} ms/window")

    return results


if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_experiment()
