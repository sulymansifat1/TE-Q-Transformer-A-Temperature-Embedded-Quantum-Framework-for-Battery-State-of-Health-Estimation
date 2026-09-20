"""Experiment E14: Architectural Component Ablation Study (from notebook).

Scientific Question:
What is the individual contribution of the rich entangler circuit, Pauli feature map,
Conv1D temporal smoother, positional encoding, and [CLS] token pooling?
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from Experiment.Proposed_Model.model import TEQTransformerConfig, rich_entangler_config
from Experiment.Proposed_Model.evaluate import evaluate_nasa


def get_architectural_variants() -> dict[str, TEQTransformerConfig]:
    """Generates the 8 canonical architectural variants from the ablation notebook."""
    base_cfg = rich_entangler_config()
    return {
        "01_rich_entangler (baseline)": base_cfg,
        "02_entangler_basic_only": replace(base_cfg, entangler_type="basic"),
        "03_with_pauli_feature_map": replace(base_cfg, use_pauli_feature_map=True),
        "04_no_temporal_conv": replace(base_cfg, use_temporal_smooth=False),
        "05_with_gru_smoother": replace(base_cfg, use_gru_smoother=True, use_temporal_smooth=False),
        "06_no_positional_encoding": replace(base_cfg, use_positional_encoding=False),
        "07_mean_pooling_no_cls": replace(base_cfg, use_cls_token=False, pooling="mean"),
        "08_with_residual_mlp": replace(base_cfg, use_residual_mlp=True),
    }


def run_experiment() -> dict:
    print("\n==================================================")
    print("EXPERIMENT E14: ARCHITECTURAL COMPONENT ABLATION")
    print("==================================================")
    variants = get_architectural_variants()
    print(f"Total architectural variants: {len(variants)}")
    for name, cfg in variants.items():
        print(f"  - {name}")

    # Evaluate primary baseline configuration
    metrics = evaluate_nasa(experiment_id="E14_architectural_ablation")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_experiment()
