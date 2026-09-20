"""Ablation studies isolating components and benchmarks of TE-Q-Transformer (lazy loaded).

Aligned strictly with the V2 publication experimental plan:
| Phase   | Experiment                             | Description                                         |
|---------|----------------------------------------|-----------------------------------------------------|
| **E01** | Baseline Reproduction                  | Verifies reproducibility of proposed TE-Q model     |
| **E02** | Conventional T vs Physics-guided T     | Contribution of physics-guided Arrhenius encoding   |
| **E03** | SEI vs Plating vs Combined             | Isolates degradation mechanisms (SEI, Plating, Both)|
| **E04** | Classical vs Quantum                   | Isolates quantum representation predictive benefit  |
| **E05** | 10+ Baseline Comparison                | Fair benchmark against 10 modern baseline models    |
| **E06** | Cross-cell Generalization              | Unseen battery cell & zero-shot cross-dataset transfer|
| **E07** | Multi-seed Robustness + Statistics     | Statistical significance across independent seeds   |
"""

from __future__ import annotations
from typing import Callable, Dict

ABLATION_EXPERIMENT_NAMES = [
    "E01",
    "E02",
    "E03",
    "E04",
    "E05",
    "E06",
    "E07",
]


def get_ablation_runner(exp_id: str) -> Callable[[], dict]:
    """Dynamically loads and returns the runner function for an ablation experiment."""
    if exp_id == "E01":
        from Experiment.Ablation_Study.E01_baseline_reproduction import run_experiment
        return run_experiment
    elif exp_id == "E02":
        from Experiment.Ablation_Study.E02_raw_vs_physics import run_experiment
        return run_experiment
    elif exp_id == "E03":
        from Experiment.Ablation_Study.E03_physics_mechanisms import run_experiment
        return run_experiment
    elif exp_id == "E04":
        from Experiment.Ablation_Study.E04_classical_vs_quantum import run_experiment
        return run_experiment
    elif exp_id == "E05":
        from Experiment.Ablation_Study.E05_baseline_comparison import run_experiment
        return run_experiment
    elif exp_id == "E06":
        from Experiment.Ablation_Study.E06_cross_cell import run_experiment
        return run_experiment
    elif exp_id == "E07":
        from Experiment.Ablation_Study.E07_multiseed_robustness import run_experiment
        return run_experiment
    else:
        raise ValueError(
            f"Unknown experiment ID '{exp_id}'. Available: {ABLATION_EXPERIMENT_NAMES}"
        )


EXPERIMENT_REGISTRY: Dict[str, Callable[[], dict]] = {
    exp_id: lambda _id=exp_id: get_ablation_runner(_id)() for exp_id in ABLATION_EXPERIMENT_NAMES
}

__all__ = ["ABLATION_EXPERIMENT_NAMES", "get_ablation_runner", "EXPERIMENT_REGISTRY"]
