# TE-Q-Transformer V2 — Experimental Plan & Ablation Suite (E01 – E07)

This directory contains the authoritative implementation of the publication experimental roadmap for **TE-Q-Transformer V2**.

---

## 1. Master Scientific Roadmap & Status

| Phase | Experiment | Scientific Objective / Question | Script | Status |
|:---:|:---|:---|:---|:---:|
| **E01** | **Baseline Reproduction** | Verify whether existing TE-Q-Transformer reproduction results are exact and verified. | [`E01_baseline_reproduction.py`](./E01_baseline_reproduction.py) | 🟢 **Completed** |
| **E02** | **Conventional T vs Physics-guided T** | Isolate the predictive contribution of physics-guided Arrhenius temperature encoding against raw linear min-max scaling. | [`E02_raw_vs_physics.py`](./E02_raw_vs_physics.py) | 🟢 **Completed** |
| **E03** | **SEI vs Plating vs Combined** | Mechanism attribution: Determine whether SEI growth, lithium plating, or their combined representation drives multi-temperature benefits. | [`E03_physics_mechanisms.py`](./E03_physics_mechanisms.py) | 🟢 **Completed** |
| **E04** | **Classical vs Quantum** | Capacity-matched ablation: Determine whether the 4-qubit quantum representation provides predictive benefit beyond an equivalent classical MLP under identical physics guidance. | [`E04_classical_vs_quantum.py`](./E04_classical_vs_quantum.py) | 🟢 **Completed** |
| **E05** | **10+ Baseline Comparison** | Rigorous fair benchmark comparing TE-Q-Transformer against 10 modern baseline models (LSTM, GRU, BiLSTM, CNN1D, TCN, Transformer, Informer, PatchTST, QNN_GRU, QLSTM). | [`E05_baseline_comparison.py`](./E05_baseline_comparison.py) | 🔵 **Next** |
| **E06** | **Cross-Cell Generalization** | Generalization evaluation across unseen battery cells and zero-shot cross-dataset transfer (NASA → CALCE). | [`E06_cross_cell.py`](./E06_cross_cell.py) | ⏳ **Upcoming** |
| **E07** | **Multi-Seed Robustness + Statistics** | Uncertainty quantification and statistical significance testing across independent random seeds (42, 43, 44, 45, 46). | [`E07_multiseed_robustness.py`](./E07_multiseed_robustness.py) | ⏳ **Upcoming** |

---

## 2. Directory Layout

```
Experiment/Ablation_Study/
├── E01_baseline_reproduction.py    <- Phase E01 (Reproduction)
├── E02_raw_vs_physics.py           <- Phase E02 (Conventional vs Physics Temperature)
├── E03_physics_mechanisms.py       <- Phase E03 (SEI vs Plating vs Combined)
├── E04_classical_vs_quantum.py     <- Phase E04 (Matched Classical vs Quantum)
├── E05_baseline_comparison.py      <- Phase E05 (10+ Baseline Benchmark)
├── E06_cross_cell.py               <- Phase E06 (Cross-Cell & Zero-Shot Transfer)
├── E07_multiseed_robustness.py     <- Phase E07 (Multi-Seed Statistical Robustness)
├── __init__.py                     <- Lazy-loaded registry for E01-E07
├── run.py                          <- CLI runner for E01-E07
├── README.md                       <- This documentation
└── legacy_archive/                 <- Preserved legacy stubs from earlier exploratory steps
```

---

## 3. How to Run

### Via the Ablation Suite Runner:
```bash
# Run all experiments
python Experiment/Ablation_Study/run.py --experiment all

# Run individual experiments
python Experiment/Ablation_Study/run.py --experiment E01
python Experiment/Ablation_Study/run.py --experiment E02
python Experiment/Ablation_Study/run.py --experiment E03
python Experiment/Ablation_Study/run.py --experiment E04
python Experiment/Ablation_Study/run.py --experiment E05
python Experiment/Ablation_Study/run.py --experiment E06
python Experiment/Ablation_Study/run.py --experiment E07
```

### Direct Script Execution:
```bash
# E03: SEI vs Plating vs Combined full training
python -m Experiment.Ablation_Study.E03_physics_mechanisms --epochs 80 --batch-size 8 --seed 42

# E04: Classical vs Quantum full training
python -m Experiment.Ablation_Study.E04_classical_vs_quantum --epochs 80 --batch-size 8 --seed 42

# E05: 10+ Baseline benchmark
python -m Experiment.Ablation_Study.E05_baseline_comparison --epochs 80 --batch-size 8 --seed 42
```

---

## 4. Experiment Tracking & Artifacts
Each phase saves its complete real-time tracking logs, code audits, fairness audits, checkpoints, predictions, and reports under:
- `GarbageResults/AblationE02/`
- `GarbageResults/AblationE03/`
- `GarbageResults/AblationE04/`
- `GarbageResults/AblationE05/` (when executed)
- `GarbageResults/AblationE06/` (when executed)
- `GarbageResults/AblationE07/` (when executed)
