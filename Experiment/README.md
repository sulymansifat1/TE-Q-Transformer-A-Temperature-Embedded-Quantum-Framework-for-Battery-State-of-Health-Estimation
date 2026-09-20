# TE-Q-Transformer V2 — Experiment Framework

Welcome to the reviewer-friendly, modular Python experiment framework for the **TE-Q-Transformer** research project.

This directory contains the self-contained, reproducible Python implementation of all datasets, models, baselines, and ablation studies. It replaces legacy notebook workflows with clean, deterministic CLI commands.

---

## 1. Directory Overview

```
Experiment/
│
├── Dataset/                     <- Standardized loaders, splits, and scalers (NASA & CALCE)
│   ├── nasa.py                  <- NASA battery loader & multi-temperature DataLoaders
│   ├── calce.py                 <- CALCE CS2 external validation loader
│   ├── preprocessing.py         <- SOH normalization & MinMax feature transformation
│   ├── splits.py                <- Canonical split constants & cell groupings
│   └── README.md
│
├── Proposed_Model/              <- Proposed TE-Q-Transformer implementation
│   ├── model.py                 <- Authoritative PennyLane + PyTorch architecture
│   ├── train.py                 <- Training loop with AdamW, ReduceLROnPlateau, MSELoss
│   ├── evaluate.py              <- Evaluation engine on NASA & CALCE
│   ├── config.py                <- Architectural and training hyperparameter configurations
│   ├── run.py                   <- Proposed model CLI runner
│   └── README.md
│
├── Baseline/                    <- 10+ baseline benchmark models
│   ├── LSTM.py                  <- Standard LSTM
│   ├── GRU.py                   <- Standard GRU
│   ├── BiLSTM.py                <- Bidirectional LSTM
│   ├── CNN1D.py                 <- 1D Temporal CNN
│   ├── TCN.py                   <- Temporal Convolutional Network with causal dilations
│   ├── Transformer.py           <- Classical Transformer Encoder without quantum layer
│   ├── Informer.py              <- Informer with self-attention & distillation
│   ├── PatchTST.py              <- Patch Time Series Transformer
│   ├── QNN_GRU.py               <- Quantum Neural Network + GRU (PennyLane)
│   ├── QLSTM.py                 <- Quantum LSTM (PennyLane)
│   ├── run.py                   <- Central runner for all baselines
│   └── README.md
│
├── Ablation_Study/              <- Experiments E01 through E15
│   ├── E01_baseline_reproduction.py    <- Frozen baseline reproduction
│   ├── E02_raw_vs_physics.py           <- Raw temperature vs Arrhenius physics
│   ├── E03_sei_only.py                 <- SEI layer growth mechanism only
│   ├── E04_plating_only.py             <- Lithium plating mechanism only
│   ├── E05_combined_physics.py         <- Dual-mechanism SEI + Plating physics
│   ├── E06_fixed_vs_trainable_ea.py    <- Fixed vs Trainable activation energies
│   ├── E07_classical_vs_quantum.py     <- Classical MLP embedding vs 4-qubit circuit
│   ├── E08_factorial_analysis.py       <- 2x2x2 Factorial interaction analysis
│   ├── E09_cross_cell.py               <- Leave-one-cell-out cross-validation
│   ├── E10_temporal_extrapolation.py   <- Early-to-late life cycle forecasting
│   ├── E11_unseen_temperature.py       <- Generalization across 4°C, 24°C, 44°C
│   ├── E12_robustness.py               <- Sensor noise perturbation analysis
│   ├── E13_efficiency.py               <- Latency (ms/window), FLOPs, and params
│   ├── E14_architectural_ablation.py   <- Entangler, smoother, pos enc, pooling
│   ├── E15_final_multiseed.py          <- 5-seed statistical validation (mean +/- std)
│   ├── run.py                          <- Central runner for all ablations
│   └── README.md
│
├── Result/                      <- Unified empirical output storage
│   ├── NASA/                    <- Metrics, plots, predictions, reports, checkpoints
│   ├── CALCE/                   <- Zero-shot metrics, plots, predictions, reports
│   ├── tables/                  <- Comparative summary CSV tables
│   ├── figures/                 <- Publication figures
│   └── README.md
│
├── utils/                       <- Shared utilities (paths, seed, metrics, plotting, logging)
│   ├── paths.py
│   ├── seed.py
│   ├── metrics.py
│   ├── plotting.py
│   └── logging.py
│
├── run.py                       <- Master CLI orchestrator
├── README.md                    <- This document
└── CONVERSION_AUDIT.md          <- Scientific audit and mapping confirmation
```

---

## 2. Quickstart: Reviewer Commands

### View Help
```bash
python Experiment/run.py --help
```

### 1. Proposed Model (TE-Q-Transformer)
```bash
# Evaluate pre-trained frozen checkpoint on NASA & CALCE
python Experiment/run.py --group proposed

# Evaluate on NASA test cells only
python Experiment/run.py --group proposed --dataset NASA

# Evaluate on CALCE zero-shot only
python Experiment/run.py --group proposed --dataset CALCE

# Train model from scratch
python Experiment/run.py --group proposed --train --epochs 80 --batch-size 8
```

### 2. Baseline Models (10+ Model Suite)
```bash
# Run all 10 baselines sequentially
python Experiment/run.py --group baseline --model all

# Run specific baseline models
python Experiment/run.py --group baseline --model LSTM
python Experiment/run.py --group baseline --model GRU
python Experiment/run.py --group baseline --model Transformer
python Experiment/run.py --group baseline --model Informer
python Experiment/run.py --group baseline --model PatchTST
python Experiment/run.py --group baseline --model QNN_GRU
python Experiment/run.py --group baseline --model QLSTM
```

### 3. Ablation Studies (E01 – E15)
```bash
# Run all 15 ablation experiments
python Experiment/run.py --group ablation --experiment all

# Run specific ablation studies
python Experiment/run.py --group ablation --experiment E01  # Baseline reproduction
python Experiment/run.py --group ablation --experiment E02  # Raw vs physics temperature
python Experiment/run.py --group ablation --experiment E07  # Classical vs quantum
python Experiment/run.py --group ablation --experiment E12  # Sensor noise robustness
python Experiment/run.py --group ablation --experiment E13  # Efficiency & latency benchmark
python Experiment/run.py --group ablation --experiment E15  # 5-seed statistical validation
```

---

## 3. Scientific Invariants Guaranteed

- **Input Dimension & Length**: Exactly $[B, 512, 4]$ with $[V, I, T_C, t_{\text{norm}}]$.
- **Temperature Handling**: $T_C$ strictly kept in physical Celsius for the Arrhenius gate ($T_K = T_C + 273.15$).
- **Data Splitting**: Exact multi-temperature cross-cell split (Train: B0005, B0006, B0007, B0029, B0030, B0031, B0053_70%; Test: B0018, B0032, B0053_30%).
- **Leakage Prevention**: Scaler fitted exclusively on NASA train split; CALCE evaluated zero-shot without refitting.
- **Metrics**: Standard RMSE, MAE, MAPE (%), R2, MaxE with consistent formulas.
- **Portability**: All file paths are resolved relative to repository root via `pathlib.Path`.
