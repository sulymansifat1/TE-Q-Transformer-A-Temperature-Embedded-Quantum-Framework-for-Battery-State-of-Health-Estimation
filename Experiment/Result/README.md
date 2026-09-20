# Experiment Result Directory

This directory stores all empirical artifacts generated during model training, baseline benchmarking, ablation studies, and cross-dataset evaluation.

---

## 1. Directory Structure

```
Experiment/Result/
├── NASA/
│   ├── metrics/       <- CSV metric files (RMSE, MAE, MAPE, R2, MaxE) per cell and macro
│   ├── plots/         <- PNG trajectory overlays and parity plots
│   ├── predictions/   <- Numpy .npy true and predicted SOH arrays
│   ├── reports/       <- Detailed JSON execution summaries
│   └── checkpoints/   <- Saved PyTorch model checkpoints (.pth)
│
├── CALCE/
│   ├── metrics/       <- Zero-shot evaluation CSV metric files
│   ├── plots/         <- Degradation trajectory overlays and parity plots
│   ├── predictions/   <- Numpy .npy prediction arrays
│   ├── reports/       <- Detailed JSON reports
│   └── checkpoints/   <- Model checkpoints
│
├── tables/            <- Publication-ready summary comparison tables (.csv)
├── figures/           <- High-resolution publication figures
└── README.md          <- This directory guide
```

---

## 2. Naming Conventions

All result files follow predictable, reproducible naming conventions:

### Metrics CSV
- Proposed model: `Result/NASA/metrics/E01_TEQ_NASA_metrics.csv`
- Baselines: `Result/NASA/metrics/{model}_NASA_metrics.csv` (e.g. `LSTM_NASA_metrics.csv`)
- Ablations: `Result/NASA/metrics/{experiment}_NASA_metrics.csv` (e.g. `E02_raw_vs_physics_NASA_metrics.csv`)
- CALCE Zero-Shot: `Result/CALCE/metrics/{experiment}_CALCE_metrics.csv`

### Plots
- Trajectory: `soh_trajectory_{experiment}_{cell}.png`
- Parity: `{experiment}_{dataset}_parity.png`
- Loss curves: `training_loss_curve.png`

### Predictions
- `{experiment}_{cell}_y_true.npy`
- `{experiment}_{cell}_y_pred.npy`

### Summary Tables
- `Result/tables/baseline_comparison_summary.csv`
- `Result/tables/E15_multiseed_statistical_summary.csv`
