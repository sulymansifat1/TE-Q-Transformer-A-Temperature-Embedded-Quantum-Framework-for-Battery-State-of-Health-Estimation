# Experimental Results & Provenance

This directory stores the empirical results, evaluation logs, and publication figures for **TE-Q-Transformer** and contemporary baselines.

---

## 1. Directory Structure

- `tables/`: Clean CSV tables recording model performance across all experimental campaigns:
  - `all_models_benchmark_metrics.csv`: Locked 10-baseline contemporary comparison under identical NASA protocol.
  - `proposed_model_reference.csv`: TE-Q-Transformer authoritative performance ($\text{RMSE} = 0.0168$, $\text{MAE} = 0.0141$, $R^2 = 0.870$).
  - `nasa_calce_reproduction_summary.csv`: Macro and per-cell reproduction figures for both NASA and CALCE.
  - `ablation_raw_vs_physics.csv`: E02 ablation comparing conventional temperature scaling vs physics-guided Arrhenius encoding (16.5% error reduction).
  - `ablation_mechanisms.csv`: E03 ablation separating SEI-only, Plating-only, and Combined mechanisms.
  - `ablation_classical_vs_quantum.csv`: E04 ablation comparing the 4-qubit quantum feature map vs a parameter-matched classical MLP (Macro RMSE 0.0215 vs 0.0431).
  - `generalization_model_metrics.csv` & `generalization_cell_metrics.csv`: Unseen cell (`B0018`, `B0032`) and temporal extrapolation (`B0053_test`) evaluations across all models.
  - `calce_zero_shot_metrics.csv`: External zero-shot transfer metrics on CALCE CS2 cells (`CS2_35`–`CS2_38`), reporting honest cross-chemistry limitation (pooled $\text{RMSE} = 0.3325$, $R^2 = -1.789$).

- `figures/`: High-resolution degradation trajectory tracking and parity plots:
  - `soh_trajectory_E01_baseline_reproduction_*.png`: Predicted vs actual SOH trajectories.
  - `E01_baseline_reproduction_NASA_parity.png`: Parity scatter plot across NASA test cells.
  - `E01_baseline_reproduction_CALCE_parity.png`: Zero-shot transfer parity scatter plot.

- `logs/`: Machine-readable JSON reports detailing runtime hardware, seeds, and per-epoch telemetry.

---

## 2. Key Scientific Findings Summary

### Main NASA Benchmark
| Model | Family | Macro RMSE | Macro MAE | Macro $R^2$ | Trainable Params |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TE-Q-Transformer (Proposed)** | Quantum-Physics-Transformer | **0.0168** | **0.0141** | **0.870** | 92,554 |
| QNN-GRU | Quantum-Recurrent | 0.0304 | 0.0248 | 0.258 | 29,533 |
| iTransformer | Attention (Inverted) | 0.0341 | 0.0298 | 0.533 | 137,473 |
| Transformer | Attention | 0.0346 | 0.0287 | 0.218 | 80,257 |
| PatchTST | Attention (Patch) | 0.0387 | 0.0333 | 0.321 | 109,962 |
| LSTM | Recurrent | 0.0416 | 0.0367 | -0.076 | 71,105 |
| QLSTM | Quantum-Recurrent | 0.0445 | 0.0386 | -0.336 | 37,853 |
| CNN1D | Convolutional | 0.0509 | 0.0457 | -3.191 | 47,041 |
| TCN | Convolutional (Dilated) | 0.0535 | 0.0452 | -0.341 | 91,841 |
| DLinear | Linear (Decomposition) | 0.0624 | 0.0568 | -2.232 | 1,031 |
| GRU | Recurrent | 0.0642 | 0.0565 | -1.680 | 54,465 |

### Physics-Guided Temperature Encoding Gain (Ablation E02)
- Conventional Raw Temperature Macro RMSE: **0.0201** ($R^2 = 0.723$)
- Physics-Guided Arrhenius Temperature Macro RMSE: **0.0168** ($R^2 = 0.870$)
- **Relative Error Reduction: 16.5%**

### Quantum Circuit Feature Mapping Gain (Ablation E04)
- Parameter-Matched Classical MLP Macro RMSE: **0.0431** ($R^2 = -0.746$)
- 4-Qubit Variational Quantum Circuit Macro RMSE: **0.0215** ($R^2 = 0.610$)
