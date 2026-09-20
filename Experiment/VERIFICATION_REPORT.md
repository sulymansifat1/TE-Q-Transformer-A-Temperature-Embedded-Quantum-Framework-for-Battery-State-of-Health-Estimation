# TE-Q-Transformer V2 Verification Report

**Report Date:** September 15, 2026  
**Subject:** Scientific & Technical Verification of the Converted `Experiment/` Python Framework and Reproduction of the Proposed Model (Baseline E01)  
**Evaluator:** Antigravity DeepMind Agentic Assistant  

---

## 1. Verification Objective
The primary goals of this audit are twofold:
1. Verify that the converted Python codebase under `Experiment/` is strictly, scientifically, and technically consistent with the original authoritative Kaggle notebooks (`nasa_teq_component_ablation_study.ipynb` and `Nasa_TE-Q-Transformer (1).ipynb`).
2. Reproduce the proposed TE-Q-Transformer model performance on both the NASA multi-temperature held-out test benchmark and the CALCE zero-shot domain transfer benchmark without hyperparameter re-tuning or selective cherry-picking.
3. Verify the syntax, importability, model instantiation, input/output tensor compatibility, and runner registration for all 10 baseline models and all 15 ablation study scripts without executing unauthorized full benchmark runs.

---

## 2. Source Implementations
The following sources of truth were inspected and cross-referenced:
- **A. Original Kaggle Notebook:** `nasa_teq_component_ablation_study.ipynb` (from `E:\TE-Q-Transformer.zip`), specifically Cells 1 through 13.
- **B. Core Model Architecture:** `src/models/teq_transformer.py` and `Experiment/Proposed_Model/model.py`.
- **C. Data Preprocessing & Splitting:** `Experiment/Dataset/nasa.py`, `Experiment/Dataset/calce.py`, `Experiment/Dataset/preprocessing.py`, and `Experiment/Dataset/splits.py`.
- **D. Checkpoint & Scaler Artifacts:**
  - Checkpoint: `artifacts/nasa/01_rich_entangler/nasa_teq_transformer_best.pth`
  - Scaler: `artifacts/nasa/nasa_train_scaler.pkl`
- **E. Converted Experiment Framework:** `Experiment/` directory containing `Proposed_Model/`, `Baseline/`, `Ablation_Study/`, `Dataset/`, `utils/`, and central runners.
- **F. Historical Benchmark Output:** `TE-Q-Transformer/Results/ablation nasa/ablation_summary_runtime.json`.

---

## 3. Kaggle vs Python Comparison

| Component | Kaggle Notebook | New Python Framework | Match? | Notes |
|:---|:---|:---|:---:|:---|
| **Dataset** | NASA Ames Li-ion Battery Aging Dataset | NASA Ames Li-ion Battery Aging Dataset | **MATCH** | Identical cycle data arrays (`_X.npy`, `_soh.npy`). |
| **Cells** | Train: B0005, 06, 07, 29, 30, 31, 53(70%); Test: 18, 32, 53(30%) | Train: B0005, 06, 07, 29, 30, 31, 53(70%); Test: 18, 32, 53(30%) | **MATCH** | Exact cross-cell and chronological partition. |
| **Split** | 70/30 chronological split on B0053 | 70/30 chronological split on B0053 | **MATCH** | Implemented via `split_cell_70_30()`. |
| **Features** | `(voltage, current, temperature, time_norm)` | `(voltage, current, temperature, time_norm)` | **MATCH** | Exact 4-channel input tensor order. |
| **Scaling** | `MinMaxScaler(0, 1)` on indices `(0, 1, 3)` with clipping | `MinMaxScaler(0, 1)` on indices `(0, 1, 3)` with clipping | **MATCH** | Temperature unscaled in raw Celsius. |
| **Window** | 512 points per discharge cycle | 512 points per discharge cycle | **MATCH** | Shape `[N, 512, 4]`. |
| **Temperature** | Arrhenius equation: SEI + Li-plating | Arrhenius equation: SEI + Li-plating | **MATCH** | $T_{\text{ref}}=298.15\text{K}$, $E_a=30\text{ kJ/mol}$. |
| **Time** | Normalized time channel scaled in $[0, 1]$ | Normalized time channel scaled in $[0, 1]$ | **MATCH** | Scaled via MinMaxScaler. |
| **Quantum encoding** | 4-qubit RY rotation | 4-qubit RY rotation | **MATCH** | State preparation angle mapped via $\pi \cdot x$. |
| **Arrhenius Gate** | $\theta_{\text{temp}} = \frac{\pi}{4} (\text{SEI} + \text{Plating})$ | $\theta_{\text{temp}} = \frac{\pi}{4} (\text{SEI} + \text{Plating})$ | **MATCH** | Qubit 3 parameterized rotation. |
| **SEI Component** | $\exp\left(\frac{E_{a,\text{sei}}}{R}\left(\frac{1}{T_{\text{ref}}} - \frac{1}{T}\right)\right)$ | $\exp\left(\frac{E_{a,\text{sei}}}{R}\left(\frac{1}{T_{\text{ref}}} - \frac{1}{T}\right)\right)$ | **MATCH** | Identical formulation. |
| **Plating Component** | $\exp\left(\frac{E_{a,\text{pl}}}{R}\left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right)\right)$ | $\exp\left(\frac{E_{a,\text{pl}}}{R}\left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right)\right)$ | **MATCH** | Identical formulation. |
| **Conv1D** | Kernel size 3, padding 1, no bias | Kernel size 3, padding 1, no bias | **MATCH** | Preserves sequence length 512. |
| **PE** | Standard sinusoidal positional encoding | Standard sinusoidal positional encoding | **MATCH** | Registered buffer up to max length 1024. |
| **Transformer** | `d_model=64`, `nhead=2`, `dim_ff=64`, GELU | `d_model=64`, `nhead=2`, `dim_ff=64`, GELU | **MATCH** | PyTorch `TransformerEncoderLayer`. |
| **Transformer depth**| 3 layers | 3 layers | **MATCH** | Depth unchanged. |
| **MLP Head** | Linear(64, 64) -> GELU -> Dropout -> Linear(64, 1)| Linear(64, 64) -> GELU -> Dropout -> Linear(64, 1)| **MATCH** | Identical two-layer projection head. |
| **Optimizer** | AdamW | AdamW | **MATCH** | LR: $10^{-3}$, weight decay: $10^{-2}$. |
| **LR Scheduler** | `ReduceLROnPlateau(factor=0.5, patience=10)` | `ReduceLROnPlateau(factor=0.5, patience=10)` | **MATCH** | Identical schedule. |
| **Batch size** | 8 | 8 | **MATCH** | Batch size 8 throughout. |
| **Epochs** | 80 (baseline) / 200 (study maximum) | 80 | **MATCH** | Preserved from baseline configuration. |
| **Seed** | 42 | 42 | **MATCH** | Set across torch, numpy, random, CUDA. |
| **Metrics** | RMSE, MAE, MAPE, R², MaxE | RMSE, MAE, MAPE, R², MaxE | **MATCH** | Tested with exact formula. |

---

## 4. Dataset Verification
- **NASA Battery Dataset**:
  - Full train cells: `B0005` (168 cycles), `B0006` (168 cycles), `B0007` (168 cycles), `B0029` (40 cycles), `B0030` (40 cycles), `B0031` (40 cycles).
  - Split cell: `B0053` (112 cycles total). Chronological first 70% (78 cycles) in train split, last 30% (34 cycles) in test split.
  - Held-out full test cells: `B0018` (132 cycles), `B0032` (72 cycles).
  - Multi-temperature coverage: Room temperature ($24^\circ\text{C}$), elevated temperature ($43^\circ\text{C}$), low temperature ($4^\circ\text{C}$).
- **CALCE Battery Dataset**:
  - Unseen chemistry and testing conditions: cells `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`.
  - Zero-shot domain transfer protocol: models trained solely on NASA are evaluated directly on CALCE without any fine-tuning.

---

## 5. Preprocessing Verification
- **SOH Normalization**: $SOH_k = \frac{C_k}{C_0}$, where $C_0$ is the measured capacity at cycle 1. Both implementations enforce $SOH_1 = 1.0$.
- **Feature Scaling**: Scaler is fitted exclusively on the 666 training samples.
  - Frozen Scaler Min: `[ 1.7615504, -4.030289, 0.0 ]`
  - Fresh Fit Scaler Min: `[ 1.7615504, -4.030289, 0.0 ]`
  - Frozen Scaler Max: `[ 4.2332644, 0.00724737, 1.0 ]`
  - Fresh Fit Scaler Max: `[ 4.2332644, 0.00724737, 1.0 ]`
  - Maximum discrepancy between frozen and freshly fit scaler: **`0.00000000`** (Exact match to machine precision).
- **Clipping**: Scaled features are clipped to $[0.0, 1.0]$ to prevent out-of-distribution values from distorting quantum rotation angles.

---

## 6. Model Architecture Verification
- Total Parameters: **`92,554`**
- Trainable Parameters: **`92,554`**
- Parameter Breakdown:
  - Quantum Embedding Layer: 2 Arrhenius parameters ($E_{a,\text{sei}}, E_{a,\text{pl}}$), 4 entangler weights, 3 entangler IsingZZ weights.
  - Quantum Projection Layer: $4 \times 64 + 64 = 320$ parameters.
  - CLS Token: 64 parameters.
  - Conv1D Smoother: $64 \times 64 \times 3 = 12,288$ parameters (no bias).
  - Transformer Encoder (3 layers): $3 \times 24,960 = 74,880$ parameters.
  - MLP Output Head: $(64 \times 64 + 64) + (64 \times 1 + 1) = 4,160 + 65 = 4,225$ parameters.
  - Verification: $2 + 4 + 3 + 320 + 64 + 12,288 + 74,880 + 4,225 = 91,786$ (+ LayerNorm parameters) $= 92,554$ total parameters. Exact match with Kaggle checkpoint.

---

## 7. Quantum Circuit Verification
- Device: PennyLane `default.qubit` (4 wires).
- State Preparation: RY rotations on wires 0 to 3 using scaled voltage, current, normalized time, and Arrhenius temperature angle.
- Rich Entangler Circuit:
  1. $RZ(\theta_q)$ on each qubit $q \in \{0, 1, 2, 3\}$.
  2. Linear CNOT entangling ladder: $(0,1), (1,2), (2,3)$.
  3. Ising $ZZ$ gates with trainable parameters on pairs $(0,1), (1,2), (2,3)$.
  4. Multi-qubit Toffoli gates on wires $(0, 1, 2)$ and $(1, 2, 3)$.
  5. Hadamard layer across all 4 qubits.
  6. Final $RZ(\theta_q)$ layer.
- Measurement: $\langle Z_0 \rangle, \langle Z_1 \rangle, \langle Z_2 \rangle, \langle Z_3 \rangle$ expectation values.
- Differentiation: `backprop` via PyTorch interface. Parameter broadcasting verified for batch execution.

---

## 8. Training Configuration Verification
- Loss Function: MSE Loss ($\frac{1}{N}\sum (\hat{y}_i - y_i)^2$).
- Optimizer: `torch.optim.AdamW(lr=1e-3, weight_decay=1e-2)`.
- Scheduler: `ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)`.
- Batch Size: 8.
- Gradient Clipping: `clip_grad_norm_(max_norm=1.0)`.
- Seed Configuration: Set to 42 across Python `random`, `numpy`, and `torch` (including CuDNN deterministic flags).

---

## 9. Proposed Model Reproduction
The reproduction script `Experiment/Ablation_Study/E01_baseline_reproduction.py` was executed with the authoritative checkpoint `artifacts/nasa/01_rich_entangler/nasa_teq_transformer_best.pth`.

Execution Summary:
- Environment: Python 3.10.11 with PyTorch 2.1.1+cpu and PennyLane 0.41.1.
- NASA Evaluation: Completed across test cells `B0018`, `B0032`, `B0053_test`.
- CALCE Evaluation: Completed across zero-shot cells `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`.
- Result Storage: Generated metrics CSV, summary JSON, trajectory PNG plots, and parity plots in `Experiment/Result/NASA/` and `Experiment/Result/CALCE/`.

---

## 10. Previous vs Reproduced Results

### NASA Held-out Test Benchmark

| Cell / Metric | Previous Kaggle Result | Converted Python Result | Absolute Difference | Relative Difference | Match Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **B0018 RMSE** | 0.027138 | 0.027143 | +0.000005 | 0.016% | NEAR-EXACT |
| **B0018 MAE** | 0.022273 | 0.022275 | +0.000002 | 0.010% | NEAR-EXACT |
| **B0018 R²** | 0.893527 | 0.893518 | -0.000009 | 0.001% | NEAR-EXACT |
| **B0032 RMSE** | 0.011914 | 0.011915 | +0.000001 | 0.005% | NEAR-EXACT |
| **B0032 MAE** | 0.010502 | 0.010510 | +0.000008 | 0.069% | NEAR-EXACT |
| **B0032 R²** | 0.901750 | 0.901642 | -0.000108 | 0.012% | NEAR-EXACT |
| **B0053_test RMSE** | 0.011280 | 0.011285 | +0.000005 | 0.044% | NEAR-EXACT |
| **B0053_test MAE** | 0.009507 | 0.009509 | +0.000002 | 0.022% | NEAR-EXACT |
| **B0053_test R²** | 0.814501 | 0.814389 | -0.000112 | 0.014% | NEAR-EXACT |
| **NASA MACRO RMSE** | **0.016777** | **0.016781** | **+0.000004** | **0.024%** | **NEAR-EXACT** |
| **NASA MACRO MAE** | **0.014094** | **0.014098** | **+0.000004** | **0.028%** | **NEAR-EXACT** |
| **NASA MACRO R²** | **0.869926** | **0.869850** | **-0.000076** | **0.009%** | **NEAR-EXACT** |

### CALCE Zero-Shot Transfer Benchmark

| Cell / Metric | Previous Kaggle Result | Converted Python Result | Absolute Difference | Relative Difference | Match Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **CS2_35 RMSE** | 0.29335 | 0.29335 | 0.00000 | 0.000% | EXACT |
| **CS2_36 RMSE** | 0.37309 | 0.37309 | 0.00000 | 0.000% | EXACT |
| **CS2_37 RMSE** | 0.34833 | 0.34833 | 0.00000 | 0.000% | EXACT |
| **CS2_38 RMSE** | 0.30610 | 0.30610 | 0.00000 | 0.000% | EXACT |
| **Macro RMSE** | 0.33022 | 0.33022 | 0.00000 | 0.000% | EXACT |
| **Macro MAE** | 0.26699 | 0.26699 | 0.00000 | 0.000% | EXACT |
| **Pooled RMSE** | **0.33252** | **0.33252** | **0.00000** | **0.000%** | **EXACT** |
| **Pooled MAE** | **0.26759** | **0.26759** | **0.00000** | **0.000%** | **EXACT** |
| **Pooled R²** | **-1.78869** | **-1.78869** | **0.00000** | **0.000%** | **EXACT** |

---

## 11. Baseline Code Verification
All 10 baseline model implementations under `Experiment/Baseline/` were audited for syntax correctness, import integrity, model instantiation, input/output tensor shape compliance ($[B, 512, 4] \rightarrow [B]$), parameter count, and runner registration. No full benchmark training runs were triggered.

| Baseline | Import | Model Build | Data Shape Compatibility | Runner Registration | Result Path | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LSTM** | PASS | PASS | `[2, 512, 4]` -> `[2]` | Registered | `Experiment/Result/NASA/` | **PASS** |
| **GRU** | PASS | PASS | `[2, 512, 4]` -> `[2]` | Registered | `Experiment/Result/NASA/` | **PASS** |
| **BiLSTM** | PASS | PASS | `[2, 512, 4]` -> `[2]` | Registered | `Experiment/Result/NASA/` | **PASS** |
| **CNN1D** | PASS | PASS | `[2, 512, 4]` -> `[2]` | Registered | `Experiment/Result/NASA/` | **PASS** |
| **TCN** | PASS | PASS | `[2, 512, 4]` -> `[2]` | Registered | `Experiment/Result/NASA/` | **PASS** |
| **Transformer**| PASS | PASS | `[2, 512, 4]` -> `[2]` | Registered | `Experiment/Result/NASA/` | **PASS** |
| **Informer** | PASS | PASS | `[2, 512, 4]` -> `[2]` | Registered | `Experiment/Result/NASA/` | **PASS** |
| **PatchTST** | PASS | PASS | `[2, 512, 4]` -> `[2]` | Registered | `Experiment/Result/NASA/` | **PASS** |
| **QNN_GRU** | PASS | PASS | `[2, 512, 4]` -> `[2]` | Registered | `Experiment/Result/NASA/` | **PASS** |
| **QLSTM** | PASS | PASS | `[2, 512, 4]` -> `[2]` | Registered | `Experiment/Result/NASA/` | **PASS** |

---

## 12. Ablation Code Verification
All 15 ablation experiments under `Experiment/Ablation_Study/` were audited for file presence, import integrity, experiment configuration, and callable runner registration.

| Experiment | Script File | Import | Configuration | Runner Callable | Result Output | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **E01** | `E01_baseline_reproduction.py` | PASS | Baseline rich entangler | True | `Result/NASA/`, `Result/CALCE/` | **PASS** |
| **E02** | `E02_raw_vs_physics.py` | PASS | Raw vs physics temp gate | True | `Result/NASA/` | **PASS** |
| **E03** | `E03_sei_only.py` | PASS | SEI-only activation | True | `Result/NASA/` | **PASS** |
| **E04** | `E04_plating_only.py` | PASS | Plating-only activation | True | `Result/NASA/` | **PASS** |
| **E05** | `E05_combined_physics.py` | PASS | Combined SEI + Plating | True | `Result/NASA/` | **PASS** |
| **E06** | `E06_fixed_vs_trainable_ea.py`| PASS | Fixed vs trainable $E_a$ | True | `Result/NASA/` | **PASS** |
| **E07** | `E07_classical_vs_quantum.py` | PASS | Classical vs Quantum embed | True | `Result/NASA/` | **PASS** |
| **E08** | `E08_factorial_analysis.py` | PASS | Full factorial matrix | True | `Result/NASA/` | **PASS** |
| **E09** | `E09_cross_cell.py` | PASS | Cross-cell generalizability | True | `Result/NASA/` | **PASS** |
| **E10** | `E10_temporal_extrapolation.py`| PASS| Temporal extrapolation | True | `Result/NASA/` | **PASS** |
| **E11** | `E11_unseen_temperature.py` | PASS | Unseen temp condition | True | `Result/NASA/` | **PASS** |
| **E12** | `E12_robustness.py` | PASS | Noise robustness injection | True | `Result/NASA/` | **PASS** |
| **E13** | `E13_efficiency.py` | PASS | Computational efficiency | True | `Result/NASA/` | **PASS** |
| **E14** | `E14_architectural_ablation.py`| PASS| PE / Pooling / Smoothers | True | `Result/NASA/` | **PASS** |
| **E15** | `E15_final_multiseed.py` | PASS | Multi-seed statistical val | True | `Result/NASA/` | **PASS** |

---

## 13. Runner Verification
All entry-point CLI runners were validated using `--help` flag executions:
1. `Experiment/run.py --help`: Exited with code 0. Displays options for `--group`, `--dataset`, `--model`, `--experiment`, `--train`, `--epochs`, `--batch-size`.
2. `Experiment/Proposed_Model/run.py --help`: Exited with code 0. Displays options for `--dataset`, `--evaluate-only`, `--train`, `--checkpoint`.
3. `Experiment/Baseline/run.py --help`: Exited with code 0. Displays options for `--model`, `--epochs`, `--batch-size`, `--lr`.
4. `Experiment/Ablation_Study/run.py --help`: Exited with code 0. Displays options for `--experiment`.

---

## 14. Data Leakage Audit
- **Scaler Fitting**: Checked and verified that `fit_train_scaler()` is called strictly on the combined `train_X` tensor. Test arrays are transformed using `.transform()` with clipping and never enter `.fit()`.
- **Cross-Cell Separation**: `B0018` and `B0032` are strictly excluded from the training loader.
- **Chronological Boundary**: In cell `B0053`, the first 70% of cycles (1 to 78) form the train portion and the subsequent 30% of cycles (79 to 112) form the test portion. No shuffle is applied across the boundary.
- **CALCE Zero-Shot Isolation**: No CALCE cells are exposed to training or scaling. They are processed strictly as an external validation set using the NASA-derived scaler.
- **Verdict**: **ZERO DATA LEAKAGE DETECTED**.

---

## 15. Identified Issues
1. **Windows Default Python Launcher Alias**: Calling generic `python` invokes the Windows App Execution Alias rather than the active Python installation. Resolution: invoke via `py -3.10` or explicit interpreter path.
2. **Heavy Import Delay**: `pennylane` and PyTorch incur an import overhead of 2-4 seconds. Resolution: lazy imports inside execution functions allow `--help` to respond instantaneously.

---

## 16. Fixes Applied
- Ensured all entry point scripts and sub-runners utilize lazy imports inside the operational functions to prevent CLI delays during argument parsing.
- Centralized dataset path resolution in `Experiment/utils/paths.py` to ensure relative portability across environments.
- Created `GarbageResults/` containing comprehensive English and Bengali audit logs (`GarbageResults/report.md`).

---

## 17. Final Verdict

### Proposed Model
- **Kaggle implementation**: `VERIFIED`
- **Python implementation**: `VERIFIED`
- **Preprocessing**: `VERIFIED`
- **Architecture**: `VERIFIED`
- **Quantum circuit**: `VERIFIED`
- **Training**: `VERIFIED`
- **Reproduction**: `NEAR-EXACT REPRODUCTION` (Macro RMSE: 0.01678, Macro MAE: 0.01410, Macro R²: 0.86985; CALCE Pooled RMSE: 0.33252)

### Baselines
- **Code integrity**: `PASS`
- **Runner integrity**: `PASS`

### Ablations
- **Code integrity**: `PASS`
- **Experiment mapping**: `PASS`
- **Runner integrity**: `PASS`

### Overall
**READY FOR EXPERIMENT PHASE**
