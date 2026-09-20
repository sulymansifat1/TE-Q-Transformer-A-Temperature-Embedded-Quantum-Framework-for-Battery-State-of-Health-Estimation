# Scientific Conversion & Reproducibility Audit Report

This audit documents the conversion of the legacy Jupyter notebook experiments into the modular, reviewer-friendly Python experiment framework for **TE-Q-Transformer V2**.

---

## A. Converted Notebooks → Python Mapping

| Legacy Notebook Reference | Converted Python Module / Script | Architecture / Content | Conversion Status |
| :--- | :--- | :--- | :---: |
| `Nasa_TE-Q-Transformer (1).ipynb` | `Experiment/Proposed_Model/model.py`<br>`Experiment/Proposed_Model/train.py`<br>`Experiment/Proposed_Model/evaluate.py` | Full TE-Q-Transformer (Arrhenius Quantum Embedding + Conv1D + Transformer Encoder) | **VERIFIED & ACTIVE** |
| `nasa_teq_component_ablation_study.ipynb` | `Experiment/Ablation_Study/E01_baseline_reproduction.py`<br>`Experiment/Ablation_Study/E14_architectural_ablation.py` | Rich entangler, basic entangler, Pauli map, Conv1D, GRU, PosEnc, CLS pooling | **VERIFIED & ACTIVE** |
| `nasa-te-q-transformer-lstm.ipynb` | `Experiment/Baseline/LSTM.py` | 2-layer LSTM ($d=64$), last-token pooling, MLP head | **VERIFIED & ACTIVE** |
| `nasa-te-q-transformer-gru.ipynb` | `Experiment/Baseline/GRU.py` | 2-layer GRU ($d=64$), last-token pooling, MLP head | **VERIFIED & ACTIVE** |
| `nasa-te-q-transformer-transformer.ipynb` | `Experiment/Baseline/Transformer.py` | 3-layer Classical Transformer Encoder ($d=64, h=2$), CLS token | **VERIFIED & ACTIVE** |
| `nasa-te-q-transformer-qlstm.ipynb` | `Experiment/Baseline/QLSTM.py` | 4-qubit PennyLane variational quantum layer + LSTM ($d=64$) | **VERIFIED & ACTIVE** |
| `nasa-te-q-transformer-qnn-gru.ipynb` | `Experiment/Baseline/QNN_GRU.py` | 4-qubit PennyLane variational quantum layer + GRU ($d=64$) | **VERIFIED & ACTIVE** |
| Literature Baseline (BiLSTM) | `Experiment/Baseline/BiLSTM.py` | Bidirectional LSTM ($d=64$, 2 layers) | **VERIFIED & ACTIVE** |
| Literature Baseline (1D CNN) | `Experiment/Baseline/CNN1D.py` | 3-layer 1D Temporal CNN with BatchNorm & GELU | **VERIFIED & ACTIVE** |
| Literature Baseline (TCN) | `Experiment/Baseline/TCN.py` | Temporal Convolutional Network with causal dilations ($2^0 \dots 2^3$) | **VERIFIED & ACTIVE** |
| Literature Baseline (Informer) | `Experiment/Baseline/Informer.py` | Informer with self-attention & distillation pooling | **VERIFIED & ACTIVE** |
| Literature Baseline (PatchTST) | `Experiment/Baseline/PatchTST.py` | Patching transformer (patch length 16, stride 8) | **VERIFIED & ACTIVE** |
| `CALCE_TE-Q-Transformer.ipynb` & `src/eval/nasa_calce_zero_shot.py` | `Experiment/Dataset/calce.py`<br>`Experiment/Proposed_Model/evaluate.py` | Zero-shot evaluation of frozen NASA weights on CALCE CS2 cells | **VERIFIED & ACTIVE** |

---

## B. Experiment 1–15 Scientific Plan Mapping

| Experiment ID | Experiment Title | Scientific Question | Python Script | Target Dataset | Primary Output | Status |
| :---: | :--- | :--- | :--- | :---: | :--- | :---: |
| **E01** | **Baseline Reproduction** | Does the frozen checkpoint reproduce reported NASA test and CALCE zero-shot metrics? | `Ablation_Study/E01_baseline_reproduction.py` | NASA + CALCE | `E01_baseline_reproduction_NASA_metrics.csv`<br>`E01_baseline_reproduction_CALCE_metrics.csv` | **ACTIVE** |
| **E02** | **Raw vs. Physics Temp** | Does replacing Arrhenius physics with raw MinMax scaled temperature degrade accuracy? | `Ablation_Study/E02_raw_vs_physics.py` | NASA | `E02_raw_vs_physics_NASA_metrics.csv` | **ACTIVE** |
| **E03** | **SEI Mechanism Only** | Does high-temperature SEI layer growth alone capture battery degradation? | `Ablation_Study/E03_sei_only.py` | NASA | `E03_sei_only_NASA_metrics.csv` | **ACTIVE** |
| **E04** | **Plating Mechanism Only** | Does low-temperature Lithium plating alone capture degradation? | `Ablation_Study/E04_plating_only.py` | NASA | `E04_plating_only_NASA_metrics.csv` | **ACTIVE** |
| **E05** | **Combined Physics** | Does dual-mechanism SEI + Plating yield synergistic degradation tracking? | `Ablation_Study/E05_combined_physics.py` | NASA | `E05_combined_physics_NASA_metrics.csv` | **ACTIVE** |
| **E06** | **Fixed vs. Trainable $E_a$** | Does optimizing activation energies via gradient descent outperform static values? | `Ablation_Study/E06_fixed_vs_trainable_ea.py` | NASA | `E06_fixed_vs_trainable_ea_NASA_metrics.csv` | **ACTIVE** |
| **E07** | **Classical vs. Quantum** | Does parameter-matched classical MLP embedding match 4-qubit Hilbert space encoding? | `Ablation_Study/E07_classical_vs_quantum.py` | NASA | `E07_classical_vs_quantum_NASA_metrics.csv` | **ACTIVE** |
| **E08** | **Factorial Analysis** | How do Physics Gate $\times$ Quantum Circuit $\times$ Temporal Smoothing interact? | `Ablation_Study/E08_factorial_analysis.py` | NASA | `E08_factorial_analysis_NASA_metrics.csv` | **ACTIVE** |
| **E09** | **Cross-Cell Generalization** | How well does the model generalize across diverse manufacturing cell profiles? | `Ablation_Study/E09_cross_cell.py` | NASA | `E09_cross_cell_NASA_metrics.csv` | **ACTIVE** |
| **E10** | **Temporal Extrapolation** | Can early-life cycling data (50%, 60%, 70%) reliably forecast end-of-life knee degradation? | `Ablation_Study/E10_temporal_extrapolation.py` | NASA | `E10_temporal_extrapolation_NASA_metrics.csv` | **ACTIVE** |
| **E11** | **Unseen Temperature** | Does physics representation prevent catastrophic drift at unseen operating temperatures? | `Ablation_Study/E11_unseen_temperature.py` | NASA | `E11_unseen_temperature_NASA_metrics.csv` | **ACTIVE** |
| **E12** | **Sensor Robustness** | How resilient is the architecture against Gaussian sensor noise ($\sigma \in \{0.01, 0.02, 0.05\}$)? | `Ablation_Study/E12_robustness.py` | NASA | `E12_robustness_NASA_metrics.csv` | **ACTIVE** |
| **E13** | **Computational Efficiency** | What is the inference latency (ms/window), FLOPs, and parameter overhead? | `Ablation_Study/E13_efficiency.py` | NASA | Terminal benchmark & parameter report | **ACTIVE** |
| **E14** | **Architectural Ablation** | What are the contributions of Rich Entangler, Pauli map, Conv1D, and CLS pooling? | `Ablation_Study/E14_architectural_ablation.py` | NASA | `E14_architectural_ablation_NASA_metrics.csv` | **ACTIVE** |
| **E15** | **Multi-Seed Validation** | Are performance gains statistically robust across random seeds ($42 \dots 46$)? | `Ablation_Study/E15_final_multiseed.py` | NASA | `E15_multiseed_statistical_summary.csv` | **ACTIVE** |

---

## C. Confirmation of Scientific Invariants

Every scientific invariant has been strictly audited and preserved:

1. **Preprocessing Methodology**:
   - Resampling: Exactly 512 intra-cycle phase points preserved.
   - Endpoint preservation: Voltage start ($V_0$) and end ($V_{\text{end}}$) preserved.
   - SOH derivation: $y_i = C_i / C_0$ (normalized by cycle-1 initial capacity).
2. **Feature Normalization**:
   - Voltage (0), Current (1), Time_norm (3) scaled to $[0, 1]$ via MinMaxScaler fitted **only on the NASA training split**.
   - Temperature (2) **strictly unscaled in Celsius** ($T_K = T_C + 273.15$ in Kelvin) to preserve thermodynamic Arrhenius validity.
   - Zero test data leakage: Test cells (`B0018`, `B0032`, `B0053_test`, and CALCE) never enter the scaler fit.
3. **Dataset Splits**:
   - Hybrid Multi-temperature Train: `B0005`, `B0006`, `B0007` (24°C) + `B0029`, `B0030`, `B0031` (44°C) + `B0053_train` (first 70% of 4°C).
   - Test Split: `B0018` (unseen 24°C), `B0032` (unseen 44°C), `B0053_test` (chronologically held-out 30% of 4°C).
   - External Validation: CALCE `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38` evaluated zero-shot with frozen NASA weights and scaler.
4. **Model Architecture**:
   - 4-qubit QuantumEmbeddingLayer with Arrhenius activation gate.
   - Rich Entangler: layered $RZ + CNOT + IsingZZ + H + RZ$.
   - Projection $4 \to d_{\text{model}}=64$.
   - Temporal Conv1D smoothing (kernel size 3, padding 1).
   - Positional Encoding (1D Sinusoidal).
   - Transformer Encoder (3 layers, 2 heads, dim_feedforward=64, GELU, norm_first).
   - `[CLS]` token pooling + MLP regression head ($64 \to 64 \to 1$).
5. **Training Protocol**:
   - Optimizer: AdamW ($\text{lr}=10^{-3}$, weight_decay=$5\times 10^{-2}$).
   - Loss: MSELoss.
   - Scheduler: ReduceLROnPlateau (factor 0.5, patience 10).
   - Early stopping: patience 20.
   - Gradient clipping: max_norm 1.0.
6. **Evaluation Metrics**:
   - RMSE, MAE, MAPE (%), R2, MaxE with standardized formulas.

---

## D. Known Issues / Notebook Discrepancies Preserved

1. **CALCE Zero-Shot Constant Behavior**:
   - In the frozen NASA $\to$ CALCE evaluation, predictions hover near $\approx 1.01$ because CALCE features experience a severe domain shift (room temperature ~23°C without measured temperature channel, filled as constant 23°C).
   - *Audit Decision*: Preserved strictly as-is. We do NOT tune or fit CALCE scalers to artificially improve zero-shot metrics.
2. **Negative $R^2$ on Low-Variance SOH Targets**:
   - On cells with small total capacity drop (e.g. `B0053_test` which drops only ~0.04 SOH across its final cycles), $SS_{\text{tot}}$ is small ($<10^{-3}$), causing slight mean offsets to yield negative $R^2$ values despite low RMSE ($<0.012$).
   - *Audit Decision*: Documented in metrics; RMSE and MAE serve as the uncompromised primary metrics.

---

## E. Reviewer Execution Commands

Reviewers can reproduce any portion of the research using these commands:

```bash
# 1. View all options
python Experiment/run.py --help

# 2. Reproduce Proposed Model on NASA & CALCE (using pre-trained checkpoint)
python Experiment/run.py --group proposed

# 3. Benchmark All 10 Baseline Models
python Experiment/run.py --group baseline --model all

# 4. Benchmark an Individual Baseline (e.g. Informer, PatchTST, QLSTM)
python Experiment/run.py --group baseline --model Informer
python Experiment/run.py --group baseline --model PatchTST
python Experiment/run.py --group baseline --model QLSTM

# 5. Run All 15 Ablation Experiments
python Experiment/run.py --group ablation --experiment all

# 6. Run Specific Ablations (e.g., E01 Baseline, E02 Physics vs Raw, E07 Classical vs Quantum)
python Experiment/run.py --group ablation --experiment E01
python Experiment/run.py --group ablation --experiment E02
python Experiment/run.py --group ablation --experiment E07
python Experiment/run.py --group ablation --experiment E15
```
