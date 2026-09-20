# E06 Generalization Benchmark: Single Unified Kaggle Notebook
## TE-Q-Transformer V2 Research Project

---

### 1. Scientific Objective & Research Question
Experiment E06 evaluates whether the state-of-health (SOH) estimation performance observed in the aggregate E05 benchmark generalizes consistently across unseen battery cells and forward in time.

The central research question is:
> *"Does the observed performance of TE-Q-Transformer remain consistent when evaluated on cells that were not used during training (unseen-cell generalization) and on a temporally held-out portion of a battery cell (temporal extrapolation)?"*

---

### 2. Complete Benchmark Suite (All 11 Models)
All 11 models are integrated inside the single notebook [`E06_Generalization.ipynb`](file:///e:/TE-Q-Transformer/TE-Q-Transformer/Experiment/E06_Generalization/E06_Generalization.ipynb) and trained sequentially using exact architectural implementations directly extracted from `baselineComparison.ipynb` and `src/models/teq_transformer.py`:

| # | Model | Family | Parameters | Architecture Provenance |
| :--- | :--- | :--- | :---: | :--- |
| 1 | **TE-Q-Transformer** | Proposed Quantum Foundation | 92,554 | Multi-head attention + Arrhenius SEI & Plating physics gates + Rich Entangler circuit |
| 2 | **LSTM** | Classical Recurrent | 71,105 | 2-layer LSTM (hidden_dim=64, dropout=0.1) (Hochreiter & Schmidhuber, 1997) |
| 3 | **GRU** | Classical Recurrent | 54,465 | 2-layer GRU (hidden_dim=64, dropout=0.1) (Cho et al., 2014) |
| 4 | **CNN1D** | Convolutional | 47,041 | 4 conv blocks with dilation/stride, adaptive avg pool (Kiranyaz et al., 2021) |
| 5 | **TCN** | Convolutional | 91,841 | Causal dilated convolutions (1,2,4,8) with residual connections (Bai et al., 2018) |
| 6 | **DLinear** | Linear | 1,031 | Moving avg series decomposition + seasonal & trend linear maps (AAAI 2023) |
| 7 | **Transformer** | Attention | 80,257 | Sinusoidal PositionalEncoding + 2-layer Multi-Head Attention Encoder (Vaswani et al., 2017) |
| 8 | **PatchTST** | Attention | 109,962 | Channel-independent patch embedding (`patch_len=16`, `stride=8`) + RevIN (ICLR 2023) |
| 9 | **iTransformer** | Attention | 137,473 | Inverted variate tokenization across sequence length 512 (ICLR 2024 Spotlight) |
| 10 | **QLSTM** | Quantum-Recurrent | 37,853 | Hybrid PennyLane QNN front-end (4 qubits, 2 layers) + LSTM (Wang & Kebede) |
| 11 | **QNN-GRU** | Quantum-Recurrent | 29,533 | Hybrid PennyLane QNN front-end (4 qubits, 2 layers) + GRU (Soon & Soon) |

---

### 3. Exact NASA Ames Data Split
Following the project's multi-temperature protocol:
- **Training Set (660 discharge cycles):**
  - Room-temperature cells: `B0005`, `B0006`, `B0007` (all cycles)
  - High-temperature cells: `B0029`, `B0030`, `B0031` (all cycles)
  - Low-temperature cell: First 70% of `B0053` (cycles 0–36, exactly 37 cycles)
- **Evaluation Set (187 discharge cycles):**
  - **Unseen-Cell Generalization (171 cycles):**
    - `B0018` (132 cycles, room-temperature)
    - `B0032` (39 cycles, high-temperature)
  - **Temporal Extrapolation (16 cycles):**
    - `B0053_test` (final 30% segment, cycles 37–52, low-temperature)

---

### 4. Critical Scientific Definitions & Terminology Safeguards
- **Unseen-Cell Generalization:** Cells `B0018` and `B0032` are completely unseen during training, feature scaling, model selection, and hyperparameter tuning.
- **Temporal Extrapolation:** Cell `B0053` is **NOT** an unseen cell; it represents chronological future extrapolation from early-life degradation (first 70%) to late-life degradation (final 30%).
- **CRITICAL TERMINOLOGY RULE:** E06 is **NEVER** designated as an "unseen-temperature" experiment. Temperature is inherently coupled with operating conditions and cell chemistry in NASA Ames; evaluations are strictly designated as `unseen_cell` and `temporal_extrapolation`.

---

### 5. Preprocessing & Zero-Leakage Policy
- **Input channels:** `[Voltage_V, Current_A, Temperature_C, Time_norm]` (length: 512 points).
- **Target SOH:** $C_k / C_0$ (normalized by initial discharge capacity).
- **Scaler:** `MinMaxScaler(feature_range=(0, 1))` fitted **strictly on training data** for Voltage, Current, and Time_norm.
- **Temperature channel:** Left in **unscaled Celsius** for model input, preserving physical Arrhenius rate kinetics ($E_{a,sei}, E_{a,pl}$).
- **Zero test leakage:** Test cells (`B0018`, `B0032`) and the `B0053` test segment are NEVER used for fitting scalers, computing intermediate validation losses, or selecting checkpoints.

---

### 6. Kaggle Execution Instructions
1. Upload `E06_Generalization.ipynb` to Kaggle as a new Notebook.
2. Under Settings:
   - **Accelerator:** GPU P100 or T4.
   - **Internet:** On (to install PennyLane if needed).
3. Attach your dataset containing the NASA battery `.npy` files.
4. Set execution mode:
   ```python
   RUN_MODE = "FULL"  # For the complete 80-epoch publication benchmark
   ```
5. Click **"Run All"**.
6. When execution finishes, all artifacts will be automatically saved under `E06_results/` and zipped into `E06_Generalization_Results.zip` for 1-click download.
