# Datasets Overview

This directory contains the processed battery datasets used for training and evaluating **TE-Q-Transformer** and contemporary baselines.

---

## 1. Directory Structure

```
datasets/
│
├── NASA/
│   └── processed/
│       ├── B0005_X.npy, B0005_soh.npy
│       ├── B0006_X.npy, B0006_soh.npy
│       ├── B0007_X.npy, B0007_soh.npy
│       ├── B0018_X.npy, B0018_soh.npy
│       ├── B0029_X.npy, B0029_soh.npy
│       ├── B0030_X.npy, B0030_soh.npy
│       ├── B0031_X.npy, B0031_soh.npy
│       ├── B0032_X.npy, B0032_soh.npy
│       ├── B0053_X.npy, B0053_soh.npy
│       ├── nasa_train_scaler.pkl
│       └── nasa_train_scaler_sklearn1.8.0.pkl
│
└── CALCE/
    ├── README.md
    └── processed/
        ├── CS2_35_X_unscaled.npy, CS2_35_soh.npy, CS2_35_cycle.npy, ...
        ├── CS2_36_X_unscaled.npy, CS2_36_soh.npy, CS2_36_cycle.npy, ...
        ├── CS2_37_X_unscaled.npy, CS2_37_soh.npy, CS2_37_cycle.npy, ...
        ├── CS2_38_X_unscaled.npy, CS2_38_soh.npy, CS2_38_cycle.npy, ...
        ├── excluded_segments.jsonl
        └── preprocess_metadata.json
```

---

## 2. NASA Ames Battery Dataset

### Source
- **Origin:** NASA Prognostics Center of Excellence (PCoE) Battery Data Set.
- **Chemistry:** Commercial 18650 LiCoO₂ / graphite cylindrical cells (nominal capacity 2.0 Ah).
- **Operating Conditions:**
  - Ambient 24°C: `B0005`, `B0006`, `B0007`, `B0018`
  - Ambient 4°C: `B0029`, `B0030`, `B0031`, `B0032`
  - Ambient 44°C: `B0053`

### Processed Tensor Format
- Each discharge cycle is interpolated to fixed-length time series:
  - Input array (`{cell}_X.npy`): Shape `[N, 512, 4]` (float32).
  - Channels (Order strictly fixed):
    1. Channel 0: **Voltage** ($V$)
    2. Channel 1: **Current** ($A$)
    3. Channel 2: **Temperature** ($T_C$ in unscaled degrees Celsius, preserved for Arrhenius calculation)
    4. Channel 3: **Normalized Time** ($t_{\text{norm}} \in [0, 1]$ within discharge cycle)
  - Target array (`{cell}_soh.npy`): Shape `[N]` (float32), representing normalized state-of-health $SOH = C_k / C_0$.

### Non-Leakage Splitting Contract
- **Train Pool:** `B0005`, `B0006`, `B0007`, `B0029`, `B0030`, `B0031`, and first 70% of `B0053` (660 cycles).
- **Test Pool:**
  - Unseen cells: `B0018` (132 cycles), `B0032` (39 cycles).
  - Temporal extrapolation: final 30% of `B0053` (16 cycles).
- **Scaler Rule:** Feature scaling (`MinMaxScaler`) is fitted **strictly on training cycles** for channels `(0, 1, 3)`. Channel 2 ($T_C$) is never normalized so the Arrhenius activation energies maintain physical meaning ($T_K = T_C + 273.15$).

---

## 3. CALCE Battery Dataset

### Source
- **Origin:** Center for Advanced Life Cycle Engineering (CALCE), University of Maryland.
- **Chemistry:** Prismatic LiCoO₂ cells (CS2 series: `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`; nominal capacity 1.1 Ah).

### Processed Tensor Format
- Interpolated to match the NASA tensor contract:
  - Input array (`{cell}_X_unscaled.npy`): Shape `[N, 512, 4]` (float32).
  - Target array (`{cell}_soh.npy`): Shape `[N]` (float32).

### Evaluation Protocol
- CALCE is used strictly for **zero-shot cross-dataset transfer**:
  - The model weights are frozen from the NASA training checkpoint.
  - The frozen NASA train scaler transforms channels (0, 1, 3).
  - Zero CALCE cycles are used for training, scaler fitting, or hyperparameter selection.

---

## 4. Git and Storage Policy

- **Committed Files:** The processed `.npy` files for NASA (~7.5 MB) and CALCE (~32 MB) are tracked in this repository for instant reproducibility.
- **Excluded Files:** Raw measurement records, unpackaged CSVs, and Excel files (>300 MB) are excluded from version control via `.gitignore`.
