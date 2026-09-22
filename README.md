# TE-Q-Transformer: A Temperature-Embedded Quantum Framework for Battery State-of-Health Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35%2B-blueviolet.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official research code repository for **TE-Q-Transformer**, a hybrid quantum-classical deep learning framework for lithium-ion battery state-of-health (SOH) estimation. The framework integrates Arrhenius degradation kinetics for elevated-temperature solid electrolyte interphase (SEI) growth and sub-ambient lithium plating into a four-qubit simulated variational quantum circuit and a temporal Conv1D-Transformer backbone.

---

## Authors

- **Zadid Al Lisan** — *Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh* ([ORCID: 0009-0001-1801-051X](https://orcid.org/0009-0001-1801-051X))  
- **Md. Sulyman Islam Sifat** — *Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh* ([ORCID: 0009-0005-6377-8973](https://orcid.org/0009-0005-6377-8973))  
- **M.S. Hossain Lipu** — *School of Electrical, Computer and Telecommunications Engineering, University of Wollongong, NSW 2522, Australia* ([ORCID: 0000-0001-9060-4454](https://orcid.org/0000-0001-9060-4454))  
- **Nazakat Ali** — *ABB, Vasteras, Sweden* ([ORCID: 0000-0002-3875-812X](https://orcid.org/0000-0002-3875-812X))  
- **Md Alamgir Kabir** *(Corresponding Author)* — *Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh* ([ORCID: 0000-0002-7136-6339](https://orcid.org/0000-0002-7136-6339), Email: `kabir.cse@diu.edu.bd`)

---

## Abstract

Reliable state-of-health (SOH) estimation is essential for the safe and efficient operation of lithium-ion batteries in electric vehicles and energy storage systems. Operating temperature strongly governs battery aging. However, existing data-driven models simply treat temperature as an ordinary numerical input, ignoring the underlying physical degradation kinetics. Consequently, these models often struggle to generalize across unseen cells. To address this, we introduce **TE-Q-Transformer**, a hybrid framework that integrates Arrhenius-based kinetics for high-temperature solid electrolyte interphase growth and low-temperature lithium plating into a four-qubit simulated quantum circuit and a Conv1D-Transformer backbone. Evaluated on the NASA cross-cell benchmark against ten contemporary baselines, TE-Q-Transformer achieves superior cross-cell estimation accuracy. Controlled ablations demonstrate that the physics-guided encoding achieves a **16.5% error reduction** over conventional temperature feature scaling, while replacing the quantum circuit with a parameter-matched classical counterpart markedly degrades performance. These results demonstrate that embedding degradation kinetics directly into input features substantially improves cross-cell state-of-health estimation.

The complete research code, datasets, and implementation framework are publicly available at:  
👉 **[https://github.com/sulymansifat1/TE-Q-Transformer](https://github.com/sulymansifat1/TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation)**

---

## Scientific Overview

Battery State of Health (SOH) is defined as the ratio of current discharge capacity $C_k$ to nominal initial capacity $C_0$:
$$\text{SOH}_k = \frac{C_k}{C_0}$$

Operating temperature fundamentally dictates battery aging mechanisms:
- **Elevated Temperatures ($T > 25\,^\circ\text{C}$):** Accelerated side reactions between active materials and electrolyte drive solid electrolyte interphase (SEI) layer growth and solvent decomposition.
- **Sub-Ambient Temperatures ($T < 25\,^\circ\text{C}$):** Retarded lithium-ion transport and solid-state diffusion induce electrode polarization, precipitating metallic lithium plating during charging.

### The Problem with Conventional Normalization
Standard sequential deep learning models typically treat temperature as an ordinary numerical channel rescaled via min-max or z-score normalization. While numerical rescaling normalizes dynamic ranges, it strips away the non-linear kinetic pathways that govern thermal degradation.

### The TE-Q-Transformer Architecture
TE-Q-Transformer addresses this by formulating a physics-guided temperature encoding based on dual Arrhenius reaction kinetics:
$$\phi(T) = \exp\left(\frac{E_{a,\text{sei}}}{R}\left(\frac{1}{T_{\text{ref}}} - \frac{1}{T_K}\right)\right) + \exp\left(\frac{E_{a,\text{pl}}}{R}\left(\frac{1}{T_K} - \frac{1}{T_{\text{ref}}}\right)\right)$$
$$\theta_{\text{temp}} = \frac{\pi \cdot \phi(T)}{4}$$
where $E_{a,\text{sei}}$ and $E_{a,\text{pl}}$ are trainable activation energies, $R = 8.314\,\text{J/(mol}\cdot\text{K)}$, and $T_{\text{ref}} = 298.15\,\text{K}$.

This physical angle is coupled with normalized voltage, current, and intra-cycle time into a **four-qubit variational quantum circuit** simulated via PennyLane (`default.qubit`). The circuit employs a rich entangler (RZ rotations, CNOT ladder, IsingZZ couplings, Toffoli gates, and Hadamard layers) with Pauli-Z expectation value readout $\langle \sigma_z^i \rangle$. The quantum representations are projected into a latent space, smoothed via Conv1D, and processed by a multi-layer Transformer encoder backbone.

> [!NOTE]
> **Quantum Simulation Disclaimer:** All quantum components are simulated classically via PennyLane state-vector routines. We claim no quantum hardware advantage or fault-tolerant quantum supremacy; the parameterized circuit serves as a non-linear feature transformation layer.

----

## Repository Structure

```
TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation/
│
├── README.md                                  # Complete research documentation & user guide
├── LICENSE                                    # MIT License
├── requirements.txt                           # Verified project dependencies
├── .gitignore                                 # Git rules excluding caches and raw archives
│
├── datasets/
│   ├── README.md                              # Dataset provenance, contracts, and formats
│   ├── NASA/
│   │   └── processed/                         # Processed NASA cycle sequences [N, 512, 4] & scalers
│   └── CALCE/
│       ├── README.md                          # CALCE source provenance & documentation
│       └── processed/                         # Processed CALCE CS2 cycle sequences [N, 512, 4]
│
├── models/
│   ├── __init__.py                            # Top-level model registry
│   ├── proposed/
│   │   ├── __init__.py
│   │   └── te_q_transformer.py                # Authoritative TE-Q-Transformer architecture
│   │
│   └── baselines/
│       ├── __init__.py                        # Dynamic baseline factory get_baseline_model()
│       ├── cnn1d.py                           # 1D Temporal CNN baseline
│       ├── tcn.py                             # Causal Dilated TCN baseline
│       ├── dlinear.py                         # LTSF-Linear series decomposition baseline
│       ├── transformer.py                     # Classical Multi-Head Attention baseline
│       ├── patchtst.py                        # Patch Time Series Transformer baseline
│       ├── itransformer.py                    # Inverted Variate Transformer baseline
│       ├── qlstm.py                           # Gate-level Quantum LSTM baseline
│       ├── qnn_gru.py                         # Quantum Neural Network + GRU baseline
│       ├── lstm.py                            # Standard LSTM baseline
│       └── gru.py                             # Standard GRU baseline
│
├── notebooks/
│   ├── benchmark/
│   │   └── all_models_benchmark.ipynb         # Executable benchmark suite (proposed + 10 baselines)
│   ├── ablation/
│   │   └── ablation_study.ipynb               # Controlled component ablations (physics, mechanisms, VQC)
│   └── generalization/
│       └── generalization_study.ipynb         # Unseen-cell, temporal extrapolation, and CALCE zero-shot
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── nasa_loader.py                     # NASA PyTorch dataset, dataloaders, scalers
│   │   └── calce_loader.py                    # CALCE CS2 loader
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py                         # RMSE, MAE, MAPE, R2, MaxE calculations
│       └── zero_shot.py                       # Zero-shot cross-dataset evaluation engine
│
└── tests/
    ├── test_models.py                         # Forward pass verification [2, 512, 4] -> [2]
    ├── test_datasets.py                       # Dataset file existence and tensor shapes verification
    └── test_scaling.py                        # Non-leakage physical scaling contract test
```

---

## Experimental Protocol

### NASA Ames Benchmark Protocol
- **Cycle Sequence Length:** Exactly 512 points per discharge cycle $(V, I, T_C, t_{\text{norm}})$.
- **Feature Channel Ordering:**
  - Index 0: Voltage ($V$)
  - Index 1: Current ($A$)
  - Index 2: Temperature ($T_C$ in Celsius, kept strictly unscaled for Arrhenius calculations)
  - Index 3: Normalized time ($t_{\text{norm}} \in [0, 1]$ within discharge cycle)
- **Held-Out Partitioning:**
  - **Training Pool (660 cycles):** `B0005`, `B0006`, `B0007` (24°C), `B0029`, `B0030`, `B0031` (4°C), and first 70% of `B0053` (44°C).
  - **Held-Out Unseen Cells:** `B0018` (132 cycles, 24°C) and `B0032` (39 cycles, 4°C).
  - **Temporal Extrapolation:** Final 30% of `B0053` (16 cycles, 44°C).
- **Non-Leakage Scaling Contract:** `MinMaxScaler` is fitted exclusively on the training pool for features `(0, 1, 3)`. Test cells are never seen by the scaler.

### External CALCE Transfer Protocol
- Evaluated zero-shot on CALCE CS2 prismatic cells (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`) with frozen model weights and the frozen NASA scaler, testing cross-chemistry domain transfer.

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.0 or higher
- PennyLane 0.35 or higher

### Local Environment Setup
```bash
# Clone the repository
git clone git@github.com:sulymansifat1/TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation.git
cd TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the automated unit tests
python -m unittest discover -s tests -p "test_*.py"
```

---

## Usage Guide

### 1. Instantiating Proposed TE-Q-Transformer
```python
import torch
from models.proposed import TEQTransformer, rich_entangler_config

# Initialize model architecture
config = rich_entangler_config()
model = TEQTransformer(config)

# Input format: [batch_size, sequence_length, features]
# Features: (Voltage, Current, Temp_Celsius, Normalized_Time)
x = torch.randn(2, 512, 4)
soh_prediction = model(x)  # Tensor of shape [2]
print(f"SOH Prediction shape: {soh_prediction.shape}")
```

### 2. Instantiating Baseline Models
```python
from models.baselines import get_baseline_model, list_baselines

print("Available baselines:", list_baselines())
# Output: ['cnn1d', 'tcn', 'dlinear', 'transformer', 'patchtst', 'itransformer', 'qlstm', 'qnn_gru', 'lstm', 'gru']

model = get_baseline_model("transformer")
x = torch.randn(2, 512, 4)
output = model(x)
```

### 3. Loading DataLoaders
```python
from pathlib import Path
from src.data.nasa_loader import get_nasa_dataloaders

data_dir = Path("datasets/NASA/processed")
train_loader, test_loaders, scaler = get_nasa_dataloaders(data_dir, batch_size=8)

print(f"Training batches: {len(train_loader)}")
for cell_id, loader in test_loaders.items():
    print(f"Test cell {cell_id}: {len(loader.dataset)} cycles")
```

---

## Running on Kaggle

Each research notebook is self-contained and equipped with automatic environment discovery:

1. Create a new notebook on [Kaggle](https://www.kaggle.com/).
2. Enable GPU acceleration under **Notebook Settings** (e.g. **GPU P100** or **GPU T4 x2**).
3. Set **Internet: ON**.
4. In the initial notebook cell, clone and enter the repository:
   ```bash
   !git clone https://github.com/sulymansifat1/TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation.git
   %cd TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation
   !pip install -r requirements.txt
   ```
5. Open and execute any of the workflow notebooks:
   - **Full Benchmark:** `notebooks/benchmark/all_models_benchmark.ipynb`
   - **Ablation Studies:** `notebooks/ablation/ablation_study.ipynb`
   - **Generalization Analysis:** `notebooks/generalization/generalization_study.ipynb`

---

## Scientific Limitations

1. **Coupled Operating Temperatures:** In the NASA Ames dataset, ambient temperatures are coupled with specific cell runs. Because 4°C, 24°C, and 44°C are represented during training, this work demonstrates unseen-cell generalization rather than unseen-temperature extrapolation.
2. **Cross-Chemistry Domain Shift:** Zero-shot transfer from cylindrical LiCoO₂ cells to prismatic CALCE cells exhibits cross-chemistry domain gaps, indicating the necessity of targeted domain adaptation.
3. **Simulated Quantum Execution:** Quantum circuits are simulated using state-vector linear algebra on classical CPU/GPU runtimes via PennyLane; no physical QPU fault tolerance is implied.
4. **Empirical Parameter Calibration:** Activation energies $E_{a,\text{sei}}$ and $E_{a,\text{pl}}$ are trainable empirical parameters initialized from electrochemical literature, not physically isolated reaction constants.

---

## Citation

If you use this codebase or model architecture in your research, please cite our paper:

```bibtex
@article{lisan2026teqtransformer,
  title   = {TE-Q-Transformer: A Temperature-Embedded Quantum Framework for Battery State-of-Health Estimation},
  author  = {
    Lisan, Zadid Al and
    Islam Sifat, Md. Sulyman and
    Lipu, M.S. Hossain and
    Ali, Nazakat and
    Kabir, Md Alamgir
  },
  journal = {Journal of Energy Storage},
  year    = {2026}
}
```

### Dataset Citations

- **NASA Ames Battery Dataset:**
  > Saha, B., & Goebel, K. (2007). *Battery Data Set*. NASA Ames Prognostics Center of Excellence (PCoE), Moffett Field, CA.

- **CALCE Battery Dataset:**
  > Center for Advanced Life Cycle Engineering (CALCE), University of Maryland. *Battery Data Set (CS2 Series)*. Available at: https://calce.umd.edu/battery-data.

---

## License

This repository is distributed under the [MIT License](LICENSE).
