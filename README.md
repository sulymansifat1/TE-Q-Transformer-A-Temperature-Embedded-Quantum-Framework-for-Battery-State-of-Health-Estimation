# TE-Q-Transformer: A Temperature-Embedded Quantum Framework for Battery State-of-Health Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35%2B-blueviolet.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official research repository for **TE-Q-Transformer**, a hybrid quantum-classical deep learning framework for lithium-ion battery state-of-health (SOH) estimation. The framework integrates Arrhenius degradation kinetics for elevated-temperature solid electrolyte interphase (SEI) growth and sub-ambient lithium plating into a four-qubit simulated variational quantum circuit and a temporal Conv1D-Transformer backbone.

---

## Authors

- **Zadid Al Lisan** — *Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh* ([ORCID: 0009-0001-1801-051X](https://orcid.org/0009-0001-1801-051X))  
- **Md. Sulyman Islam Sifat** — *Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh* ([ORCID: 0009-0005-6377-8973](https://orcid.org/0009-0005-6377-8973))  
- **M.S. Hossain Lipu** — *School of Electrical, Computer and Telecommunications Engineering, University of Wollongong, NSW 2522, Australia* ([ORCID: 0000-0001-9060-4454](https://orcid.org/0000-0001-9060-4454))  
- **Nazakat Ali** — *ABB, Vasteras, Sweden* ([ORCID: 0000-0002-3875-812X](https://orcid.org/0000-0002-3875-812X))  
- **Md Alamgir Kabir** *(Corresponding Author)* — *Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh* ([ORCID: 0000-0002-7136-6339](https://orcid.org/0000-0002-7136-6339), Email: `kabir.cse@diu.edu.bd`)

---

## Overview

Battery State of Health (SOH) is defined as the ratio of current discharge capacity $C_k$ to the initial nominal capacity $C_0$:
$$\text{SOH}_k = \frac{C_k}{C_0}$$

Operating temperature fundamentally governs battery aging mechanisms:
- **Elevated Temperatures ($T > 25\,^\circ\text{C}$):** Accelerated side reactions between active materials and electrolyte drive solid electrolyte interphase (SEI) layer growth and solvent decomposition.
- **Sub-Ambient Temperatures ($T < 25\,^\circ\text{C}$):** Retarded lithium-ion transport and solid-state diffusion induce electrode polarization, precipitating metallic lithium plating during charging.

### The Scientific Problem
Conventional deep sequential models treat temperature as an ordinary numerical channel passed through standard min-max or z-score normalization. While numerical normalization rescales feature magnitudes, it discards the non-linear kinetic pathways that govern thermal degradation.

### The TE-Q-Transformer Solution
TE-Q-Transformer formulates a physics-guided temperature encoding:
$$\phi(T) = \exp\left(\frac{E_{a,\text{sei}}}{R}\left(\frac{1}{T_{\text{ref}}} - \frac{1}{T_K}\right)\right) + \exp\left(\frac{E_{a,\text{pl}}}{R}\left(\frac{1}{T_K} - \frac{1}{T_{\text{ref}}}\right)\right)$$
$$\theta_{\text{temp}} = \frac{\pi \cdot \phi(T)}{4}$$
where $E_{a,\text{sei}}$ and $E_{a,\text{pl}}$ are trainable empirical parameters, $R = 8.314\,\text{J/(mol}\cdot\text{K)}$, and $T_{\text{ref}} = 298.15\,\text{K}$.

This physical angle is coupled with scaled voltage, current, and normalized intra-cycle time into a **four-qubit variational quantum circuit** simulated in PennyLane (`default.qubit`). The circuit employs a rich entangler (RZ rotations, CNOT ladder, IsingZZ couplings, Toffoli gates, and Hadamard layers) with Pauli-Z expectation value readout $\langle \sigma_z^i \rangle$, projected into a 64-dimensional latent space, smoothed via Conv1D, and processed by a multi-layer Transformer encoder backbone.

> [!NOTE]
> **Quantum Simulation Note:** The quantum components are simulated classically via PennyLane. We claim **no** quantum hardware advantage or fault-tolerant quantum supremacy; rather, the parameterized circuit functions as a non-linear feature transformation layer.

---

## Key Contributions

1. **Physics-Guided Temperature Representation:** Formulates a dual-mechanism Arrhenius encoding embedding high-temperature SEI growth and low-temperature lithium plating directly into feature space, outperforming conventional linear temperature scaling by 16.5% in macro RMSE.
2. **Controlled Component Analysis:** Systematically decouples the empirical contributions of physics encoding, thermal mechanisms, the quantum circuit layer (benchmarked against a parameter-matched classical MLP), and temporal modules.
3. **Rigorous Cross-Cell and External Evaluation:** Evaluated under a fixed NASA Ames held-out-cell protocol for unseen-cell generalization and short temporal extrapolation, complemented by an honest zero-shot transfer test on external CALCE prismatic cells.

---

## Repository Structure

```
TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation/
│
├── README.md                                  # Complete research documentation & user guide
├── LICENSE                                    # MIT License
├── requirements.txt                           # Minimal, verified project dependencies
├── .gitignore                                 # Git rules excluding caches and raw large data
│
├── datasets/
│   ├── README.md                              # Dataset provenance, contracts, and formats
│   ├── NASA/
│   │   └── processed/                         # Processed NASA arrays [N, 512, 4] & scalers (~7.5 MB)
│   └── CALCE/
│       ├── README.md                          # CALCE source provenance & download guide
│       └── processed/                         # Processed CALCE arrays [N, 512, 4] (~32 MB)
│
├── models/
│   ├── __init__.py                            # Top-level model registry
│   ├── proposed/
│   │   ├── __init__.py
│   │   └── te_q_transformer.py                # Authoritative 92,554-param TE-Q-Transformer
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
│   │   └── all_models_benchmark.ipynb         # Proposed model + all 10 scored baselines
│   ├── ablation/
│   │   └── ablation_study.ipynb               # E02 (Raw vs Physics), E03 (Mechanisms), E04 (Q vs C)
│   └── generalization/
│       └── generalization_study.ipynb         # Unseen cells, Temporal extrapolation, CALCE zero-shot
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── nasa_loader.py                     # NASA PyTorch dataset, dataloaders, scalers
│   │   └── calce_loader.py                    # CALCE CS2 unscaled loader
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py                         # RMSE, MAE, MAPE, R2, MaxE calculations
│   │   └── zero_shot.py                       # Zero-shot cross-dataset evaluation engine
│   └── models/
│       └── teq_transformer.py                 # Backward-compatibility alias to models.proposed
│
├── results/
│   ├── README.md                              # Provenance of experimental tables and figures
│   ├── tables/                                # Benchmark, ablation, and generalization CSVs
│   ├── figures/                               # Parity and degradation trajectory plots
│   └── logs/                                  # Audit reports & JSON evaluation logs
│
└── tests/
    ├── test_models.py                         # Forward pass verification [2, 512, 4] -> [2]
    ├── test_datasets.py                       # Verifies dataset file existence and tensor shapes
    └── test_scaling.py                        # Verifies non-leakage physical scaling contract
```

---

## Experimental Protocol

### NASA Ames Protocol
- **Cycle Sequence Length:** Exactly 512 points per discharge cycle $(V, I, T_C, t_{\text{norm}})$.
- **Feature Order:**
  - Index 0: Voltage ($V$)
  - Index 1: Current ($A$)
  - Index 2: Temperature ($T_C$ in Celsius, kept strictly unscaled for Arrhenius calculations)
  - Index 3: Normalized time ($t_{\text{norm}} \in [0, 1]$ within discharge cycle)
- **Held-Out Partitioning:**
  - **Training Pool (660 cycles):** `B0005`, `B0006`, `B0007` (24°C), `B0029`, `B0030`, `B0031` (4°C), and first 70% of `B0053` (44°C).
  - **Unseen Cells Test:** `B0018` (132 cycles, 24°C) and `B0032` (39 cycles, 4°C).
  - **Temporal Extrapolation Test:** Final 30% of `B0053` (16 cycles, 44°C).
- **Non-Leakage Scaling:** `MinMaxScaler` is fitted exclusively on the training pool for features `(0, 1, 3)`. Test cells are never seen by the scaler.
- **Important Scientific Distinction:** This is an **unseen-cell** and **temporal extrapolation** benchmark, **NOT** an unseen-temperature benchmark, because training cells already span 4°C, 24°C, and 44°C.

---

## Main Experimental Results

### 1. NASA Ames Benchmark (Locked 11-Model Comparison)
All models evaluated under the identical sequence length (512), multi-temperature train/test split, and non-leakage feature scaling:

| Model | Family | Macro RMSE | Macro MAE | Macro $R^2$ | Trainable Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TE-Q-Transformer (Proposed)** | Quantum-Physics-Transformer | **0.0168** | **0.0141** | **0.870** | **92,554** |
| QNN-GRU | Quantum-Recurrent | 0.0304 | 0.0248 | 0.258 | 29,533 |
| iTransformer | Inverted Attention | 0.0341 | 0.0298 | 0.533 | 137,473 |
| Transformer | Classical Attention | 0.0346 | 0.0287 | 0.218 | 80,257 |
| PatchTST | Patch Attention | 0.0387 | 0.0333 | 0.321 | 109,962 |
| LSTM | Classical Recurrent | 0.0416 | 0.0367 | -0.076 | 71,105 |
| QLSTM | Quantum-Recurrent (Gate VQC) | 0.0445 | 0.0386 | -0.336 | 37,853 |
| CNN1D | Convolutional | 0.0509 | 0.0457 | -3.191 | 47,041 |
| TCN | Causal Dilated Conv | 0.0535 | 0.0452 | -0.341 | 91,841 |
| DLinear | Series Decomposition Linear | 0.0624 | 0.0568 | -2.232 | 1,031 |
| GRU | Classical Recurrent | 0.0642 | 0.0565 | -1.680 | 54,465 |

#### Per-Cell Breakdown for TE-Q-Transformer:
- **Cell B0018 (Unseen Cell, 24°C):** $\text{RMSE} = 0.0271$, $\text{MAE} = 0.0223$, $R^2 = 0.894$
- **Cell B0032 (Unseen Cell, 4°C):** $\text{RMSE} = 0.0119$, $\text{MAE} = 0.0105$, $R^2 = 0.902$
- **Cell B0053_test (Temporal Extrapolation, 44°C):** $\text{RMSE} = 0.0113$, $\text{MAE} = 0.0095$, $R^2 = 0.814$

---

### 2. Controlled Ablation Studies

#### A. Conventional vs. Physics-Guided Temperature Encoding (E02)
- **Conventional Linear Temperature Scaling:** Macro $\text{RMSE} = 0.0201$, $R^2 = 0.723$
- **Physics-Guided Arrhenius Temperature:** Macro $\text{RMSE} = 0.0168$, $R^2 = 0.870$
- **Relative Error Reduction: 16.5%**

#### B. Degradation Mechanism Decomposition (E03)
- **SEI-Only:** Macro $\text{RMSE} = 0.0330$
- **Plating-Only:** Macro $\text{RMSE} = 0.0249$
- **Combined Dual-Mechanism:** Macro $\text{RMSE} = 0.0215$ (Both mechanisms are essential to span low and high temperatures).

#### C. Quantum vs. Parameter-Matched Classical Representation (E04)
- **Parameter-Matched Classical MLP:** Macro $\text{RMSE} = 0.0431$, $R^2 = -0.746$
- **4-Qubit Variational Quantum Circuit:** Macro $\text{RMSE} = 0.0215$, $R^2 = 0.610$

---

### 3. CALCE Zero-Shot External Transfer (Honest Reporting)
The NASA-trained TE-Q-Transformer was tested directly on four CALCE prismatic LiCoO₂ cells (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`) with **frozen weights and frozen NASA scaler**:

| Metric | CS2_35 | CS2_36 | CS2_37 | CS2_38 | Macro Average | Pooled Combined |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RMSE** | 0.2933 | 0.3731 | 0.3483 | 0.3061 | **0.3302** | **0.3325** |
| **MAE** | 0.2415 | 0.2937 | 0.2803 | 0.2524 | **0.2670** | **0.2676** |
| **$R^2$** | -2.070 | -1.567 | -1.776 | -2.115 | **-1.882** | **-1.789** |

> [!WARNING]
> **Scientific Interpretation:** CALCE transfer is an **external zero-shot domain shift test**, not a successful benchmark. The negative $R^2$ score reflects severe degradation tracking limitations due to differing cell chemistry (prismatic 1.1 Ah vs cylindrical 2.0 Ah) and cycling protocols. We openly report this as a scientific limitation rather than claiming cross-dataset superiority.

---

## Running the Project on Kaggle

Follow these exact steps to run any notebook on Kaggle:

### Step 1: Create a Kaggle Notebook
1. Go to [kaggle.com](https://www.kaggle.com/) and click **New Notebook**.
2. Under **Notebook settings** (right sidebar), select **Accelerator: GPU P100** or **GPU T4 x2**.
3. Set **Internet: ON**.

### Step 2: Clone the Repository
In the first notebook cell, execute:
```bash
!git clone https://github.com/sulymansifat1/TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation.git
```

### Step 3: Change Working Directory
```bash
%cd /kaggle/working/TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation
```

### Step 4: Install Dependencies
```bash
!pip install -r requirements.txt
```

### Step 5: Verify the Root Configuration Cell
Every notebook begins with a self-adaptive path configuration cell. On Kaggle, it automatically resolves `REPO_ROOT`:
```python
import sys
from pathlib import Path

REPO_NAME = "TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation"
REPO_ROOT = Path("/kaggle/working") / REPO_NAME
if not REPO_ROOT.exists():
    REPO_ROOT = Path.cwd().resolve()

DATA_ROOT = REPO_ROOT / "datasets"
MODEL_ROOT = REPO_ROOT / "models"
RESULT_ROOT = REPO_ROOT / "results"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"Repository Root: {REPO_ROOT}")
```

### Step 6: Run Desired Research Notebook
Open or run any of the three research notebooks:
- **Benchmark:** `notebooks/benchmark/all_models_benchmark.ipynb`
- **Ablations:** `notebooks/ablation/ablation_study.ipynb`
- **Generalization:** `notebooks/generalization/generalization_study.ipynb`

### Kaggle Dataset Attachment Mode (Optional)
If you prefer attaching datasets via Kaggle Datasets instead of Git:
1. Upload the `datasets/NASA/processed` folder as a private Kaggle Dataset named `nasa-battery-dataset`.
2. In the configuration cell, set:
   ```python
   DATA_ROOT = Path("/kaggle/input/nasa-battery-dataset")
   ```

---

## Reproducibility

### Local Setup
```bash
# Clone the repository
git clone git@github.com:sulymansifat1/TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation.git
cd TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run unit tests
python -m unittest discover -s tests -p "test_*.py"
```

### Reproducing Benchmark Metrics
To evaluate the pre-trained weights without retraining:
```bash
python -c "
from models.proposed import TEQTransformer, rich_entangler_config
from src.data.nasa_loader import get_nasa_dataloaders
from src.eval.metrics import calculate_metrics, calculate_macro_metrics
from pathlib import Path
import torch, numpy as np

device = torch.device('cpu')
model = TEQTransformer(rich_entangler_config())
state_dict = torch.load('artifacts/nasa/01_rich_entangler/nasa_teq_transformer_best.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

_, test_loaders, _ = get_nasa_dataloaders(Path('datasets/NASA/processed'))
results = {}
with torch.no_grad():
    for cell, loader in test_loaders.items():
        preds = [model(bx).numpy() for bx, _ in loader]
        actuals = [by.numpy() for _, by in loader]
        m = calculate_metrics(np.concatenate(actuals), np.concatenate(preds))
        results[cell] = m
        print(f'{cell}: RMSE={m[\"RMSE\"]:.5f}, MAE={m[\"MAE\"]:.5f}, R2={m[\"R2\"]:.5f}')

macro = calculate_macro_metrics(list(results.values()))
print(f'\\nNASA MACRO: RMSE={macro[\"RMSE\"]:.5f}, MAE={macro[\"MAE\"]:.5f}, R2={macro[\"R2\"]:.5f}')
"
```

---

## Scientific Limitations

1. **Coupled Operating Temperatures:** In the NASA Ames dataset, ambient temperatures are coupled with specific cell runs. Because 4°C, 24°C, and 44°C are represented during training, this work does **not** demonstrate unseen-temperature generalization.
2. **Cross-Chemistry Domain Shift:** Zero-shot transfer from NASA cylindrical LiCoO₂ cells to CALCE prismatic cells shows poor degradation tracking ($R^2 = -1.789$), confirming that cell geometry and chemistry differences require dedicated domain adaptation.
3. **Simulated Quantum Execution:** Quantum circuits are simulated using state-vector linear algebra on classical CPU/GPU runtimes via PennyLane. No physical QPU fault tolerance or quantum advantage is claimed.
4. **Empirical Parameter Calibration:** Activation energies $E_{a,\text{sei}}$ and $E_{a,\text{pl}}$ are trainable empirical parameters initialized from literature values (30 kJ/mol), not physically measured reaction constants.

---

## Citation

If you use this code, model architectures, or experimental methodology in your research, please cite our paper:

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
