# TE-Q-Transformer: A Temperature-Embedded Quantum Framework for Battery State-of-Health Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35%2B-blueviolet.svg)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TE-Q-Transformer is a hybrid quantum-classical deep learning framework for lithium-ion battery state-of-health (SOH) estimation. The codebase addresses cycle-level capacity degradation estimation by integrating physics-guided Arrhenius temperature transformations directly into a parameterized quantum circuit representation. The core implementation couples temperature-dependent degradation kinetics with a four-qubit simulated variational circuit, a 1D convolutional smoothing layer, and a multi-layer Transformer encoder backbone. This repository provides the complete model implementation, baseline architectures, data loading pipelines, evaluation routines, unit tests, and notebook workflows to reproduce the experimental pipeline.

---

## Repository Contents

- **Proposed Architecture (`models/proposed/`)**: Implementation of the TE-Q-Transformer architecture and its configuration factory (`TEQTransformer`, `TEQTransformerConfig`, `rich_entangler_config`).
- **Baseline Models (`models/baselines/`)**: Implementations of 10 baseline architectures (LSTM, GRU, CNN1D, TCN, DLinear, Transformer, PatchTST, iTransformer, QLSTM, QNN-GRU) with a unified registry factory (`get_baseline_model`, `list_baselines`).
- **Data Loaders & Preprocessing (`src/data/`)**: PyTorch Dataset classes, data loader generators, and zero-leakage feature scaling routines for NASA and CALCE battery datasets, plus CALCE raw-to-processed extraction scripts.
- **Evaluation Utilities (`src/eval/`)**: Standard regression metrics calculation (`RMSE`, `MAE`, `MAPE`, `R2`, `MaxE`) and cross-cell evaluation routines.
- **Workflow Notebooks (`notebooks/`)**: Self-contained Jupyter notebooks for full baseline benchmarking, ablation studies, generalization experiments, and CALCE cross-cell evaluation.
- **Unit Test Suite (`tests/`)**: Automated tests verifying model forward passes, dataset array presence, tensor shape compliance, and scaling contracts.

---

## Model Architecture

The TE-Q-Transformer architecture processes multi-channel cycle sequences through the following sequential stages:

```
Input [B, 512, 4]
       │
       ├── Channel 2 (Temp C) ──> Arrhenius Transform ──> θ_temp
       └── Channels 0, 1, 3 (V, I, t) ─────────────────> θ_v, θ_i, θ_t
                                                               │
                                                               ▼
                                                  4-Qubit Variational Circuit
                                                  (PennyLane default.qubit)
                                                               │
                                                               ▼
                                                  Pauli-Z Expectation [B, 512, 4]
                                                               │
                                                               ▼
                                                  Linear Projection (d_model = 64)
                                                               │
                                                               ▼
                                                  Conv1D Temporal Smoothing (k = 3)
                                                               │
                                                               ▼
                                                  Prepend [CLS] + Positional Encoding
                                                               │
                                                               ▼
                                                  3-Layer Transformer Encoder
                                                               │
                                                               ▼
                                                  Extract [CLS] Token
                                                               │
                                                               ▼
                                                  MLP Regression Head
                                                               │
                                                               ▼
                                                  SOH Output [B]
```

1. **Input Sequence**: Each battery discharge cycle is represented as a fixed-length tensor of shape `[B, 512, 4]` corresponding to Voltage, Current, Temperature in Celsius, and Normalized Time within the discharge cycle.
2. **Physics-Guided Arrhenius Temperature Representation**: Temperature values ($T_C$) are converted to Kelvin ($T_K = T_C + 273.15$) and passed through a dual Arrhenius kinetic transformation:
   $$\phi(T) = \exp\left(\frac{E_{a,\text{sei}}}{R}\left(\frac{1}{T_{\text{ref}}} - \frac{1}{T_K}\right)\right) + \exp\left(\frac{E_{a,\text{pl}}}{R}\left(\frac{1}{T_K} - \frac{1}{T_{\text{ref}}}\right)\right)$$
   $$\theta_{\text{temp}} = \frac{\pi \cdot \phi(T)}{4}$$
   where $R = 8.314\,\text{J/(mol}\cdot\text{K)}$, $T_{\text{ref}} = 298.15\,\text{K}$, and $E_{a,\text{sei}}$, $E_{a,\text{pl}}$ are trainable empirical activation parameters initialized from electrochemical literature.
3. **Four-Qubit Quantum Feature Map**: Implemented via PennyLane using the `default.qubit` state-vector simulator. The 4 features are mapped to rotational angles across 4 qubits. The circuit applies a rich entangler consisting of RZ rotations, a CNOT ladder, IsingZZ couplings, multi-qubit Toffoli gates, and Hadamard layers.
4. **Quantum Readout**: Computes Pauli-Z expectation values $\langle \sigma_z^i \rangle$ on all four qubits, returning a 4-dimensional vector per time step.
5. **Linear Projection**: Projects the 4-dimensional quantum representation into the latent model dimension ($d_{\text{model}} = 64$).
6. **Conv1D Temporal Smoothing**: Applies a 1D convolution (`kernel_size=3`, `padding=1`) along the sequence length dimension.
7. **Transformer Encoder Backbone**: Prepends a learnable `[CLS]` token, adds sinusoidal positional encodings, and processes the sequence through 3 Transformer encoder layers (`nhead=2`, `dim_feedforward=64`, GELU activation, pre-layer normalization).
8. **SOH Regression Head**: Extracts the transformed `[CLS]` token representation and passes it through an MLP head (`Linear(64, 64) -> GELU -> Dropout -> Linear(64, 1)`) to predict a scalar State of Health ($C_k / C_0$) for the cycle.

*Note on Quantum Simulation*: All quantum operations are executed classically via state-vector simulation in PennyLane. The parameterized quantum circuit functions as a non-linear feature transformation layer.

---

## Repository Structure

```
.
├── README.md                                  # Repository documentation and reproduction guide
├── LICENSE                                    # MIT License
├── requirements.txt                           # Project dependencies
├── .gitignore                                 # Git ignore patterns
│
├── datasets/
│   ├── README.md                              # Dataset documentation and contracts
│   ├── NASA/
│   │   └── processed/                         # Processed NASA arrays [N, 512, 4] and fitted scalers
│   └── CALCE/
│       ├── README.md                          # CALCE dataset provenance and details
│       └── processed/                         # Processed CALCE arrays [N, 512, 4] and metadata
│
├── models/
│   ├── __init__.py                            # Model package entry point
│   ├── proposed/
│   │   ├── __init__.py
│   │   └── te_q_transformer.py                # TE-Q-Transformer architecture and config
│   └── baselines/
│       ├── __init__.py                        # Baseline registry and factory get_baseline_model()
│       ├── cnn1d.py                           # 1D Temporal CNN baseline
│       ├── tcn.py                             # Causal Dilated TCN baseline
│       ├── dlinear.py                         # Series decomposition linear baseline
│       ├── transformer.py                     # Classical Multi-Head Attention baseline
│       ├── patchtst.py                        # Patch Time Series Transformer baseline
│       ├── itransformer.py                    # Inverted Variate Transformer baseline
│       ├── qlstm.py                           # Quantum LSTM baseline
│       ├── qnn_gru.py                         # Quantum Neural Network + GRU baseline
│       ├── lstm.py                            # Standard LSTM baseline
│       └── gru.py                             # Standard GRU baseline
│
├── notebooks/
│   ├── benchmark/
│   │   └── all_models_benchmark.ipynb         # NASA benchmark suite (proposed + 10 baselines)
│   ├── ablation/
│   │   └── ablation_study.ipynb               # Architectural component ablation experiments
│   ├── generalization/
│   │   └── generalization_study.ipynb         # Unseen-cell and temporal extrapolation experiments
│   ├── statistical_significance_test_NASA.ipynb
│   ├── integrated_gradients_explainability_NASA.ipynb
│   ├── baselineComparison_Nasa.ipynb          # Stored-prediction NASA baseline table
│   └── Calce.ipynb                            # CALCE cross-cell and temporal extrapolation evaluation
├── data/
│   └── nasa_manuscript/                       # Held-out predictions and IG records used by the notebooks
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── nasa_loader.py                     # NASA dataset loaders and zero-leakage scaler
│   │   ├── calce_loader.py                    # CALCE dataset loaders
│   │   └── calce/
│   │       ├── __init__.py
│   │       └── preprocess.py                  # CALCE raw Arbin Excel extraction script
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py                         # RMSE, MAE, MAPE, R2, MaxE calculations
│       ├── zero_shot.py                       # Cross-cell evaluation helper
│       └── nasa_calce_zero_shot.py            # Standalone evaluation script
│
└── tests/
    ├── test_models.py                         # Forward pass verification for all architectures
    ├── test_datasets.py                       # Dataset file existence and tensor shape tests
    ├── test_scaling.py                        # Non-leakage feature scaling contract test
    └── test_calce_preprocess.py               # CALCE resampling and validation tests
```

---

## Environment & Installation

### Requirements
- Python 3.10 or higher
- PyTorch >= 2.0.0
- PennyLane >= 0.35.0
- NumPy >= 1.24.0, < 2.0.0
- SciPy >= 1.10.0
- Pandas >= 2.0.0
- Scikit-Learn >= 1.3.0

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sulymansifat1/TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation.git
   cd TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Linux/macOS:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows:
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the installation by running the test suite:**
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```

---

## Dataset Preparation & Contracts

The repository processes battery cycle measurements into fixed-length arrays matching a strict input contract.

### Expected Tensor Format

- **Input Array Shape**: `[B, 512, 4]` (Batch size $B$, sequence length 512, 4 channels).
- **Target Array Shape**: `[B]` (Scalar float32 representing $SOH_k = C_k / C_0$).
- **Channel Ordering**:
  - Index 0: **Voltage** ($V$)
  - Index 1: **Current** ($A$)
  - Index 2: **Temperature** ($T_C$ in degrees Celsius)
  - Index 3: **Normalized Time** ($t_{\text{norm}} \in [0, 1]$ within the discharge cycle)

### Feature Scaling & Non-Leakage Contract

- Channels 0, 1, and 3 are normalized using `sklearn.preprocessing.MinMaxScaler` fitted **exclusively on the training cell partition**.
- Channel 2 (**Temperature**) is kept strictly unscaled in degrees Celsius throughout preprocessing and data loading. This ensures that physical temperatures can be converted to Kelvin ($T_K = T_C + 273.15$) for Arrhenius kinetic equations.
- Test cell cycles are transformed using the fitted training scaler without refitting.

### NASA Ames Battery Dataset

- **Source**: NASA Prognostics Center of Excellence (PCoE) Battery Data Set.
- **Chemistry**: Commercial 18650 LiCoO₂ / graphite cylindrical cells (nominal capacity 2.0 Ah).
- **Stored Location**: `datasets/NASA/processed/` (`{cell}_X.npy`, `{cell}_soh.npy`).
- **Partitioning**:
  - **Training Pool (660 cycles)**: `B0005`, `B0006`, `B0007` (24°C), `B0029`, `B0030`, `B0031` (4°C), and first 70% of `B0053` (44°C, 37 cycles).
  - **Held-Out Unseen Cells (171 cycles)**: `B0018` (24°C, 132 cycles) and `B0032` (4°C, 39 cycles).
  - **Temporal Extrapolation (16 cycles)**: Final 30% of `B0053` (44°C).

### CALCE Battery Dataset

- **Source**: Center for Advanced Life Cycle Engineering (CALCE), University of Maryland.
- **Chemistry**: Prismatic LiCoO₂ CS2 series (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`, nominal capacity 1.1 Ah).
- **Stored Location**: `datasets/CALCE/processed/` (`{cell}_X_unscaled.npy`, `{cell}_soh.npy`, `{cell}_cycle.npy`).
- **Partitioning**:
  - **Training Split (2,309 cycles)**: `CS2_35`, `CS2_36`, and the first 70% of `CS2_37`.
  - **Held-Out Unseen Cell Test Split (966 cycles)**: `CS2_38`.
  - **Temporal Extrapolation Test Split (275 cycles)**: Remaining 30% of `CS2_37` (`CS2_37_test`).
- **Preprocessing from Raw Files (Optional)**:
  Processed CALCE `.npy` arrays are tracked in the repository. To re-extract from raw Arbin Excel logs:
  ```bash
  python -m src.data.calce.preprocess --raw-root <path_to_raw_calce> --output-dir datasets/CALCE/processed
  ```

---

## Usage Guide

### 1. Instantiating the Proposed Model

```python
import torch
from models.proposed import TEQTransformer, rich_entangler_config

# Load standard model configuration
config = rich_entangler_config()
model = TEQTransformer(config)

# Dummy input: batch_size=2, sequence_length=512, channels=4
# Channels: (Voltage, Current, Temperature_C, Normalized_Time)
x = torch.randn(2, 512, 4)
x[:, :, 2] = 24.0  # Temperature in Celsius

# Forward pass: [2, 512, 4] -> [2]
soh_pred = model(x)
print(f"Output shape: {soh_pred.shape}")  # torch.Size([2])
```

### 2. Instantiating Baseline Models

```python
import torch
from models.baselines import get_baseline_model, list_baselines

# List available baseline architectures
print(list_baselines())
# ['LSTM', 'GRU', 'CNN1D', 'TCN', 'DLinear', 'Transformer', 'PatchTST', 'iTransformer', 'QLSTM', 'QNN_GRU']

# Instantiate a baseline model by name
model = get_baseline_model("Transformer")

x = torch.randn(2, 512, 4)
output = model(x)
print(f"Baseline output shape: {output.shape}")  # torch.Size([2])
```

### 3. Loading Data

#### NASA Ames DataLoaders
```python
from pathlib import Path
from src.data.nasa_loader import get_nasa_dataloaders

data_dir = Path("datasets/NASA/processed")
train_loader, test_loaders, scaler = get_nasa_dataloaders(data_dir, batch_size=8)

print(f"Training batches: {len(train_loader)} ({len(train_loader.dataset)} cycles)")
for cell_id, loader in test_loaders.items():
    print(f"Test partition {cell_id}: {len(loader.dataset)} cycles")
```

#### CALCE Cell Loading
```python
from pathlib import Path
from src.data.calce_loader import load_calce_cell, CALCE_CELL_IDS

calce_dir = Path("datasets/CALCE/processed")
for cell_id in CALCE_CELL_IDS:
    X, y, cycles = load_calce_cell(calce_dir, cell_id)
    print(f"{cell_id}: X shape {X.shape}, y shape {y.shape}, cycles {len(cycles)}")
```

### 4. Running Workflow Notebooks

The repository provides executable Jupyter notebooks under `notebooks/`:

| Notebook | Path | Description |
| :--- | :--- | :--- |
| **All Models Benchmark** | `notebooks/benchmark/all_models_benchmark.ipynb` | Trains and evaluates TE-Q-Transformer and all 10 baseline architectures on the NASA dataset. |
| **Ablation Study** | `notebooks/ablation/ablation_study.ipynb` | Trains architectural ablation variants (physics-guided vs. conventional scaling, individual degradation mechanisms, quantum vs. classical layers). |
| **Generalization Study** | `notebooks/generalization/generalization_study.ipynb` | Evaluates generalization on unseen NASA cells (`B0018`, `B0032`), temporal extrapolation on `B0053`, and cross-dataset evaluation. |
| **CALCE Evaluation** | `notebooks/Calce.ipynb` | Trains and evaluates TE-Q-Transformer on the CALCE CS2 cross-cell and temporal extrapolation partition (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`). |

---

## Running on Kaggle

Each notebook dynamically resolves repository paths and can be executed in Kaggle environments:

1. Create a new notebook on [Kaggle](https://www.kaggle.com/).
2. Configure **Accelerator** under Notebook Settings:
   - Select **GPU P100** or **GPU T4 x2** (Quantum circuit simulation over 512-point cycles executes faster with CUDA; baseline notebooks enforce `cuda:0`).
   - If using CPU, ensure batch sizes and sequence subsets are adjusted accordingly.
3. Set **Internet** to **ON** in the notebook settings panel.
4. In the initial notebook cell, clone the repository and install requirements:
   ```bash
   !git clone https://github.com/sulymansifat1/TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation.git
   %cd TE-Q-Transformer-A-Temperature-Embedded-Quantum-Framework-for-Battery-State-of-Health-Estimation
   !pip install -r requirements.txt
   ```
5. **Datasets**: The processed `.npy` files for NASA and CALCE are tracked directly in the Git repository under `datasets/`, so no external dataset mount is required for standard runs.
6. Open and run the target notebook from the `notebooks/` directory.

---

## Testing

The test suite validates model construction, dataset integrity, and preprocessing rules:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### What the Tests Verify

- **`tests/test_models.py`**:
  - Instantiates `TEQTransformer` with `rich_entangler_config()` and confirms the forward pass transforms `[2, 512, 4]` to `[2]` without NaN outputs.
  - Instantiates all 10 baseline models and confirms forward pass execution.
  - Verifies the `get_baseline_model` factory against all registered baseline names.
- **`tests/test_datasets.py`**:
  - Verifies the physical presence of all required `.npy` arrays for NASA (`B0005`, `B0006`, `B0007`, `B0018`, `B0029`, `B0030`, `B0031`, `B0032`, `B0053`).
  - Verifies the physical presence of processed CALCE files (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`).
  - Validates that array dimensions match `[N, 512, 4]` for inputs and `[N]` for targets.
- **`tests/test_scaling.py`**:
  - Verifies that `transform_with_scaler` maps channels 0, 1, and 3 to `[0, 1]`.
  - Asserts that channel 2 (temperature in Celsius) remains unmodified to preserve physical Arrhenius dynamics.
- **`tests/test_calce_preprocess.py`**:
  - Tests the discharge phase resampling function ensuring output shapes of `(512, 4)` and endpoint preservation.
  - Tests validation filter rules for identifying incomplete discharge cycles.

---

## Expected Outputs

When executing training and evaluation scripts or notebooks in this repository, the following artifacts are produced:

- **Model Predictions**: Tensors of shape `[B]` representing scalar SOH estimates ($C_k / C_0$) corresponding to each input cycle window.
- **Training Logs**: Output logs reporting Mean Squared Error (MSE) training loss per epoch, learning rate scheduler updates, and periodic validation metrics.
- **Evaluation Metrics**: Regression metric summaries computed via `src.eval.metrics.calculate_metrics`:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Mean Absolute Percentage Error (MAPE)
  - Coefficient of Determination ($R^2$)
  - Maximum Error (MaxE)
- **Model Checkpoints**: Serialized PyTorch state dictionaries (`.pt` files) saved to designated checkpoint directories.
- **Visualizations**: Generated matplotlib plots showing training loss trajectories, predicted versus measured SOH degradation curves across cycles, and error distributions.

---

## Reproducibility Notes

To ensure consistent experimental reproduction across environments:

- **Random Seed Handling**: Notebooks implement a centralized seeding routine (`seed_everything(seed=42)`) that fixes seeds for Python `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and sets `torch.backends.cudnn.deterministic = True`.
- **Consistent Tensor Dimensions**: All loaders and models enforce the `[B, 512, 4]` tensor contract and channel order `[Voltage, Current, Temperature_C, Normalized_Time]`.
- **Data Non-Leakage Contract**: `MinMaxScaler` parameters are computed exclusively on designated training partitions. Test sets are transformed using the fitted scaler and are never used to compute minimum or maximum bounds.
- **Preserved Physical Temperature**: Channel 2 is never rescaled; it remains in degrees Celsius so that absolute temperature ($T_K = T_C + 273.15$) is substituted into the Arrhenius equation.
- **Deterministic Dependencies**: Package versions are specified in `requirements.txt`. For quantum simulations, PennyLane state-vector routines are executed on fixed seed states.

---

## Dataset Citations

#### NASA Ames Battery Dataset
```bibtex
@misc{saha2007battery,
  title={Battery Data Set},
  author={Saha, Bhaskar and Goebel, Kai},
  year={2007},
  publisher={NASA Ames Prognostics Center of Excellence (PCoE)},
  url={https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/}
}
```

#### CALCE Battery Dataset
```bibtex
@misc{calce2011battery,
  title={CALCE Battery Data Set},
  author={{Center for Advanced Life Cycle Engineering (CALCE)}},
  year={2011},
  publisher={University of Maryland},
  url={https://calce.umd.edu/battery-data}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
