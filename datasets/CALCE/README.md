# CALCE Battery Dataset (CS2 Series)

This folder contains the processed CALCE battery arrays structured to match the NASA tensor contract for zero-shot cross-dataset evaluation.

---

## 1. Source & Citation

- **Source:** Center for Advanced Life Cycle Engineering (CALCE), University of Maryland.
- **URL:** [https://calce.umd.edu/battery-data](https://calce.umd.edu/battery-data)
- **Cell Details:**
  - Form factor: Prismatic
  - Nominal Capacity: 1.1 Ah
  - Cathode: LiCoO₂
  - Anode: Graphite
  - Test cells: `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`
  - Cycling: Room temperature 0.5C constant-current charge and discharge

---

## 2. Processed Files

The processed folder (`datasets/CALCE/processed/`) contains:

| File Pattern | Description | Shape / Type |
| :--- | :--- | :--- |
| `CS2_{id}_X_unscaled.npy` | 512-point interpolated discharge profiles $(V, I, T_C, t_{\text{norm}})$ | `[N, 512, 4]` float32 |
| `CS2_{id}_soh.npy` | Capacity ratio normalized SOH ($C_k / C_0$) | `[N]` float32 |
| `CS2_{id}_cycle.npy` | Cycle sequence numbers | `[N]` int32 |
| `CS2_{id}_capacity_ah.npy` | Raw measured discharge capacity (Ah) | `[N]` float32 |
| `CS2_{id}_duration_seconds.npy`| Discharge duration in seconds | `[N]` float32 |
| `CS2_{id}_n_raw_points.npy` | Number of raw logged points per cycle | `[N]` int32 |
| `preprocess_metadata.json` | Provenance parameters and interpolation settings | JSON |

---

## 3. How to Obtain Raw CALCE Data (Optional)

The raw Arbin tester `.xlsx` or `.csv` files are not tracked in Git due to storage limits. If you wish to reproduce raw-to-processed preprocessing:
1. Download the raw CS2 dataset archives (`CS2_35.zip`, `CS2_36.zip`, etc.) from the official CALCE data repository.
2. Place the archives in a local non-version-controlled directory (e.g. `datasets/CALCE/raw/`).
3. Run the preprocessing script:
   ```bash
   python -m src.data.calce_loader --raw-dir datasets/CALCE/raw --output-dir datasets/CALCE/processed
   ```
