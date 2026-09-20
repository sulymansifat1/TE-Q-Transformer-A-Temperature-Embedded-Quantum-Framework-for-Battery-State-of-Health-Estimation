# E07 Multi-Seed Robustness — Final Kaggle Run

**FINAL HANDOFF VERSION for Lisan.**

## Purpose

Evaluate five models across five fixed random seeds.

This is a multi-seed reproducibility / robustness experiment. It is **not** a statistical-significance test and **not** an unseen-temperature experiment.

## Models

1. TE-Q-Transformer
2. QNN-GRU
3. iTransformer
4. Transformer
5. PatchTST

## Seeds

42, 43, 44, 45, 46

## Total

**25 full training runs** (5 models × 5 seeds), sequential, one at a time.

## Evaluation

- **B0018** — unseen cell
- **B0032** — unseen cell
- **B0053 final 30% (cycles 37–52)** — temporal extrapolation (not an unseen cell)

Training uses B0005, B0006, B0007, B0029, B0030, B0031, and B0053 cycles 0–36.

## Instructions

1. Open Kaggle Notebook.
2. Select **GPU** accelerator (P100 or T4).
3. Turn **Internet ON** (PennyLane may need to install).
4. Upload / use the provided notebook.
5. Attach a dataset that contains the NASA files `B0005_X.npy` … `B0053_soh.npy`.
6. **Do NOT change `RUN_MODE`.**
7. **`RUN_MODE` is already `FULL`.**
8. Click **Run All**.
9. Wait for all 25 runs to finish. Pre-flight checks run first; if they pass, training starts automatically.
10. If Kaggle disconnects, Run All again. Only **verified complete** runs are skipped.
11. Download `E07_results` after completion (and `E07_MultiSeed_Robustness_Results.zip` if present).

**Do not modify model code, seed list, dataset split, preprocessing, or training configuration.**

**Do not run this notebook locally on CPU.** FULL mode will stop if CUDA is missing.

## Notes that must not be changed

- QNN-GRU `batch_size = 16` is the existing E05/E06 protocol, not an E07 retune.
- TE-Q `weight_decay = 0.01` is retained from the established campaign and is not tuned during E07.
- Best checkpoint is chosen from **training MSE only**, never from test RMSE.

## Expected outputs (`E07_results/`)

| File | Rows after FULL |
|---|---|
| `metrics/E07_run_metrics.csv` | 25 |
| `metrics/E07_cell_metrics.csv` | 75 |
| `metrics/E07_MultiSeed_Summary.csv` | 5 |
| `metrics/E07_MultiSeed_Cell_Summary.csv` | 15 |
| `metrics/E07_Seed_Wise_Detail.csv` | 25 |
| `predictions/E07_predictions.csv` | 4675 (25 × 187) |

Plus checkpoints, training history, provenance, and reconciliation report.

This notebook does **not** produce publication figures.
