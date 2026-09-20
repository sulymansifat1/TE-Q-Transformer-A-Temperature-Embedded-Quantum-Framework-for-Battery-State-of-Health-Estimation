# NASA → CALCE Zero-Shot Report

## Executive Summary

This experiment is **zero-shot cross-dataset/cross-cell evaluation** from a frozen NASA-trained TE-Q-Transformer to official CALCE CS2 cells CS2_35–CS2_38. NASA weights never saw any CALCE cell. No NASA experiment was retrained. No CALCE scaler was fitted. Oxford was not used.

This is **not** a cross-temperature experiment and **not** leave-one-battery-out. CALCE temperature is a constant 23 °C ambient fill because the official Excel logs have no temperature column. NASA room-temperature cells use a 24 °C ambient fill.

Overall pooled RMSE = **0.33252**, MAE = **0.26759**, R² = **−1.78869** on 3909 discharge windows. Unweighted mean ± std over the four CALCE cells: RMSE 0.33022 ± 0.03700, MAE 0.26699 ± 0.02420, R² −1.88221 ± 0.25815.

The frozen NASA model did **not** track CALCE capacity fade. Predictions stayed near **1.01** for every cell, so error grows as true SOH declines. This is a negative zero-shot transfer result under the stated protocol, not a NASA in-domain result.

## Frozen NASA Model

- Checkpoint: `artifacts/nasa/01_rich_entangler/nasa_teq_transformer_best.pth`
- Source zip: `Results/NASA_TEQ_Transformer_ablationOutputs (3).zip`
- Inner path: `nasa_results/ablation_study/01_rich_entangler/nasa_teq_transformer_best.pth`
- Variant: `01_rich_entangler` (`entangler_type='rich'`)
- Architecture: TE-Q-Transformer, input `[B, 512, 4]`, `d_model=64`, `n_heads=2`, `n_layers=3`, temporal Conv1d smoother, CLS pooling
- Weights loaded with `strict=True`; `model.eval()`; `torch.no_grad()`
- NASA notebooks, splits, and reported NASA metrics were not modified

Reported NASA ablation baseline (copied, not recomputed):

| Test cell | RMSE | MAE | R² |
|---|---:|---:|---:|
| B0018 | 0.02714 | 0.02227 | 0.89353 |
| B0032 | 0.01191 | 0.01050 | 0.90175 |
| B0053_test | 0.01128 | 0.00951 | 0.81450 |
| Macro | 0.01678 | 0.01409 | 0.86993 |

## NASA Preprocessing Used

- Scaler: frozen `MinMaxScaler` fitted on NASA training cycles only (`n=660`)
- Scaler file: `artifacts/nasa/nasa_train_scaler.pkl`
- Channels scaled: indices `(0, 1, 3)` = Voltage, Current, Time_norm
- Temperature left in physical °C
- CALCE values outside NASA train range were counted **before** clipping, then clipped to `[0, 1]` exactly as in the NASA notebook test transform
- In this run, **no CALCE value was out of range**, so clipping did not change any input

## CALCE Dataset

- Official source: CALCE Battery Research Group, https://web.calce.umd.edu/batteries/data/
- Cells: CS2_35, CS2_36, CS2_37, CS2_38 (all used as unseen external test cells)
- Chemistry: prismatic CS2, LiCoO2, nominal 1.1 Ah
- Protocol: CCCV charge 0.5C to 4.2 V; 1C discharge; cutoff 2.7 V
- Current: measured Arbin `Current(A)`, discharge negative (already NASA-compatible)
- Temperature: no column in Excel; constant 23 °C fill
- Existing CALCE notebook split and `calce_teq_transformer_best.pth` were **not** used

## CALCE Preprocessing

Complete 1C discharges were taken from chronological Excel files. A window never crosses a file or `Cycle_Index`. Each discharge was linearly resampled onto 512 phase points in `[0, 1]`. Feature order is `[Voltage, Current, Temperature_C, Time_norm]`. Per-cycle capacity is the Arbin `Discharge_Capacity(Ah)` increment over that discharge, not a possibly cumulative register maximum. Duplicate resampled windows and physically impossible capacities (>1.30 Ah) were dropped.

Accepted windows: CS2_35=880, CS2_36=968, CS2_37=1036, CS2_38=1025.

The first CS2_35 window matches the existing local `Dataset/calce/calce/CS2_35_X.npy[0]` exactly. The rebuilt set keeps later-life cycles below ~40% SOH that those older arrays appear to have dropped.

## SOH Construction

`SOH_i = C_i / C_0` for each cell, where `C_i` is the per-discharge ampere-hour increment (`abs(last − first)` of Arbin `Discharge_Capacity(Ah)` on that 1C step) and `C_0` is the first accepted discharge of the same cell. First-cycle `C_0` values are 1.1385, 1.1448, 1.1349, and 1.1395 Ah.

## Leakage Prevention

- NASA weights frozen; no NASA retraining
- NASA scaler frozen; no CALCE fit
- No CALCE-trained checkpoint
- No hyperparameter search on CALCE
- Oxford not substituted

## Out-of-range fractions before clip

| Cell | V below | V above | I below | I above | Time below | Time above |
|---|---:|---:|---:|---:|---:|---:|
| CS2_35 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CS2_36 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CS2_37 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| CS2_38 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Zero-Shot Protocol

```text
frozen NASA 01_rich_entangler checkpoint
        ↓
CALCE [N, 512, 4] + frozen NASA-train MinMax on (V, I, Time)
        ↓
model.eval() inference on CS2_35–38
```

## Overall Results

| Metric | Value |
|---|---:|
| N | 3909 |
| RMSE | 0.33252 |
| MAE | 0.26759 |
| MAPE (%) | 58.17943 |
| R² | −1.78869 |
| MaxE | 0.83796 |
| Mean actual SOH | 0.73976 |
| Mean predicted SOH | 1.00735 |

## Per-Cell Results

| Cell | N | RMSE | MAE | MAPE (%) | R² | MaxE | mean error |
|---|---:|---:|---:|---:|---:|---:|---:|
| CS2_35 | 880 | 0.29335 | 0.24147 | 42.56210 | −2.07047 | 0.80288 | −0.24147 |
| CS2_36 | 968 | 0.37309 | 0.29374 | 77.27467 | −1.56720 | 0.83796 | −0.29374 |
| CS2_37 | 1036 | 0.34833 | 0.28035 | 65.72572 | −1.77642 | 0.82838 | −0.28035 |
| CS2_38 | 1025 | 0.30610 | 0.25242 | 45.92685 | −2.11472 | 0.83436 | −0.25242 |

Mean ± std over cells: RMSE 0.33022 ± 0.03700; MAE 0.26699 ± 0.02420; R² −1.88221 ± 0.25815.

## Error Analysis

Mean error is actual − predicted. All four cells have **negative** mean error: the NASA model overestimates CALCE SOH.

Predictions are nearly constant. Across 3909 windows the predicted SOH range is about 0.978–1.041, with a pooled mean of 1.007. True SOH ranges from about 0.14 to 1.00. The first cycle of each cell is close (absolute error ≈ 0.01) only because the true label is also ≈ 1. After that, the frozen model does not follow the fade trajectory.

A constant predictor of 1.0 has pooled RMSE **0.32768**, slightly **better** than the NASA model (0.33252), because the model sits a little above 1. Restricting to SOH ≥ 0.4 (3521 windows, closer to the local CALCE `.npy` end-of-life cutoff) still yields pooled RMSE 0.25336 and R² −2.66; predictions remain ≈ 1.01. This is not a clipping artifact: every CALCE Voltage/Current/Time_norm value was inside the NASA training min/max.

Likely contributors, not isolated here:

- different cell format and chemistry (NASA 18650 ~2 Ah vs CALCE prismatic CS2 1.1 Ah)
- CALCE 1C current is almost constant at −1.1 A, which occupies a narrow band after NASA min–max scaling
- NASA temperature is a per-cell ambient fill and CALCE is a 23 °C fill, so the Arrhenius gate is nearly constant
- voltage-curve aging is present in the 512-point windows but is not sufficient, under these frozen weights, to move the SOH head off ~1

## Comparison with NASA

NASA in-domain ablation macro RMSE was 0.01678. The CALCE numbers are a **different experimental question** (unseen dataset, unseen cells, different form factor). They are not a replacement for the NASA table and were not used to select the NASA checkpoint.

- NASA ablation macro RMSE: 0.01678
- CALCE zero-shot pooled RMSE: 0.33252
- CALCE constant-1 baseline RMSE: 0.32768
- CALCE zero-shot mean-over-cells RMSE: 0.33022

## Limitations

- CALCE has no measured intra-cycle temperature; 23 °C is an ambient fill, analogous to NASA’s per-cell ambient fill
- Form factor and nominal capacity differ (NASA 18650 2 Ah vs CALCE prismatic 1.1 Ah)
- This test does not isolate temperature as a causal factor
- No CALCE value was clipped by the NASA scaler; domain shift is not an out-of-range scaling issue
- Late-life cycles below ~40% SOH are included; they worsen RMSE but do not create the near-constant prediction

## What This Experiment Demonstrates

Zero-shot **evaluation** of a frozen NASA TE-Q-Transformer on CALCE CS2 cells CS2_35–38 at ~23–24 °C, using measured CALCE current and a NASA-training-only scaler. Under this protocol the NASA model does not transfer capacity-fade tracking: it remains near SOH 1.

## What This Experiment Does NOT Demonstrate

- Successful NASA→CALCE fade tracking
- Cross-temperature generalisation
- Leave-one-battery-out within NASA or CALCE
- That a CALCE-trained TE-Q-Transformer would obtain this number
- Any change to previously reported NASA metrics

## Reproducibility

See commands at the end of this file. Random seed is unused because no training is performed.

## Files Created

- `reports/calce/nasa_calce_zero_shot_report.md`
- `reports/calce/calce_paper_ready_text.md`
- `reports/calce/nasa_calce_zero_shot_metrics.json`
- `reports/calce/nasa_calce_sensitivity.json`
- `reports/calce/figures/soh_trajectory_CS2_*.png/.pdf`
- `reports/calce/figures/error_violin_CS2_*.png/.pdf`
- `reports/calce/figures/parity_*.png/.pdf`
- `Dataset/calce/processed_nasa_contract/`
- `Dataset/calce/nasa_scaled/`
- `artifacts/nasa/01_rich_entangler/nasa_teq_transformer_best.pth`

## Exact Commands

```text
.\.tools\uv\uv.exe run --python 3.12 python scripts\extract_nasa_checkpoint.py
.\.tools\uv\uv.exe run --python 3.12 --with numpy --with pandas --with openpyxl --with python-calamine python -m src.data.calce.preprocess
.\.tools\uv\uv.exe run --python 3.12 --with numpy --with torch --with pennylane --with scikit-learn --with matplotlib python -m src.eval.nasa_calce_zero_shot --batch-size 8
```
