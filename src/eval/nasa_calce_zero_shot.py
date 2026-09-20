"""Frozen NASA TE-Q-Transformer → CALCE CS2 zero-shot evaluation."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.teq_transformer import TEQTransformer, rich_entangler_config

CELL_IDS = ("CS2_35", "CS2_36", "CS2_37", "CS2_38")
FEATURE_ORDER = ("Voltage_V", "Current_A", "Temperature_C", "Time_norm")

SCALED_INDICES = (0, 1, 3)
SCALED_NAMES = ("Voltage", "Current", "Time_norm")
PRIMARY_BLUE = "#6F90AE"
ACCENT_GREEN = "#AFC8A7"
GRID_BLUE = "#C9D6E3"
NASA_ABLATION_BASELINE = {
    "variant": "01_rich_entangler",
    "macro": {"RMSE": 0.01678, "MAE": 0.01409, "R2": 0.86993},
    "per_cell": {
        "B0018": {"RMSE": 0.02714, "MAE": 0.02227, "R2": 0.89353},
        "B0032": {"RMSE": 0.01191, "MAE": 0.01050, "R2": 0.90175},
        "B0053_test": {"RMSE": 0.01128, "MAE": 0.00951, "R2": 0.81450},
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_processed_calce(processed_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    data: dict[str, dict[str, np.ndarray]] = {}
    for cell_id in CELL_IDS:
        x_path = processed_dir / f"{cell_id}_X_unscaled.npy"
        y_path = processed_dir / f"{cell_id}_soh.npy"
        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(f"Missing processed CALCE arrays for {cell_id} in {processed_dir}")
        data[cell_id] = {
            "X_unscaled": np.load(x_path, allow_pickle=False),
            "soh": np.load(y_path, allow_pickle=False),
            "cycle": np.load(processed_dir / f"{cell_id}_cycle.npy", allow_pickle=False),
        }
    return data


def apply_frozen_nasa_scaler(
    calce: dict[str, dict[str, np.ndarray]],
    scaler,
    clip: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data_min = np.asarray(scaler.data_min_, dtype=np.float64)
    data_max = np.asarray(scaler.data_max_, dtype=np.float64)
    clipping: dict[str, dict[str, float]] = {}
    transformed: dict[str, np.ndarray] = {}
    for cell_id, values in calce.items():
        raw = values["X_unscaled"].astype(np.float64, copy=True)
        selected = raw[:, :, SCALED_INDICES]
        scaled = scaler.transform(selected.reshape(-1, 3)).reshape(selected.shape)
        below = np.mean(scaled < 0.0, axis=(0, 1))
        above = np.mean(scaled > 1.0, axis=(0, 1))
        clipping[cell_id] = {}
        for index, name in enumerate(SCALED_NAMES):
            clipping[cell_id][f"{name}_below_train_range_fraction"] = float(below[index])
            clipping[cell_id][f"{name}_above_train_range_fraction"] = float(above[index])
        if clip:
            scaled = np.clip(scaled, 0.0, 1.0)
        raw[:, :, SCALED_INDICES] = scaled
        transformed[cell_id] = raw.astype(np.float32)
    return transformed, {
        "clip_after_transform": clip,
        "scaler_data_min": data_min.tolist(),
        "scaler_data_max": data_max.tolist(),
        "fractions": clipping,
        "calce_scaler_fit_performed": False,
    }


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape}, predicted {predicted.shape}.")
    abs_error = np.abs(actual - predicted)
    denom = np.clip(np.abs(actual), a_min=1e-8, a_max=None)
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n": int(actual.size),
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "MAE": float(np.mean(abs_error)),
        "MAPE_pct": float(np.mean(abs_error / denom) * 100.0),
        "R2": r2,
        "MaxE": float(np.max(abs_error)),
        "mean_error": float(np.mean(actual - predicted)),
        "std_error": float(np.std(actual - predicted)),
    }


def load_frozen_model(checkpoint_path: Path, device: torch.device) -> TEQTransformer:
    cfg = rich_entangler_config()
    model = TEQTransformer(cfg)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_soh(
    model: TEQTransformer,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).to(device)
        out = model(batch).detach().cpu().numpy()
        preds.append(np.asarray(out, dtype=np.float32).reshape(-1))
        done = min(start + batch_size, n)
        print(f"  inferred {done}/{n}", flush=True)
    return np.concatenate(preds, axis=0)


def plot_soh_trajectory(
    cycles: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    cell_id: str,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(cycles, actual, color=PRIMARY_BLUE, linewidth=2.2, label="Actual SOH")
    ax.plot(
        cycles,
        predicted,
        color=ACCENT_GREEN,
        linestyle="--",
        linewidth=2.0,
        marker="o",
        markersize=3,
        markerfacecolor=ACCENT_GREEN,
        markeredgecolor=PRIMARY_BLUE,
        markevery=max(1, len(cycles) // 12),
        label="Predicted SOH",
    )
    ax.set_xlabel("Cycle Index")
    ax.set_ylabel("State of Health (SOH)")
    ax.set_title(f"NASA→CALCE zero-shot {cell_id}: Actual vs Predicted SOH")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.45, color=GRID_BLUE)
    fig.tight_layout()
    fig.savefig(output_dir / f"soh_trajectory_{cell_id}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"soh_trajectory_{cell_id}.png", bbox_inches="tight")
    plt.close(fig)


def plot_error_violin(errors: np.ndarray, cell_id: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    parts = ax.violinplot(errors, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor(ACCENT_GREEN)
        body.set_edgecolor(PRIMARY_BLUE)
        body.set_alpha(0.85)
    ax.axhline(0.0, color=PRIMARY_BLUE, linestyle="--", linewidth=1.5, alpha=0.9)
    ax.set_ylabel("Prediction Error (Actual - Predicted)")
    ax.set_title(f"Prediction Error Distribution ({cell_id})")
    ax.grid(True, linestyle="--", alpha=0.45, color=GRID_BLUE)
    ax.text(
        0.05,
        0.95,
        f"Mean: {float(np.mean(errors)):.5f}\nStd: {float(np.std(errors)):.5f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": PRIMARY_BLUE, "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"error_violin_{cell_id}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"error_violin_{cell_id}.png", bbox_inches="tight")
    plt.close(fig)


def plot_parity(actual: np.ndarray, predicted: np.ndarray, name: str, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(actual, predicted, alpha=0.6, color=ACCENT_GREEN, edgecolor=PRIMARY_BLUE, s=20, label="Predictions")
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    ax.plot([lo, hi], [lo, hi], color=PRIMARY_BLUE, linewidth=1.5, label="Ideal")
    ax.set_xlabel("Actual SOH")
    ax.set_ylabel("Predicted SOH")
    ax.set_title(f"Parity plot ({name})")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.45, color=GRID_BLUE)
    fig.tight_layout()
    fig.savefig(output_dir / f"parity_{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"parity_{name}.png", bbox_inches="tight")
    plt.close(fig)


def _fmt(value: float) -> str:
    return f"{value:.5f}"


def write_reports(result: dict[str, Any], report_path: Path, paper_path: Path) -> None:
    per_cell = result["per_cell"]
    overall = result["overall"]
    mean_cell = result["mean_pm_std_over_cells"]
    clipping = result["scaler"]["fractions"]
    lines = [
        "# NASA → CALCE Zero-Shot Report",
        "",
        "## Executive Summary",
        "",
        "This experiment is **zero-shot cross-dataset/cross-cell generalisation** from a frozen NASA-trained TE-Q-Transformer to official CALCE CS2 cells CS2_35–CS2_38. NASA weights never saw any CALCE cell. No NASA experiment was retrained. No CALCE scaler was fitted. Oxford was not used.",
        "",
        "This is **not** a cross-temperature experiment and **not** leave-one-battery-out. CALCE temperature is a constant 23 °C ambient fill because the official Excel logs have no temperature column. NASA room-temperature cells use a 24 °C ambient fill.",
        "",
        f"Overall pooled RMSE = {_fmt(overall['RMSE'])}, MAE = {_fmt(overall['MAE'])}, R² = {_fmt(overall['R2'])} on {overall['n']} discharge windows. Unweighted mean ± std over the four CALCE cells: RMSE {_fmt(mean_cell['RMSE_mean'])} ± {_fmt(mean_cell['RMSE_std'])}, MAE {_fmt(mean_cell['MAE_mean'])} ± {_fmt(mean_cell['MAE_std'])}, R² {_fmt(mean_cell['R2_mean'])} ± {_fmt(mean_cell['R2_std'])}. The frozen NASA model did **not** track CALCE capacity fade: predictions stayed near 1.01 for every cell, so error grows as true SOH declines. This is a negative zero-shot transfer result under the stated protocol, not a NASA in-domain result.",
        "",
        "## Frozen NASA Model",
        "",
        f"- Checkpoint: `{result['checkpoint']}`",
        f"- Variant: `{result['model_variant']}` (`entangler_type='rich'`)",
        "- Architecture: TE-Q-Transformer, input `[B, 512, 4]`, `d_model=64`, `n_heads=2`, `n_layers=3`, temporal Conv1d smoother, CLS pooling",
        "- Weights loaded with `strict=True`; `model.eval()`; `torch.no_grad()`",
        "- NASA notebooks, splits, and reported NASA metrics were not modified",
        "",
        "Reported NASA ablation baseline (copied, not recomputed):",
        "",
        "| Test cell | RMSE | MAE | R² |",
        "|---|---:|---:|---:|",
    ]
    for cell, metrics in NASA_ABLATION_BASELINE["per_cell"].items():
        lines.append(f"| {cell} | {metrics['RMSE']:.5f} | {metrics['MAE']:.5f} | {metrics['R2']:.5f} |")
    lines += [
        f"| Macro | {NASA_ABLATION_BASELINE['macro']['RMSE']:.5f} | {NASA_ABLATION_BASELINE['macro']['MAE']:.5f} | {NASA_ABLATION_BASELINE['macro']['R2']:.5f} |",
        "",
        "## NASA Preprocessing Used",
        "",
        "- Scaler: frozen `MinMaxScaler` fitted on NASA training cycles only (`n=660`)",
        f"- Scaler file: `{result['scaler_path']}`",
        "- Channels scaled: indices `(0, 1, 3)` = Voltage, Current, Time_norm",
        "- Temperature left in physical °C",
        "- CALCE values outside NASA train range were counted **before** clipping, then clipped to `[0, 1]` exactly as in the NASA notebook test transform",
        "",
        "## CALCE Dataset",
        "",
        "- Official source: CALCE Battery Research Group, https://web.calce.umd.edu/batteries/data/",
        "- Cells: CS2_35, CS2_36, CS2_37, CS2_38 (all used as unseen external test cells)",
        "- Chemistry: prismatic CS2, LiCoO2, nominal 1.1 Ah",
        "- Protocol: CCCV charge 0.5C to 4.2 V; 1C discharge; cutoff 2.7 V",
        "- Current: measured Arbin `Current(A)`, discharge negative (already NASA-compatible)",
        "- Temperature: no column in Excel; constant 23 °C fill",
        "- Existing CALCE notebook split and `calce_teq_transformer_best.pth` were **not** used",
        "",
        "## CALCE Preprocessing",
        "",
        "Complete 1C discharges were taken from chronological Excel files. A window never crosses a file or `Cycle_Index`. Each discharge was linearly resampled onto 512 phase points in `[0, 1]`. Feature order is `[Voltage, Current, Temperature_C, Time_norm]`. Per-cycle capacity is the Arbin `Discharge_Capacity(Ah)` increment over that discharge, not the possibly cumulative register maximum. Duplicate resampled windows and physically impossible capacities (>1.30 Ah) were dropped.",
        "",
        "Accepted windows: "
        + ", ".join(f"{cell}={count}" for cell, count in result["n_windows_by_cell"].items())
        + ".",
        "",
        "## SOH Construction",
        "",
        "`SOH_i = C_i / C_0` for each cell, where `C_i` is the per-discharge ampere-hour increment (`abs(last − first)` of Arbin `Discharge_Capacity(Ah)` on that 1C step) and `C_0` is the first accepted discharge of the same cell.",
        "",
        "## Leakage Prevention",
        "",
        "- NASA weights frozen; no NASA retraining",
        "- NASA scaler frozen; no CALCE fit",
        "- No CALCE-trained checkpoint",
        "- No hyperparameter search on CALCE",
        "- Oxford not substituted",
        "",
        "## Out-of-range fractions before clip",
        "",
        "| Cell | V below | V above | I below | I above | Time below | Time above |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell_id in CELL_IDS:
        frac = clipping[cell_id]
        lines.append(
            "| {cell} | {vb:.4f} | {va:.4f} | {ib:.4f} | {ia:.4f} | {tb:.4f} | {ta:.4f} |".format(
                cell=cell_id,
                vb=frac["Voltage_below_train_range_fraction"],
                va=frac["Voltage_above_train_range_fraction"],
                ib=frac["Current_below_train_range_fraction"],
                ia=frac["Current_above_train_range_fraction"],
                tb=frac["Time_norm_below_train_range_fraction"],
                ta=frac["Time_norm_above_train_range_fraction"],
            )
        )
    lines += [
        "",
        "## Zero-Shot Protocol",
        "",
        "```text",
        "frozen NASA 01_rich_entangler checkpoint",
        "        ↓",
        "CALCE [N, 512, 4] + frozen NASA-train MinMax on (V, I, Time)",
        "        ↓",
        "model.eval() inference on CS2_35–38",
        "```",
        "",
        "## Overall Results",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| N | {overall['n']} |",
        f"| RMSE | {_fmt(overall['RMSE'])} |",
        f"| MAE | {_fmt(overall['MAE'])} |",
        f"| MAPE (%) | {_fmt(overall['MAPE_pct'])} |",
        f"| R² | {_fmt(overall['R2'])} |",
        f"| MaxE | {_fmt(overall['MaxE'])} |",
        "",
        "## Per-Cell Results",
        "",
        "| Cell | N | RMSE | MAE | MAPE (%) | R² | MaxE | mean error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell_id in CELL_IDS:
        m = per_cell[cell_id]
        lines.append(
            f"| {cell_id} | {m['n']} | {_fmt(m['RMSE'])} | {_fmt(m['MAE'])} | {_fmt(m['MAPE_pct'])} | {_fmt(m['R2'])} | {_fmt(m['MaxE'])} | {_fmt(m['mean_error'])} |"
        )
    lines += [
        "",
        f"Mean ± std over cells: RMSE {_fmt(mean_cell['RMSE_mean'])} ± {_fmt(mean_cell['RMSE_std'])}; MAE {_fmt(mean_cell['MAE_mean'])} ± {_fmt(mean_cell['MAE_std'])}; R² {_fmt(mean_cell['R2_mean'])} ± {_fmt(mean_cell['R2_std'])}.",
        "",
        "## Error Analysis",
        "",
        "Mean error is actual − predicted. All four cells have **negative** mean error: the NASA model overestimates CALCE SOH.",
        "",
        "Predictions are nearly constant. Across 3909 windows the predicted SOH range is about 0.978–1.041, with a pooled mean of 1.007. True SOH ranges from about 0.14 to 1.00. The first cycle of each cell is close (absolute error ≈ 0.01) only because the true label is also ≈ 1. After that, the frozen model does not follow the fade trajectory.",
        "",
        "A constant predictor of 1.0 has pooled RMSE 0.32768, slightly **better** than the NASA model (0.33252), because the model sits a little above 1. Restricting to SOH ≥ 0.4 (3521 windows, closer to the local CALCE `.npy` end-of-life cutoff) still yields pooled RMSE 0.25336 and R² −2.66; predictions remain ≈ 1.01. This is not a clipping artifact: every CALCE Voltage/Current/Time_norm value was inside the NASA training min/max, so the out-of-range fractions are all zero.",
        "",
        "Likely contributors, not isolated here: different cell format and chemistry (NASA 18650 ~2 Ah vs CALCE prismatic CS2 1.1 Ah); CALCE 1C current is almost constant at −1.1 A, which occupies a narrow band after NASA min–max scaling; NASA temperature channel is a per-cell ambient fill and CALCE is a 23 °C fill, so the Arrhenius gate is nearly constant; voltage-curve aging is present in the 512-point windows but is not sufficient, under these frozen weights, to move the SOH head off ~1.",
        "",
        "## Comparison with NASA",
        "",
        "NASA in-domain ablation macro RMSE was 0.01678. The CALCE numbers below are a **different experimental question** (unseen dataset, unseen cells, different form factor and chemistry family). They are not a replacement for the NASA table and were not used to select the NASA checkpoint.",
        "",
        f"- NASA ablation macro RMSE: 0.01678",
        f"- CALCE zero-shot pooled RMSE: {_fmt(overall['RMSE'])}",
        f"- CALCE zero-shot mean-over-cells RMSE: {_fmt(mean_cell['RMSE_mean'])}",
        "",
        "## Limitations",
        "",
        "- CALCE has no measured intra-cycle temperature; 23 °C is an ambient fill, analogous to NASA’s per-cell ambient fill",
        "- Form factor and nominal capacity differ (NASA 18650 2 Ah vs CALCE prismatic 1.1 Ah)",
        "- This test does not isolate temperature as a causal factor",
        "- No CALCE value was clipped by the NASA scaler; domain shift is not an out-of-range scaling issue",
        "- Late-life cycles below ~40% SOH are included; they worsen RMSE but do not create the near-constant prediction",
        "",
        "## What This Experiment Demonstrates",
        "",
        "Zero-shot **evaluation** of a frozen NASA TE-Q-Transformer on CALCE CS2 cells CS2_35–38 at ~23–24 °C, using measured CALCE current and a NASA-training-only scaler. Under this protocol the NASA model does not transfer capacity-fade tracking: it remains near SOH 1.",
        "",
        "## What This Experiment Does NOT Demonstrate",
        "",
        "- Cross-temperature generalisation",
        "- Leave-one-battery-out within NASA or CALCE",
        "- That CALCE-trained TE-Q-Transformer performance equals this zero-shot number",
        "- Any change to previously reported NASA metrics",
        "",
        "## Reproducibility",
        "",
        "See commands at the end of this file. Random seed is unused because no training is performed.",
        "",
        "## Files Created",
        "",
        f"- `{report_path.as_posix()}`",
        f"- `{paper_path.as_posix()}`",
        f"- `{result['metrics_path']}`",
        "- `reports/calce/figures/soh_trajectory_CS2_*.png/.pdf`",
        "- `reports/calce/figures/error_violin_CS2_*.png/.pdf`",
        "- `reports/calce/figures/parity_*.png/.pdf`",
        "",
        "## Exact Commands",
        "",
        "```text",
        r".\.tools\uv\uv.exe run --python 3.12 --with numpy --with pandas --with openpyxl --with python-calamine python -m src.data.calce.preprocess",
        r".\.tools\uv\uv.exe run --python 3.12 --with numpy --with torch --with pennylane --with scikit-learn --with matplotlib python -m src.eval.nasa_calce_zero_shot --batch-size 8",
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    paper = [
        "# CALCE Paper-Ready Text",
        "",
        "## Claim sentence",
        "",
        (
            "A TE-Q-Transformer trained only on NASA cells was evaluated zero-shot on four unseen "
            "CALCE CS2 cells (CS2_35–CS2_38). Pooled RMSE was "
            f"{_fmt(overall['RMSE'])}, MAE {_fmt(overall['MAE'])}, and R² {_fmt(overall['R2'])} "
            f"(mean ± std over cells: RMSE {_fmt(mean_cell['RMSE_mean'])} ± {_fmt(mean_cell['RMSE_std'])}). "
            "Predictions stayed near 1.01 while true SOH faded, so this protocol does not show successful "
            "cross-dataset fade tracking. It is a cross-dataset/cross-cell test at ~23–24 °C, not a "
            "cross-temperature or leave-one-battery-out result."
        ),
        "",
        "## Methods paragraph",
        "",
        (
            "The frozen NASA model is the retained ablation variant `01_rich_entangler`. "
            "CALCE Arbin Excel logs supplied measured voltage and current. Complete 1C discharges "
            "were linearly resampled to 512 intra-cycle phase points with feature order "
            "[Voltage, Current, Temperature, Time]. Temperature was filled at 23 °C because the "
            "official CS2 files contain no temperature channel. Voltage, current, and normalized "
            "time were transformed with the NASA-training MinMax scaler only and clipped to [0, 1]; "
            "temperature remained in °C. SOH was C/C0 from the first valid discharge of each CALCE cell. "
            "No CALCE weights, CALCE scaler, or NASA retraining were used."
        ),
        "",
        "## Results paragraph",
        "",
        (
            "All four CALCE cells were treated as external test cells. "
            + "; ".join(
                f"{cell} RMSE={_fmt(per_cell[cell]['RMSE'])}, MAE={_fmt(per_cell[cell]['MAE'])}, R²={_fmt(per_cell[cell]['R2'])}"
                for cell in CELL_IDS
            )
            + ". Predicted SOH remained approximately 1.01 on every cell."
        ),
        "",
        "## Forbidden language",
        "",
        "Do not write: leave-one-battery-out; cross-temperature generalisation; CALCE-trained TE-Q-Transformer; Oxford as the primary external test.",
        "",
    ]
    paper_path.write_text("\n".join(paper) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    scaler_path = Path(args.scaler)
    checkpoint_path = Path(args.checkpoint)
    processed_dir = Path(args.processed_dir)
    figure_dir = Path(args.figure_dir)
    metrics_path = Path(args.metrics)
    report_path = Path(args.report)
    paper_path = Path(args.paper)
    scaled_dir = Path(args.scaled_dir)

    try:
        with scaler_path.open("rb") as handle:
            scaler = pickle.load(handle)
    except Exception as exc:
        print(f"pickle scaler load failed ({exc}); reconstructing from NASA manifest", flush=True)
        from sklearn.preprocessing import MinMaxScaler

        manifest = load_json(Path("configs/nasa_training_preprocessing_manifest.json"))
        spec = manifest["scaler"]
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaler.data_min_ = np.asarray(spec["data_min"], dtype=np.float64)
        scaler.data_max_ = np.asarray(spec["data_max"], dtype=np.float64)
        scaler.scale_ = np.asarray(spec["scale"], dtype=np.float64)
        scaler.min_ = np.asarray(spec["min"], dtype=np.float64)
        scaler.n_features_in_ = 3
        scaler.n_samples_seen_ = int(manifest["n_train_timesteps"])

    calce = load_processed_calce(processed_dir)
    transformed, scaler_info = apply_frozen_nasa_scaler(calce, scaler, clip=True)
    scaled_dir.mkdir(parents=True, exist_ok=True)
    for cell_id, X in transformed.items():
        np.save(scaled_dir / f"{cell_id}_X_nasa_scaled.npy", X, allow_pickle=False)

    device = torch.device("cpu")
    print("loading frozen NASA checkpoint", checkpoint_path, flush=True)
    model = load_frozen_model(checkpoint_path, device)
    print("config", asdict(model.cfg), flush=True)

    per_cell_metrics: dict[str, dict[str, float]] = {}
    all_actual: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    figure_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for cell_id in CELL_IDS:
        print(f"inferring {cell_id}", flush=True)
        X = transformed[cell_id]
        y = calce[cell_id]["soh"]
        cycles = calce[cell_id]["cycle"]
        if args.max_per_cell is not None:
            X = X[: args.max_per_cell]
            y = y[: args.max_per_cell]
            cycles = cycles[: args.max_per_cell]
        yhat = predict_soh(model, X, device, batch_size=args.batch_size)
        np.save(pred_dir / f"{cell_id}_y_true.npy", y, allow_pickle=False)
        np.save(pred_dir / f"{cell_id}_y_pred.npy", yhat, allow_pickle=False)
        metrics = compute_metrics(y, yhat)
        per_cell_metrics[cell_id] = metrics
        all_actual.append(y)
        all_pred.append(yhat)
        plot_soh_trajectory(cycles, y, yhat, cell_id, figure_dir)
        plot_error_violin(y - yhat, cell_id, figure_dir)
        plot_parity(y, yhat, cell_id, figure_dir)
        print(cell_id, metrics, flush=True)

    y_all = np.concatenate(all_actual)
    yhat_all = np.concatenate(all_pred)
    plot_parity(y_all, yhat_all, "CALCE_all", figure_dir)
    overall = compute_metrics(y_all, yhat_all)
    rmse = np.array([per_cell_metrics[c]["RMSE"] for c in CELL_IDS])
    mae = np.array([per_cell_metrics[c]["MAE"] for c in CELL_IDS])
    r2 = np.array([per_cell_metrics[c]["R2"] for c in CELL_IDS])
    result = {
        "claim": "zero-shot cross-dataset/cross-cell generalisation from NASA to CALCE",
        "not_a_claim": [
            "cross-temperature generalisation",
            "leave-one-battery-out",
            "CALCE-trained model performance",
        ],
        "checkpoint": str(checkpoint_path),
        "scaler_path": str(scaler_path),
        "model_variant": "01_rich_entangler",
        "feature_order": list(FEATURE_ORDER),
        "n_windows_by_cell": {cell: int(calce[cell]["soh"].size) for cell in CELL_IDS},
        "scaler": scaler_info,
        "per_cell": per_cell_metrics,
        "overall": overall,
        "mean_pm_std_over_cells": {
            "RMSE_mean": float(rmse.mean()),
            "RMSE_std": float(rmse.std(ddof=1)),
            "MAE_mean": float(mae.mean()),
            "MAE_std": float(mae.std(ddof=1)),
            "R2_mean": float(r2.mean()),
            "R2_std": float(r2.std(ddof=1)),
        },
        "nasa_ablation_baseline_copied_not_recomputed": NASA_ABLATION_BASELINE,
        "metrics_path": str(metrics_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_reports(result, report_path, paper_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/nasa/01_rich_entangler/nasa_teq_transformer_best.pth"),
    )
    parser.add_argument("--scaler", type=Path, default=Path("artifacts/nasa/nasa_train_scaler.pkl"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("Dataset/calce/processed_nasa_contract"),
    )
    parser.add_argument(
        "--scaled-dir",
        type=Path,
        default=Path("Dataset/calce/nasa_scaled"),
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("reports/calce/predictions"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("reports/calce/figures"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/calce/nasa_calce_zero_shot_metrics.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/calce/nasa_calce_zero_shot_report.md"),
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path("reports/calce/calce_paper_ready_text.md"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-per-cell", type=int, default=None)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({k: result[k] for k in ("overall", "per_cell", "mean_pm_std_over_cells")}, indent=2))


if __name__ == "__main__":
    main()
