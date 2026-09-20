"""Publication-quality plotting utilities for battery SOH trajectories and errors."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PRIMARY_BLUE = "#6F90AE"
ACCENT_GREEN = "#AFC8A7"
GRID_BLUE = "#C9D6E3"
ERROR_ORANGE = "#E08963"


def plot_soh_trajectory(
    cycles: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    test_cell: str,
    output_dir: Path,
    title_suffix: str = "",
) -> Path:
    """Plots actual vs predicted SOH degradation trajectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=300)
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
    ax.set_xlabel("Cycle Index", fontsize=11)
    ax.set_ylabel("State of Health (SOH)", fontsize=11)
    title = f"Cell {test_cell}: Actual vs Predicted SOH"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6, color=GRID_BLUE)
    plt.tight_layout()

    out_path = output_dir / f"soh_trajectory_{test_cell}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_parity(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    output_path: Path,
) -> Path:
    """Parity plot (Predicted vs Actual with perfect prediction diagonal line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=300)
    ax.scatter(actual, predicted, alpha=0.5, color=PRIMARY_BLUE, edgecolors="none", s=25)
    lims = [
        min(float(np.min(actual)), float(np.min(predicted))),
        max(float(np.max(actual)), float(np.max(predicted))),
    ]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.5, label="Perfect Line (y=x)")
    ax.set_xlabel("Ground Truth SOH", fontsize=11)
    ax.set_ylabel("Predicted SOH", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6, color=GRID_BLUE)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_loss_curves(
    train_loss: list[float],
    val_loss: list[float] | None,
    output_path: Path,
    title: str = "Training Convergence",
) -> Path:
    """Plots training and validation loss curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=300)
    ax.plot(train_loss, label="Train Loss", color=PRIMARY_BLUE, linewidth=2.0)
    if val_loss and len(val_loss) == len(train_loss):
        ax.plot(val_loss, label="Val Loss", color=ERROR_ORANGE, linestyle="--", linewidth=2.0)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6, color=GRID_BLUE)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
