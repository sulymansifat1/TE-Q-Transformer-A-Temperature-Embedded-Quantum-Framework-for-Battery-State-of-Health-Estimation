"""Centralized, portable path resolution for the TE-Q-Transformer experiment framework."""

from __future__ import annotations

import os
from pathlib import Path

# Project root is two levels up from Experiment/utils/
ROOT_DIR = Path(__file__).resolve().parents[2]

# Core project data directories (supports canonical datasets/ and legacy Dataset/)
DATASET_DIR = ROOT_DIR / "datasets" if (ROOT_DIR / "datasets").exists() else ROOT_DIR / "Dataset"
NASA_DATA_DIR = (
    ROOT_DIR / "datasets" / "NASA" / "processed"
    if (ROOT_DIR / "datasets" / "NASA" / "processed").exists()
    else DATASET_DIR / "nasa"
)
CALCE_DATA_DIR = (
    ROOT_DIR / "datasets" / "CALCE"
    if (ROOT_DIR / "datasets" / "CALCE").exists()
    else DATASET_DIR / "calce"
)
CALCE_RAW_DIR = CALCE_DATA_DIR / "raw"
CALCE_PROCESSED_DIR = (
    ROOT_DIR / "datasets" / "CALCE" / "processed"
    if (ROOT_DIR / "datasets" / "CALCE" / "processed").exists()
    else CALCE_DATA_DIR / "processed_nasa_contract"
)

# Preprocessing configs and model artifacts
CONFIGS_DIR = ROOT_DIR / "configs"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
NASA_FROZEN_SCALER_PATH = NASA_DATA_DIR / "nasa_train_scaler.pkl"
NASA_FROZEN_CHECKPOINT_PATH = (
    ARTIFACTS_DIR / "nasa" / "01_rich_entangler" / "nasa_teq_transformer_best.pth"
)
NASA_PREPROCESSING_MANIFEST_PATH = (
    CONFIGS_DIR / "nasa_training_preprocessing_manifest.json"
)

# Experiment framework directories
EXPERIMENT_ROOT = ROOT_DIR / "Experiment"
RESULT_DIR = EXPERIMENT_ROOT / "Result"
RESULT_NASA_DIR = RESULT_DIR / "NASA"
RESULT_CALCE_DIR = RESULT_DIR / "CALCE"
RESULT_TABLES_DIR = RESULT_DIR / "tables"
RESULT_FIGURES_DIR = RESULT_DIR / "figures"


def get_result_subdirs(dataset: str = "NASA") -> dict[str, Path]:
    """Returns and ensures existence of result subdirectories for a dataset."""
    base = RESULT_NASA_DIR if dataset.upper() == "NASA" else RESULT_CALCE_DIR
    subdirs = {
        "metrics": base / "metrics",
        "plots": base / "plots",
        "predictions": base / "predictions",
        "reports": base / "reports",
        "checkpoints": base / "checkpoints",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    RESULT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return subdirs
