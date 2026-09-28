"""Unit tests for dataset presence, shapes, and loader integration."""

import unittest
from pathlib import Path
import numpy as np
import torch

from src.data.nasa_loader import (
    NASA_TRAIN_CELLS,
    NASA_TEST_CELLS,
    NASA_SPLIT_CELL,
    load_cell_arrays,
    load_full_cell,
    split_cell_70_30,
)
from src.data.calce_loader import (
    CALCE_CELL_IDS,
    load_calce_cell,
)


class TestDatasets(unittest.TestCase):
    """Verifies dataset files and shapes under datasets/."""

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.nasa_dir = self.repo_root / "datasets" / "NASA" / "processed"
        self.calce_dir = self.repo_root / "datasets" / "CALCE" / "processed"

    def test_nasa_files_exist(self):
        """Verifies that all processed NASA files exist."""
        all_cells = list(NASA_TRAIN_CELLS) + list(NASA_TEST_CELLS) + [NASA_SPLIT_CELL]
        for cell_id in all_cells:
            x_file = self.nasa_dir / f"{cell_id}_X.npy"
            y_file = self.nasa_dir / f"{cell_id}_soh.npy"
            self.assertTrue(x_file.exists(), f"Missing {x_file}")
            self.assertTrue(y_file.exists(), f"Missing {y_file}")

    def test_nasa_tensor_shapes(self):
        """Verifies that NASA arrays conform to [N, 512, 4] and [N]."""
        for cell_id in ["B0005", "B0018"]:
            X, y = load_cell_arrays(self.nasa_dir, cell_id)
            self.assertEqual(X.ndim, 3)
            self.assertEqual(X.shape[1:], (512, 4))
            self.assertEqual(y.ndim, 1)
            self.assertEqual(len(X), len(y))

    def test_calce_files_exist(self):
        """Verifies that all processed CALCE arrays exist."""
        for cell_id in CALCE_CELL_IDS:
            x_file = self.calce_dir / f"{cell_id}_X_unscaled.npy"
            y_file = self.calce_dir / f"{cell_id}_soh.npy"
            self.assertTrue(x_file.exists(), f"Missing {x_file}")
            self.assertTrue(y_file.exists(), f"Missing {y_file}")

    def test_calce_tensor_shapes(self):
        """Verifies CALCE shapes match the NASA contract."""
        for cell_id in CALCE_CELL_IDS:
            X, y, cycle = load_calce_cell(self.calce_dir, cell_id)
            self.assertEqual(X.ndim, 3)
            self.assertEqual(X.shape[1:], (512, 4))
            self.assertEqual(y.ndim, 1)
            self.assertEqual(len(X), len(y))


if __name__ == "__main__":
    unittest.main()
