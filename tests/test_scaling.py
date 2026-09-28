"""Unit tests for non-leakage feature scaling and temperature channel preservation."""

import unittest
import numpy as np
import torch

from src.data.nasa_loader import (
    fit_train_scaler,
    transform_with_scaler,
    FEATURE_IDX_TO_SCALE,
)


class TestScaling(unittest.TestCase):
    """Verifies that feature scaling obeys the non-leakage physical contract."""

    def test_temperature_is_unmodified(self):
        """Verifies that channel 2 (Temperature) is strictly preserved in Celsius."""
        N, L, C = 4, 512, 4
        # Create synthetic data: V in [2, 4], I in [-2, 0], Temp in [20, 30], Time in [0, 1]
        raw = np.zeros((N, L, C), dtype=np.float32)
        raw[:, :, 0] = np.random.uniform(2.5, 4.2, size=(N, L))
        raw[:, :, 1] = np.random.uniform(-3.0, 0.0, size=(N, L))
        raw[:, :, 2] = 24.5  # Fixed physical temperature in Celsius
        raw[:, :, 3] = np.linspace(0, 1, L).reshape(1, L)

        scaler = fit_train_scaler(raw)
        scaled_tensor = transform_with_scaler(raw, scaler, clip=True)

        # Features (0, 1, 3) must be scaled to [0, 1]
        self.assertTrue(torch.all(scaled_tensor[:, :, 0] >= 0.0) and torch.all(scaled_tensor[:, :, 0] <= 1.0))
        self.assertTrue(torch.all(scaled_tensor[:, :, 1] >= 0.0) and torch.all(scaled_tensor[:, :, 1] <= 1.0))
        self.assertTrue(torch.all(scaled_tensor[:, :, 3] >= 0.0) and torch.all(scaled_tensor[:, :, 3] <= 1.0))

        # Temperature channel (index 2) must remain exactly 24.5
        np.testing.assert_allclose(
            scaled_tensor[:, :, 2].numpy(),
            raw[:, :, 2],
            rtol=1e-5,
            err_msg="Temperature channel was modified by feature scaling!",
        )


if __name__ == "__main__":
    unittest.main()
