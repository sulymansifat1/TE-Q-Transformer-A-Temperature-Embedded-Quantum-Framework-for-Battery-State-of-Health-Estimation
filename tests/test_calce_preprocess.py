"""Lightweight checks for CALCE NASA-contract preprocessing helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.data.calce.preprocess import _resample_discharge, _validate_discharge


class CalcePreprocessTests(unittest.TestCase):
    def test_resample_is_512_phase_and_preserves_endpoints(self) -> None:
        time_s = np.array([10.0, 20.0, 40.0, 50.0], dtype=np.float64)
        voltage = np.array([4.2, 3.9, 3.4, 2.7], dtype=np.float64)
        current = np.array([-1.1, -1.1, -1.09, -1.08], dtype=np.float64)
        window = _resample_discharge(time_s, voltage, current, 512)
        self.assertEqual(window.shape, (512, 4))
        np.testing.assert_allclose(window[0, 0], 4.2, atol=1e-6)
        np.testing.assert_allclose(window[-1, 0], 2.7, atol=1e-6)
        np.testing.assert_allclose(window[0, 3], 0.0)
        np.testing.assert_allclose(window[-1, 3], 1.0)
        np.testing.assert_allclose(window[:, 2], 23.0)

    def test_incomplete_discharge_is_rejected(self) -> None:
        time_s = np.linspace(0, 100, 30)
        voltage = np.linspace(4.1, 3.8, 30)
        current = np.full(30, -1.1)
        reasons = _validate_discharge(time_s, voltage, current, 0.4)
        self.assertTrue(any("cutoff" in item for item in reasons))


if __name__ == "__main__":
    unittest.main()
