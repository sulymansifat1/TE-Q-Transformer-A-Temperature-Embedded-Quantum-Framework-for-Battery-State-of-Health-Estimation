"""Unit tests for TE-Q-Transformer and baseline models instantiation and forward pass."""

import unittest
import torch

from models.proposed import TEQTransformer, rich_entangler_config
from models.baselines import (
    CNN1DModel,
    TCNModel,
    DLinearSOHModel,
    TransformerModel,
    PatchTSTSOHModel,
    ITransformerSOHModel,
    QLSTMSOHModel,
    QNNGRUModel,
    LSTMModel,
    GRUModel,
    get_baseline_model,
    list_baselines,
)


class TestModels(unittest.TestCase):
    """Verifies that all models instantiate and perform forward inference."""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 512
        self.input_dim = 4
        # Channel 0: Voltage (scaled [0, 1])
        # Channel 1: Current (scaled [0, 1])
        # Channel 2: Temperature (Celsius, e.g. 24.0)
        # Channel 3: Normalized time ([0, 1])
        torch.manual_seed(42)
        self.dummy_x = torch.rand(self.batch_size, self.seq_len, self.input_dim)
        self.dummy_x[:, :, 2] = 24.0  # Physical temperature in Celsius

    def test_te_q_transformer_forward(self):
        """Verifies proposed TE-Q-Transformer forward pass."""
        model = TEQTransformer(rich_entangler_config())
        model.eval()
        with torch.no_grad():
            out = model(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size,))
        self.assertFalse(torch.isnan(out).any())

    def test_all_baselines_forward(self):
        """Verifies that all 10 active baselines complete forward pass [2, 512, 4] -> [2]."""
        baselines = {
            "LSTM": LSTMModel(),
            "GRU": GRUModel(),
            "CNN1D": CNN1DModel(),
            "TCN": TCNModel(),
            "DLinear": DLinearSOHModel(),
            "Transformer": TransformerModel(),
            "PatchTST": PatchTSTSOHModel(),
            "iTransformer": ITransformerSOHModel(),
            "QLSTM": QLSTMSOHModel(seq_len=8),  # test on shorter seq for fast unit test
            "QNN_GRU": QNNGRUModel(),
        }

        for name, model in baselines.items():
            with self.subTest(model_name=name):
                model.eval()
                with torch.no_grad():
                    if name == "QLSTM":
                        test_input = self.dummy_x[:, :8, :]
                    else:
                        test_input = self.dummy_x
                    out = model(test_input)
                self.assertEqual(out.shape, (self.batch_size,))
                self.assertFalse(torch.isnan(out).any(), f"NaN output detected in {name}")

    def test_baseline_factory(self):
        """Verifies get_baseline_model factory function."""
        for name in list_baselines():
            model = get_baseline_model(name)
            self.assertIsInstance(model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
