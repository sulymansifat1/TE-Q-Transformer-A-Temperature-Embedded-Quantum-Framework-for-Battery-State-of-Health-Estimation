"""Experiment E02: Raw vs. Physics-Guided Temperature Representation.

Scientific Question:
Does physics-guided temperature encoding improve SOH estimation compared with conventional
temperature processing?

Condition E02-A: Conventional / Raw Temperature (MinMax angle mapping on Qubit 3, no Arrhenius gate)
Condition E02-B: Physics-Guided Temperature (Arrhenius thermodynamic gate on Qubit 3)
All other architecture, quantum circuit, and training components are strictly identical.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pennylane as qml
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Experiment.utils.paths import (
    NASA_DATA_DIR,
    NASA_FROZEN_CHECKPOINT_PATH,
    ROOT_DIR,
)
from Experiment.utils.seed import seed_everything
from Experiment.utils.metrics import compute_metrics
from Experiment.Dataset.nasa import get_nasa_dataloaders
from Experiment.Proposed_Model.model import (
    TEQTransformer,
    TEQTransformerConfig,
    rich_entangler_config,
    PositionalEncoding,
)

E02_OUTPUT_DIR = ROOT_DIR / "GarbageResults" / "AblationE02"


class RawTemperatureQuantumEmbedding(nn.Module):
    """Conventional / Raw Temperature Quantum Embedding Layer.
    
    Identical 4-qubit circuit with Rich Entangler as QuantumEmbeddingLayer,
    but maps temperature linearly to [0, pi] without the Arrhenius equation.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        q_device: str = "default.qubit",
        entangler_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.entangler_weights = nn.Parameter(
            0.01 * torch.randn(entangler_layers, n_qubits, dtype=torch.float32)
        )
        self.entangler_rzz = nn.Parameter(
            0.01 * torch.randn(entangler_layers, max(1, n_qubits - 1), dtype=torch.float32)
        )
        dev = qml.device(q_device, wires=n_qubits)

        def _apply_rich_entangler(weights: torch.Tensor, rzz: torch.Tensor) -> None:
            for layer in range(weights.shape[0]):
                for q in range(n_qubits):
                    qml.RZ(weights[layer, q], wires=q)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                for q in range(n_qubits - 1):
                    qml.IsingZZ(rzz[layer, q], wires=[q, q + 1])
                if n_qubits >= 3:
                    qml.Toffoli(wires=[0, 1, 2])
                    if n_qubits >= 4:
                        qml.Toffoli(wires=[1, 2, 3])
                for q in range(n_qubits):
                    qml.Hadamard(wires=q)
                    qml.RZ(weights[layer, q], wires=q)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(
            inputs: torch.Tensor,
            entangler_weights: torch.Tensor,
            entangler_rzz: torch.Tensor,
        ):
            qml.RY(inputs[:, 0], wires=0)
            qml.RY(inputs[:, 1], wires=1)
            qml.RY(inputs[:, 2], wires=2)
            qml.RY(inputs[:, 3], wires=3)
            _apply_rich_entangler(entangler_weights, entangler_rzz)
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError(f"Expected [B_flat, 4], got {tuple(x.shape)}")

        out_device = x.device
        out_dtype = x.dtype

        # Wire 0: Voltage, Wire 1: Current, Wire 2: Time_norm
        voltage_angle = x[:, 0] * torch.pi
        current_angle = x[:, 1] * torch.pi
        time_angle = x[:, 3] * torch.pi

        # Wire 3: Conventional Raw Temperature MinMax linear angle mapping
        # Maps 4°C -> 0.0, 43°C -> 1.0 linearly without Arrhenius physics
        temp_c = x[:, 2]
        temp_norm = torch.clamp((temp_c - 4.0) / 39.0, 0.0, 1.0)
        theta_temp = temp_norm * torch.pi

        angles = torch.stack([voltage_angle, current_angle, time_angle, theta_temp], dim=1)

        angles_cpu = angles.to("cpu")
        entangler_cpu = self.entangler_weights.to("cpu")
        entangler_rzz_cpu = self.entangler_rzz.to("cpu")
        q_out = self.circuit(angles_cpu, entangler_cpu, entangler_rzz_cpu)
        q_tensor = torch.stack(q_out, dim=1).to(device=out_device, dtype=out_dtype)
        return q_tensor


class RawTemperatureTEQTransformer(nn.Module):
    """TE-Q-Transformer with conventional raw temperature encoding.
    
    Architecture is strictly identical to TEQTransformer in all layers
    except for using RawTemperatureQuantumEmbedding instead of QuantumEmbeddingLayer.
    """

    def __init__(self, d_model: int = 64, seq_len: int = 512) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.quantum_embed = RawTemperatureQuantumEmbedding(n_qubits=4)
        self.quantum_proj = nn.Linear(4, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 2)

        self.temporal_smooth = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=64,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(batch_size * seq_len, 4)
        q_features = self.quantum_embed(x_flat)
        q_features = self.quantum_proj(q_features)
        q_sequence = q_features.reshape(batch_size, seq_len, self.d_model)

        q_sequence = q_sequence.transpose(1, 2)
        q_sequence = self.temporal_smooth(q_sequence)
        q_sequence = q_sequence.transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        q_sequence = torch.cat([cls_tokens, q_sequence], dim=1)
        q_sequence = self.pos_encoder(q_sequence)

        transformed = self.transformer(q_sequence)
        pooled = transformed[:, 0]
        soh = self.head(pooled)
        return soh.squeeze(-1)


def evaluate_model_on_nasa_loaders(
    model: nn.Module,
    test_loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Evaluates model across NASA test cells and returns metrics and predictions."""
    model.eval()
    results = {}
    preds_dict = {}
    trues_dict = {}

    with torch.no_grad():
        for cell_id, loader in test_loaders.items():
            y_true_list, y_pred_list = [], []
            for bx, by in loader:
                bx = bx.to(device)
                out = model(bx)
                y_true_list.append(by.cpu().numpy().reshape(-1))
                y_pred_list.append(out.cpu().numpy().reshape(-1))

            y_true = np.concatenate(y_true_list)
            y_pred = np.concatenate(y_pred_list)
            results[cell_id] = compute_metrics(y_true, y_pred)
            trues_dict[cell_id] = y_true
            preds_dict[cell_id] = y_pred

    macro = {
        k: float(np.mean([results[c][k] for c in test_loaders]))
        for k in ["RMSE", "MAE", "MAPE (%)", "R2", "MaxE"]
    }
    results["macro"] = macro
    return results, preds_dict, trues_dict


def run_experiment(
    epochs: int = 80,
    batch_size: int = 8,
    seed: int = 42,
    patience: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-2,
) -> Dict[str, Any]:
    """Runs the controlled E02 ablation experiment (E02-A vs E02-B)."""
    print("\n==================================================")
    print("EXPERIMENT E02: RAW VS. PHYSICS-GUIDED TEMPERATURE")
    print("==================================================")

    seed_everything(seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device} (torch threads: {torch.get_num_threads()})")

    # Ensure output directories exist
    base_out = E02_OUTPUT_DIR
    metrics_dir = base_out / "metrics"
    preds_raw_dir = base_out / "predictions" / "raw_temperature"
    preds_phys_dir = base_out / "predictions" / "physics_temperature"
    ckpts_raw_dir = base_out / "checkpoints" / "raw_temperature"
    ckpts_phys_dir = base_out / "checkpoints" / "physics_temperature"
    logs_raw_dir = base_out / "logs" / "raw_temperature"
    logs_phys_dir = base_out / "logs" / "physics_temperature"
    configs_dir = base_out / "configs"

    for d in [metrics_dir, preds_raw_dir, preds_phys_dir, ckpts_raw_dir, ckpts_phys_dir, logs_raw_dir, logs_phys_dir, configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_loader, test_loaders, _ = get_nasa_dataloaders(batch_size=batch_size)

    # ----------------------------------------------------
    # CONDITION E02-B: Physics-Guided Temperature
    # ----------------------------------------------------
    print("\n--- [Condition E02-B] Physics-Guided Temperature ---")
    phys_model = TEQTransformer(rich_entangler_config()).to(device)
    if NASA_FROZEN_CHECKPOINT_PATH.exists():
        print(f"Loading canonical physics-guided checkpoint: {NASA_FROZEN_CHECKPOINT_PATH}")
        phys_model.load_state_dict(torch.load(NASA_FROZEN_CHECKPOINT_PATH, map_location=device))
        # Copy to E02 directory
        torch.save(phys_model.state_dict(), ckpts_phys_dir / "best_model.pth")
    else:
        raise FileNotFoundError(f"Frozen checkpoint not found at {NASA_FROZEN_CHECKPOINT_PATH}")

    phys_results, phys_preds, phys_trues = evaluate_model_on_nasa_loaders(phys_model, test_loaders, device)

    # Save physics predictions
    for cell_id, pred in phys_preds.items():
        np.save(preds_phys_dir / f"{cell_id}_pred.npy", pred)
        np.save(preds_phys_dir / f"{cell_id}_true.npy", phys_trues[cell_id])

    # Save physics config
    phys_cfg_dict = {
        "condition": "E02-B_physics_guided_temperature",
        "temperature_encoding": "Arrhenius thermodynamic gate (SEI + Plating)",
        "entangler_type": "rich",
        "d_model": 64,
        "n_layers": 3,
        "n_heads": 2,
        "parameters": sum(p.numel() for p in phys_model.parameters()),
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
    }
    (configs_dir / "physics_temperature_config.json").write_text(json.dumps(phys_cfg_dict, indent=2), encoding="utf-8")

    # Save physics metrics CSV
    phys_rows = []
    for cell in ["B0018", "B0032", "B0053_test"]:
        m = phys_results[cell]
        phys_rows.append({"condition": "physics_guided", "cell": cell, **m})
    phys_rows.append({"condition": "physics_guided", "cell": "MACRO_AVG", **phys_results["macro"]})

    with (metrics_dir / "e02_physics_temperature_metrics.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(phys_rows[0].keys()))
        w.writeheader()
        w.writerows(phys_rows)

    print("Physics-Guided Results:")
    for row in phys_rows:
        print(f"  {row['cell']:12s} | RMSE: {row['RMSE']:.5f} | MAE: {row['MAE']:.5f} | R2: {row['R2']:.5f}")

    # ----------------------------------------------------
    # CONDITION E02-A: Conventional / Raw Temperature
    # ----------------------------------------------------
    print("\n--- [Condition E02-A] Conventional / Raw Temperature ---")
    seed_everything(seed)
    raw_model = RawTemperatureTEQTransformer(d_model=64).to(device)
    raw_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Model parameters: {raw_params} (vs physics: {phys_cfg_dict['parameters']})")

    # Save raw config
    raw_cfg_dict = {
        "condition": "E02-A_conventional_raw_temperature",
        "temperature_encoding": "Conventional MinMax linear angle mapping [0, pi] on Qubit 3",
        "entangler_type": "rich",
        "d_model": 64,
        "n_layers": 3,
        "n_heads": 2,
        "parameters": raw_params,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }
    (configs_dir / "raw_temperature_config.json").write_text(json.dumps(raw_cfg_dict, indent=2), encoding="utf-8")

    raw_optimizer = optim.AdamW(raw_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raw_scheduler = ReduceLROnPlateau(raw_optimizer, mode="min", factor=0.5, patience=10)
    raw_criterion = nn.MSELoss()

    best_loss = float("inf")
    patience_cnt = 0
    raw_ckpt_path = ckpts_raw_dir / "best_model.pth"
    raw_log_lines = []

    print(f"Training Raw-Temperature TE-Q-Transformer (Max epochs: {epochs}, Patience: {patience})...")
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        raw_model.train()
        ep_losses = []

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            raw_optimizer.zero_grad()
            out = raw_model(bx)
            loss = raw_criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            raw_optimizer.step()
            ep_losses.append(loss.item())

        mean_ep_loss = float(np.mean(ep_losses))
        ep_elapsed = time.perf_counter() - ep_t0
        cur_lr = float(raw_optimizer.param_groups[0]["lr"])
        raw_scheduler.step(mean_ep_loss)

        log_str = f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {mean_ep_loss:.6f} | LR: {cur_lr:.2e} | Time: {ep_elapsed:.2f}s"
        raw_log_lines.append(log_str)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {log_str}")

        if mean_ep_loss < best_loss:
            best_loss = mean_ep_loss
            patience_cnt = 0
            torch.save(raw_model.state_dict(), raw_ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.6f}")
                raw_log_lines.append(f"Early stopping at epoch {epoch}")
                break

    train_total_time = time.perf_counter() - train_start
    print(f"Training completed in {train_total_time:.1f}s.")

    # Save log
    (logs_raw_dir / "train_and_eval_log.txt").write_text("\n".join(raw_log_lines), encoding="utf-8")

    # Load best checkpoint for evaluation
    if raw_ckpt_path.exists():
        raw_model.load_state_dict(torch.load(raw_ckpt_path, map_location=device))
    raw_results, raw_preds, raw_trues = evaluate_model_on_nasa_loaders(raw_model, test_loaders, device)

    # Save raw predictions
    for cell_id, pred in raw_preds.items():
        np.save(preds_raw_dir / f"{cell_id}_pred.npy", pred)
        np.save(preds_raw_dir / f"{cell_id}_true.npy", raw_trues[cell_id])

    # Save raw metrics CSV
    raw_rows = []
    for cell in ["B0018", "B0032", "B0053_test"]:
        m = raw_results[cell]
        raw_rows.append({"condition": "raw_temperature", "cell": cell, **m})
    raw_rows.append({"condition": "raw_temperature", "cell": "MACRO_AVG", **raw_results["macro"]})

    with (metrics_dir / "e02_raw_temperature_metrics.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)

    print("\nRaw-Temperature Results:")
    for row in raw_rows:
        print(f"  {row['cell']:12s} | RMSE: {row['RMSE']:.5f} | MAE: {row['MAE']:.5f} | R2: {row['R2']:.5f}")

    # ----------------------------------------------------
    # COMPARISON TABLE
    # ----------------------------------------------------
    comp_rows = []
    for cell in ["B0018", "B0032", "B0053_test", "MACRO_AVG"]:
        phys_m = phys_results[cell if cell != "MACRO_AVG" else "macro"]
        raw_m = raw_results[cell if cell != "MACRO_AVG" else "macro"]
        comp_rows.append({
            "cell": cell,
            "Raw_RMSE": raw_m["RMSE"],
            "Physics_RMSE": phys_m["RMSE"],
            "RMSE_Diff (Physics - Raw)": phys_m["RMSE"] - raw_m["RMSE"],
            "RMSE_Improvement_%": ((raw_m["RMSE"] - phys_m["RMSE"]) / raw_m["RMSE"]) * 100.0,
            "Raw_MAE": raw_m["MAE"],
            "Physics_MAE": phys_m["MAE"],
            "MAE_Diff (Physics - Raw)": phys_m["MAE"] - raw_m["MAE"],
            "MAE_Improvement_%": ((raw_m["MAE"] - phys_m["MAE"]) / raw_m["MAE"]) * 100.0,
            "Raw_R2": raw_m["R2"],
            "Physics_R2": phys_m["R2"],
            "R2_Gain": phys_m["R2"] - raw_m["R2"],
        })

    with (metrics_dir / "e02_comparison.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(comp_rows[0].keys()))
        w.writeheader()
        w.writerows(comp_rows)

    print("\n==================================================")
    print("E02 COMPARISON SUMMARY (Physics vs Raw Temperature):")
    print("==================================================")
    for r in comp_rows:
        print(f"  {r['cell']:12s} | Raw RMSE: {r['Raw_RMSE']:.5f} | Phys RMSE: {r['Physics_RMSE']:.5f} | Improvement: {r['RMSE_Improvement_%']:+.2f}% | R2 Gain: {r['R2_Gain']:+.4f}")

    return {
        "physics": phys_results,
        "raw": raw_results,
        "comparison": comp_rows,
        "raw_training_time_seconds": train_total_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment E02: Raw vs Physics-Guided Temperature.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiment(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
