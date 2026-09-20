"""Experiment E04: Matched Classical vs Quantum Representation under Physics-Guided Temperature Encoding.

Scientific Question:
Does the quantum representation provide additional predictive benefit beyond a comparable
classical representation when both use the same physics-guided temperature formulation?

Condition E04-A: Classical + Physics-guided temperature (2-layer MLP [4 -> 16 -> 4, Tanh])
Condition E04-B: Quantum + Physics-guided temperature (4-qubit VQC, Rich Entangler, PauliZ)

Strict single-variable isolation: Both models use identical combined Arrhenius degradation
kinetics (SEI + Plating with learnable Ea), identical 4-dimensional representation output,
identical Linear(4, 64) latent projection, identical Conv1D temporal smoothing, identical
3-layer Transformer encoder, and identical training/evaluation protocols.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Literal

import numpy as np
import pennylane as qml
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Experiment.utils.paths import (
    NASA_DATA_DIR,
    ROOT_DIR,
)
from Experiment.utils.seed import seed_everything
from Experiment.utils.metrics import compute_metrics
from Experiment.Dataset.nasa import get_nasa_dataloaders
from Experiment.Proposed_Model.model import PositionalEncoding

E04_OUTPUT_DIR = ROOT_DIR / "GarbageResults" / "AblationE04"


class ClassicalPhysicsEmbeddingLayer(nn.Module):
    """Capacity-matched Classical Non-linear Embedding with Combined Arrhenius Physics Gate."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.R = 8.314462618
        self.T_ref = 298.15

        # Learnable activation energies initialized to 3.0 (30 kJ/mol)
        self.Ea_sei = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        self.Ea_pl = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))

        # 2-layer non-linear MLP mapping R^4 -> R^hidden -> R^4 bounded in [-1, 1]
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Tanh(),
        )

    def compute_theta_temp(self, temp_c: torch.Tensor) -> torch.Tensor:
        temp_k = torch.clamp(temp_c + 273.15, min=1.0)
        inv_t = 1.0 / temp_k
        inv_t_ref = 1.0 / self.T_ref

        Ea_sei_actual = self.Ea_sei * 10000.0
        Ea_pl_actual = self.Ea_pl * 10000.0
        sei_term = torch.exp((Ea_sei_actual / self.R) * (inv_t_ref - inv_t))
        plating_term = torch.exp((Ea_pl_actual / self.R) * (inv_t - inv_t_ref))
        phi = sei_term + plating_term
        theta_temp = torch.pi * phi / 4.0
        return theta_temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_flat, 4] where channels are [V, I, T, time_norm]
        voltage_angle = x[:, 0] * torch.pi
        current_angle = x[:, 1] * torch.pi
        time_angle = x[:, 3] * torch.pi
        theta_temp = self.compute_theta_temp(x[:, 2])

        angles = torch.stack([voltage_angle, current_angle, time_angle, theta_temp], dim=1)
        c_features = self.mlp(angles)  # Output in [-1, 1]
        return c_features


class QuantumPhysicsEmbeddingLayer(nn.Module):
    """4-Qubit Variational Quantum Circuit Embedding with Combined Arrhenius Physics Gate."""

    def __init__(
        self,
        n_qubits: int = 4,
        q_device: str = "default.qubit",
        entangler_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.R = 8.314462618
        self.T_ref = 298.15

        # Learnable activation energies initialized to 3.0 (30 kJ/mol)
        self.Ea_sei = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        self.Ea_pl = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))

        # Rich Entangler weights
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

    def compute_theta_temp(self, temp_c: torch.Tensor) -> torch.Tensor:
        temp_k = torch.clamp(temp_c + 273.15, min=1.0)
        inv_t = 1.0 / temp_k
        inv_t_ref = 1.0 / self.T_ref

        Ea_sei_actual = self.Ea_sei * 10000.0
        Ea_pl_actual = self.Ea_pl * 10000.0
        sei_term = torch.exp((Ea_sei_actual / self.R) * (inv_t_ref - inv_t))
        plating_term = torch.exp((Ea_pl_actual / self.R) * (inv_t - inv_t_ref))
        phi = sei_term + plating_term
        theta_temp = torch.pi * phi / 4.0
        return theta_temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_device = x.device
        out_dtype = x.dtype

        voltage_angle = x[:, 0] * torch.pi
        current_angle = x[:, 1] * torch.pi
        time_angle = x[:, 3] * torch.pi
        theta_temp = self.compute_theta_temp(x[:, 2])

        angles = torch.stack([voltage_angle, current_angle, time_angle, theta_temp], dim=1)

        angles_cpu = angles.to("cpu")
        entangler_cpu = self.entangler_weights.to("cpu")
        entangler_rzz_cpu = self.entangler_rzz.to("cpu")
        q_out = self.circuit(angles_cpu, entangler_cpu, entangler_rzz_cpu)
        q_tensor = torch.stack(q_out, dim=1).to(device=out_device, dtype=out_dtype)
        return q_tensor


class AblationE04Model(nn.Module):
    """Complete Sequence Model with interchangeable Classical vs Quantum embedding."""

    def __init__(
        self,
        mode: Literal["classical_physics", "quantum_physics"] = "quantum_physics",
        seq_len: int = 512,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 3,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        temporal_kernel_size: int = 3,
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.seq_len = seq_len
        self.d_model = d_model

        if mode == "classical_physics":
            self.embed = ClassicalPhysicsEmbeddingLayer(hidden_dim=16)
        elif mode == "quantum_physics":
            self.embed = QuantumPhysicsEmbeddingLayer(n_qubits=4, q_device="default.qubit", entangler_layers=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.latent_proj = nn.Linear(4, d_model)

        self.temporal_smooth = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
            bias=False,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len + 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 4:
            raise ValueError(f"Expected [B, L, 4], got {tuple(x.shape)}")
        batch_size, seq_len, _ = x.shape

        x_flat = x.reshape(batch_size * seq_len, 4)
        features = self.embed(x_flat)
        features = self.latent_proj(features)
        sequence = features.reshape(batch_size, seq_len, self.d_model)

        sequence = sequence.transpose(1, 2)
        sequence = self.temporal_smooth(sequence)
        sequence = sequence.transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, sequence], dim=1)
        sequence = self.pos_encoder(sequence)

        transformed = self.transformer(sequence)
        pooled = transformed[:, 0]
        soh = self.head(pooled)
        return soh.squeeze(-1)


def evaluate_model_on_nasa_loaders(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Evaluates the model across test cells and computes standard NASA SOH metrics."""
    model.eval()
    test_metrics = {}
    test_predictions = {}

    for name, loader in loaders.items():
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                pred = model(batch_x).cpu().numpy()
                target = batch_y.numpy()
                all_preds.append(pred)
                all_targets.append(target)

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        metrics = compute_metrics(y_true, y_pred)
        test_metrics[name] = metrics
        test_predictions[name] = (y_true, y_pred)

    # Compute Macro Average
    macro = {}
    for metric_key in ["RMSE", "MAE", "MAPE (%)", "R2", "MaxE"]:
        macro[metric_key] = float(np.mean([test_metrics[k][metric_key] for k in test_metrics]))
    test_metrics["MACRO_AVG"] = macro

    return test_metrics, test_predictions


def run_condition(
    mode: Literal["classical_physics", "quantum_physics"],
    epochs: int = 80,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
    patience: int = 20,
    seed: int = 42,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Trains and evaluates one E04 condition."""
    print(f"\n{'='*50}")
    print(f"CONDITION E04: Mode = {mode.upper()}")
    print(f"{'='*50}")

    seed_everything(seed)
    device = torch.device(device_str)

    ckpt_dir = E04_OUTPUT_DIR / "checkpoints" / mode
    log_dir = E04_OUTPUT_DIR / "logs" / mode
    pred_dir = E04_OUTPUT_DIR / "predictions" / mode
    cfg_dir = E04_OUTPUT_DIR / "configs" / mode
    metrics_dir = E04_OUTPUT_DIR / "metrics"

    for d in [ckpt_dir, log_dir, pred_dir, cfg_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_dict = {
        "mode": mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "seed": seed,
        "device": device_str,
    }
    with open(cfg_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Data loaders
    train_loader, test_loaders, scaler = get_nasa_dataloaders(
        data_dir=NASA_DATA_DIR,
        batch_size=batch_size,
    )

    model = AblationE04Model(mode=mode).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count}")
    print(f"Initial Ea_sei: {model.embed.Ea_sei.item():.4f} | Initial Ea_pl: {model.embed.Ea_pl.item():.4f}")

    log_file = log_dir / "train_and_eval_log.txt"
    ckpt_path = ckpt_dir / "best_model.pth"

    # Check if complete training record already exists
    already_trained = False
    if log_file.exists() and ckpt_path.exists():
        with open(log_file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip().startswith("Epoch")]
            if len(lines) == epochs:
                print(f"Verified complete training record for {mode} ({epochs} epochs logged). Proceeding directly to evaluation...")
                already_trained = True

    train_start = time.time()
    if not already_trained:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        best_epoch = -1
        patience_counter = 0

        with open(log_file, "w") as lf:
            lf.write(f"=== E04 Training Log: {mode} (Seed {seed}) ===\n")
            lf.write(f"Total parameters: {param_count}\n\n")

            print(f"Training {mode} TE-Q-Transformer (Max epochs: {epochs}, Patience: {patience})...")
            for epoch in range(1, epochs + 1):
                ep_start = time.time()
                model.train()
                running_loss = 0.0
                total_batches = 0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    out = model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    running_loss += loss.item()
                    total_batches += 1

                epoch_loss = running_loss / max(1, total_batches)
                scheduler.step(epoch_loss)
                ep_time = time.time() - ep_start
                curr_lr = optimizer.param_groups[0]["lr"]

                ea_sei_val = f"{model.embed.Ea_sei.item():.4f}"
                ea_pl_val = f"{model.embed.Ea_pl.item():.4f}"

                log_str = (
                    f"Epoch {epoch:02d}/{epochs} | Train Loss: {epoch_loss:.6f} | "
                    f"LR: {curr_lr:.2e} | Ea_sei: {ea_sei_val} | Ea_pl: {ea_pl_val} | Time: {ep_time:.2f}s"
                )
                lf.write(log_str + "\n")
                lf.flush()

                if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                    print("  " + log_str)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    patience_counter += 1

        total_train_time = time.time() - train_start
        print(f"Training completed in {total_train_time:.1f}s. Best epoch: {best_epoch} (Loss: {best_loss:.6f})")
    else:
        total_train_time = 0.0

    # Load best checkpoint for evaluation
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    eval_metrics, eval_predictions = evaluate_model_on_nasa_loaders(model, test_loaders, device)

    # Save predictions as npy arrays
    for cell_name, (y_true, y_pred) in eval_predictions.items():
        np.save(pred_dir / f"{cell_name}_true.npy", y_true)
        np.save(pred_dir / f"{cell_name}_pred.npy", y_pred)

    # Save condition metrics to CSV
    metrics_csv = metrics_dir / f"e04_{mode}_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "cell", "RMSE", "MAE", "MAPE (%)", "R2", "MaxE"])
        for cell_name, m in eval_metrics.items():
            writer.writerow([mode, cell_name, m["RMSE"], m["MAE"], m["MAPE (%)"], m["R2"], m["MaxE"]])

    # Print results
    print(f"\n{mode.upper()} Results:")
    for cell_name, m in eval_metrics.items():
        print(f"  {cell_name:<12} | RMSE: {m['RMSE']:.5f} | MAE: {m['MAE']:.5f} | R2: {m['R2']:.5f}")

    return {
        "mode": mode,
        "param_count": param_count,
        "metrics": eval_metrics,
        "train_time": total_train_time,
        "Ea_sei": model.embed.Ea_sei.item(),
        "Ea_pl": model.embed.Ea_pl.item(),
    }


def run_full_e04_experiment(
    epochs: int = 80,
    batch_size: int = 8,
    seed: int = 42,
    patience: int = 20,
    device_str: str = "cpu",
) -> None:
    """Executes the full E04 Ablation Study across both conditions."""
    print("\n" + "="*50)
    print("EXPERIMENT E04: MATCHED CLASSICAL VS QUANTUM REPRESENTATION")
    print("="*50)
    print(f"Running on device: {device_str} (torch threads: {torch.get_num_threads()})")

    results = {}
    modes: list[Literal["classical_physics", "quantum_physics"]] = [
        "classical_physics",
        "quantum_physics",
    ]

    for mode in modes:
        results[mode] = run_condition(
            mode=mode,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            patience=patience,
            device_str=device_str,
        )

    # Export comparison summary CSV
    comparison_csv = E04_OUTPUT_DIR / "metrics" / "e04_comparison.csv"
    cells = ["B0018", "B0032", "B0053_test", "MACRO_AVG"]

    with open(comparison_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cell",
            "Classical_RMSE", "Quantum_RMSE",
            "Classical_MAE", "Quantum_MAE",
            "Classical_R2", "Quantum_R2",
            "Winner",
        ])
        for c in cells:
            cl_rmse = results["classical_physics"]["metrics"][c]["RMSE"]
            q_rmse = results["quantum_physics"]["metrics"][c]["RMSE"]
            cl_mae = results["classical_physics"]["metrics"][c]["MAE"]
            q_mae = results["quantum_physics"]["metrics"][c]["MAE"]
            cl_r2 = results["classical_physics"]["metrics"][c]["R2"]
            q_r2 = results["quantum_physics"]["metrics"][c]["R2"]

            winner = "Quantum" if q_rmse < cl_rmse else "Classical"
            writer.writerow([c, cl_rmse, q_rmse, cl_mae, q_mae, cl_r2, q_r2, winner])

    print("\n" + "="*80)
    print("E04 CLASSICAL VS QUANTUM COMPARISON SUMMARY:")
    print("="*80)
    for c in cells:
        cl_r = results["classical_physics"]["metrics"][c]["RMSE"]
        q_r = results["quantum_physics"]["metrics"][c]["RMSE"]
        winner = "Quantum" if q_r < cl_r else "Classical"
        print(f"  {c:<12} | Classical: {cl_r:.5f} | Quantum: {q_r:.5f} | Winner: {winner}")

    return results


def run_experiment() -> Dict[str, Any]:
    """Standard entry point for registry invocation."""
    return run_full_e04_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E04 Classical vs Quantum Ablation Study")
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs (default: 80)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping (default: 20)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (default: cpu)")
    args = parser.parse_args()

    run_full_e04_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        patience=args.patience,
        device_str=args.device,
    )
