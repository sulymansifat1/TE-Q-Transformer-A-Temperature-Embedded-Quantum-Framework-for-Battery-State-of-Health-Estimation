"""Training engine for TE-Q-Transformer under the NASA multi-temperature protocol."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Experiment.utils.paths import NASA_DATA_DIR, RESULT_NASA_DIR
from Experiment.utils.seed import seed_everything
from Experiment.utils.plotting import plot_loss_curves
from Experiment.Dataset.nasa import get_nasa_dataloaders
from Experiment.Proposed_Model.model import TEQTransformer, TEQTransformerConfig, rich_entangler_config
from Experiment.Proposed_Model.config import get_training_hyperparameters


def train_proposed_model(
    data_dir: Path = NASA_DATA_DIR,
    model_cfg: TEQTransformerConfig | None = None,
    output_dir: Path | None = None,
    batch_size: int = 8,
    num_epochs: int = 80,
    patience: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-2,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
    device: torch.device | None = None,
) -> Tuple[TEQTransformer, Dict[str, list], Path]:
    """Trains the TE-Q-Transformer model on NASA battery data."""
    seed_everything(seed)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    if output_dir is None:
        output_dir = RESULT_NASA_DIR / "checkpoints" / "proposed_teq"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loaders, scaler = get_nasa_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
    )

    model_cfg = model_cfg or rich_entangler_config()
    model = TEQTransformer(model_cfg).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience)
    criterion = nn.MSELoss()

    history: Dict[str, list] = {"epoch": [], "train_loss": [], "lr": [], "epoch_seconds": []}
    best_loss = float("inf")
    patience_counter = 0
    best_checkpoint_path = output_dir / "best_model.pth"

    print(f"[Training] TE-Q-Transformer on {device} for max {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        epoch_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}: {loss.item()}")

            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        elapsed = time.perf_counter() - t0
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(epoch)
        history["train_loss"].append(mean_loss)
        history["lr"].append(current_lr)
        history["epoch_seconds"].append(elapsed)

        scheduler.step(mean_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {mean_loss:.6f} | LR: {current_lr:.2e} | Time: {elapsed:.2f}s")

        # Early stopping and checkpointing
        if mean_loss < best_loss:
            best_loss = mean_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    # Save training history and loss plot
    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_loss_curves(
        history["train_loss"],
        val_loss=None,
        output_path=output_dir / "training_loss_curve.png",
        title="TE-Q-Transformer Training Loss Curve",
    )

    # Load best weights before returning
    if best_checkpoint_path.exists():
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    return model, history, best_checkpoint_path
