"""Experiment E05: Locked 11-entry baseline comparison (10 baselines + 1 reference).

LOCKED E05 ACTIVE MODEL SET (NASA ONLY):
Classical (8):
  - LSTM, GRU, CNN1D, TCN, DLinear, Transformer, PatchTST, iTransformer
Quantum (2):
  - QLSTM, QGRU
Proposed Reference (1, NOT retrained):
  - TE-Q-Transformer

IMPORTANT SAFETY POLICY (implementation stage):
- Default execution mode is **dry_run** (no full training).
- Use **sanity_train** for 1–3 epochs on a tiny subset only.
- Full benchmark training is intentionally disabled in this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Iterable

import numpy as np
import torch
from torch import nn, optim

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiment.Baseline import get_baseline_model_class
from Experiment.Dataset.nasa import get_nasa_dataloaders
from Experiment.Proposed_Model.model import TEQTransformer, rich_entangler_config
from Experiment.utils.paths import ROOT_DIR
from Experiment.utils.seed import seed_everything


E05_OUT = ROOT_DIR / "GarbageResults" / "BaselineE05"
E05_DRY = E05_OUT / "dry_run"
E05_SANITY = E05_OUT / "sanity_train"


LOCKED_E05_MODELS: List[str] = [
    "LSTM",
    "GRU",
    "CNN1D",
    "TCN",
    "DLinear",
    "Transformer",
    "PatchTST",
    "iTransformer",
    "QLSTM",
    "QGRU",
    "TE-Q-Transformer",
]


def _ensure_dirs() -> None:
    for d in [E05_OUT, E05_DRY, E05_SANITY, E05_OUT / "provenance", E05_OUT / "audit"]:
        d.mkdir(parents=True, exist_ok=True)


def _param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _iter_first_n(loader: Iterable, n: int):
    for i, batch in enumerate(loader):
        if i >= n:
            break
        yield batch


def dry_run_one_model(model_name: str, batch_size: int, seed: int, device: str) -> Dict[str, Any]:
    seed_everything(seed)
    dev = torch.device(device)
    train_loader, _, _ = get_nasa_dataloaders(batch_size=batch_size, shuffle_train=True)
    bx, by = next(iter(train_loader))
    bx = bx.to(dev)
    by = by.to(dev)

    if model_name == "TE-Q-Transformer":
        model = TEQTransformer(rich_entangler_config()).to(dev)
    else:
        model_cls = get_baseline_model_class(model_name)
        model = model_cls().to(dev)

    model.train()
    n_params = _param_count(model)

    out = model(bx)
    if out.ndim != 1 or out.shape[0] != bx.shape[0]:
        raise RuntimeError(f"{model_name}: expected output [B], got {tuple(out.shape)}")
    loss = nn.MSELoss()(out, by)
    if not torch.isfinite(loss):
        raise RuntimeError(f"{model_name}: loss is not finite: {loss.item()}")

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    opt.zero_grad()
    loss.backward()
    opt.step()

    return {
        "model": model_name,
        "device": str(dev),
        "batch_size": int(batch_size),
        "param_count": n_params,
        "loss": float(loss.detach().cpu().item()),
        "ok": True,
    }


def sanity_train_one_model(
    model_name: str,
    batch_size: int,
    seed: int,
    device: str,
    epochs: int,
    max_train_batches: int,
) -> Dict[str, Any]:
    seed_everything(seed)
    dev = torch.device(device)
    train_loader, _, _ = get_nasa_dataloaders(batch_size=batch_size, shuffle_train=True)

    if model_name == "TE-Q-Transformer":
        return {
            "model": model_name,
            "device": str(dev),
            "skipped": True,
            "reason": "Proposed model is not sanity-trained in E05 baseline implementation stage.",
        }

    model_cls = get_baseline_model_class(model_name)
    model = model_cls().to(dev)
    model.train()

    n_params = _param_count(model)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = nn.MSELoss()

    loss_hist: List[float] = []
    for _ep in range(1, epochs + 1):
        losses = []
        for bx, by in _iter_first_n(train_loader, max_train_batches):
            bx = bx.to(dev)
            by = by.to(dev)
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{model_name}: non-finite loss in sanity train: {loss.item()}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        loss_hist.append(float(np.mean(losses)) if losses else float("nan"))

    return {
        "model": model_name,
        "device": str(dev),
        "batch_size": int(batch_size),
        "param_count": n_params,
        "epochs": int(epochs),
        "max_train_batches": int(max_train_batches),
        "loss_history": loss_hist,
        "ok": True,
    }


def run_experiment() -> Dict[str, Any]:
    """Standard entry point for registry invocation (safe dry-run)."""
    _ensure_dirs()
    res = {}
    for m in LOCKED_E05_MODELS:
        try:
            res[m] = dry_run_one_model(m, batch_size=8, seed=42, device="cpu")
        except Exception as e:
            res[m] = {"model": m, "ok": False, "error": repr(e)}
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment E05: baseline implementation validation (dry_run/sanity_train).")
    parser.add_argument("--mode", type=str, default="dry_run", choices=["dry_run", "sanity_train"])
    parser.add_argument("--model", type=str, default="all", help="One model name, or 'all' for locked suite.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", help="Device for dry/sanity ('cpu' recommended for quantum).")
    parser.add_argument("--epochs", type=int, default=2, help="Sanity-train epochs (1–3 recommended).")
    parser.add_argument("--max-train-batches", type=int, default=2, help="Sanity-train train batches (tiny subset).")
    args = parser.parse_args()

    _ensure_dirs()

    selected = LOCKED_E05_MODELS if args.model == "all" else [args.model]
    if args.model != "all" and args.model not in LOCKED_E05_MODELS:
        raise SystemExit(f"Model '{args.model}' is not in the locked E05 set: {LOCKED_E05_MODELS}")

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "models": selected,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
    }

    per_model: Dict[str, Any] = {}
    for m in selected:
        try:
            if args.mode == "dry_run":
                per_model[m] = dry_run_one_model(m, args.batch_size, args.seed, args.device)
            else:
                per_model[m] = sanity_train_one_model(
                    m, args.batch_size, args.seed, args.device, args.epochs, args.max_train_batches
                )
        except Exception as e:
            per_model[m] = {"model": m, "ok": False, "error": repr(e)}

    results["per_model"] = per_model
    out_dir = E05_DRY if args.mode == "dry_run" else E05_SANITY
    model_tag = ("ALL" if args.model == "all" else args.model).replace(" ", "_").replace("/", "_")
    out_path = out_dir / f"e05_{args.mode}_{model_tag}_seed{args.seed}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[E05:{args.mode}] Results written to: {out_path}")


if __name__ == "__main__":
    main()
