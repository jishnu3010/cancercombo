"""General utilities for reproducibility, checkpointing, and model profiling."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters categorized by component."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    molformer_params = 0
    if hasattr(model, "molformer"):
        molformer_params = sum(p.numel() for p in model.molformer.parameters())

    new_params = total_params - molformer_params
    new_trainable = sum(
        p.numel() for name, p in model.named_parameters()
        if "molformer" not in name and p.requires_grad
    )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "molformer_params": molformer_params,
        "new_params": new_params,
        "new_trainable_params": new_trainable,
    }


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    best_metric: float = float("inf"),
    config: Optional[Dict[str, Any]] = None,
    cell_preprocessor_stats: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> None:
    """Save training checkpoint with full state and reproducibility metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "config": config,
        "cell_preprocessor_stats": cell_preprocessor_stats,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load model and optimizer state from checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def get_gpu_memory_mb() -> float:
    """Return current allocated GPU memory in MB if CUDA is available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0
