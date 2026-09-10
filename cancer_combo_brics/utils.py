"""General utilities for reproducibility, checkpointing, and model profiling."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple
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


def save_rng_state() -> Dict[str, Any]:
    """Capture current RNG state across Python, NumPy, PyTorch CPU, and PyTorch CUDA."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_dict: Optional[Dict[str, Any]]) -> None:
    """Restore RNG state across Python, NumPy, PyTorch CPU, and PyTorch CUDA where supported."""
    if not rng_dict or not isinstance(rng_dict, dict):
        return

    if "python" in rng_dict and rng_dict["python"] is not None:
        try:
            random.setstate(rng_dict["python"])
        except Exception:
            pass

    if "numpy" in rng_dict and rng_dict["numpy"] is not None:
        try:
            np.random.set_state(rng_dict["numpy"])
        except Exception:
            pass

    if "torch_cpu" in rng_dict and rng_dict["torch_cpu"] is not None:
        try:
            cpu_state = rng_dict["torch_cpu"]
            if isinstance(cpu_state, torch.Tensor):
                cpu_state = cpu_state.cpu()
            torch.set_rng_state(cpu_state)
        except Exception:
            pass

    if "torch_cuda" in rng_dict and rng_dict["torch_cuda"] is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng_dict["torch_cuda"])
        except Exception:
            pass


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    epoch: int = 0,
    best_metric: float = float("inf"),
    training_history: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    cell_preprocessor_stats: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    arch_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically save training checkpoint with full model, optimizer, scheduler, scaler, RNG, and history state."""
    abs_filepath = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(abs_filepath), exist_ok=True)

    default_arch = {
        "pooling_mode": "mean_max",
        "r_AB_dim": 512,
        "r_DC_dim": 1536,
        "mol2vec": "pretrained_300dim",
        "mol2vec_frozen": True,
    }
    if arch_metadata is not None:
        default_arch.update(arch_metadata)

    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "seed": seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "cell_preprocessor_stats": cell_preprocessor_stats,
        "training_history": training_history if training_history is not None else [],
        "rng_state": save_rng_state(),
        "arch_metadata": default_arch,
    }

    tmp_filepath = abs_filepath + ".tmp"
    try:
        torch.save(checkpoint, tmp_filepath)
        os.replace(tmp_filepath, abs_filepath)
    except Exception as e:
        if os.path.exists(tmp_filepath):
            try:
                os.remove(tmp_filepath)
            except Exception:
                pass
        raise RuntimeError(f"Failed to save checkpoint atomically to '{filepath}': {e}") from e


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: str = "cpu",
    validate_arch: bool = True,
) -> Dict[str, Any]:
    """Load model, optimizer, scheduler, scaler, history, and RNG state from checkpoint with strict validation."""
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: '{filepath}'")

    try:
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Corrupted or invalid checkpoint file at '{filepath}': {e}") from e

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Corrupted checkpoint structure in '{filepath}': missing 'model_state_dict'")

    # Validate architecture metadata if present
    if validate_arch and "arch_metadata" in checkpoint:
        arch = checkpoint["arch_metadata"]
        expected_r_AB = 512
        expected_r_DC = 1536
        if arch.get("r_AB_dim") != expected_r_AB or arch.get("r_DC_dim") != expected_r_DC:
            raise ValueError(
                f"Architecture mismatch in checkpoint '{filepath}': "
                f"expected r_AB={expected_r_AB}, r_DC={expected_r_DC}, got r_AB={arch.get('r_AB_dim')}, r_DC={arch.get('r_DC_dim')}"
            )
        if arch.get("pooling_mode") != "mean_max":
            raise ValueError(
                f"Architecture mismatch in checkpoint '{filepath}': "
                f"expected pooling_mode='mean_max', got '{arch.get('pooling_mode')}'"
            )
        if arch.get("mol2vec") != "pretrained_300dim":
            raise ValueError(
                f"Architecture mismatch in checkpoint '{filepath}': "
                f"expected mol2vec='pretrained_300dim', got '{arch.get('mol2vec')}'"
            )
        if arch.get("mol2vec_frozen") is not True:
            raise ValueError(
                f"Architecture mismatch in checkpoint '{filepath}': "
                f"expected mol2vec_frozen=True, got {arch.get('mol2vec_frozen')}"
            )

    # Restore model parameters
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as e:
        try:
            missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if missing:
                print(f"[WARNING] Missing keys when loading checkpoint '{filepath}': {missing}")
            if unexpected:
                print(f"[WARNING] Unexpected keys in checkpoint '{filepath}' were ignored: {len(unexpected)} keys")
        except Exception as inner_e:
            raise ValueError(f"Failed to load model state_dict from '{filepath}': {e}") from inner_e

    # Ensure pretrained Mol2Vec embeddings remain frozen
    if hasattr(model, "fragment_encoder") and hasattr(model.fragment_encoder, "mol2vec"):
        mol2vec = model.fragment_encoder.mol2vec
        if hasattr(mol2vec, "embeddings"):
            mol2vec.embeddings.weight.requires_grad = False

    # Restore optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore scaler state
    if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Restore RNG state
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])

    return checkpoint


def get_gpu_memory_mb() -> float:
    """Return current allocated GPU memory in MB if CUDA is available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0
