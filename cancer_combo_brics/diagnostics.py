"""Diagnostic and monitoring utilities for training stability and scientific validation."""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn


def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """Compute global gradient norm and per-module gradient norms before clipping."""
    module_names = [
        "cell_encoder",
        "fragment_encoder",
        "pairwise_interaction",
        "drug_cell",
        "parameter_heads",
        "dose_bias",
    ]

    norms: Dict[str, float] = {}
    total_norm_sq = 0.0

    # Per module
    for mod_name in module_names:
        if hasattr(model, mod_name):
            submod = getattr(model, mod_name)
            submod_norm_sq = 0.0
            for p in submod.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().double().norm(2).item()
                    if np.isfinite(param_norm):
                        submod_norm_sq += param_norm ** 2
            norms[f"grad_norm_{mod_name}"] = float(np.sqrt(submod_norm_sq))

    # Global norm
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().double().norm(2).item()
            if np.isfinite(param_norm):
                total_norm_sq += param_norm ** 2

    norms["global_grad_norm"] = float(np.sqrt(total_norm_sq))
    return norms


def inspect_surface_diagnostics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_hill: torch.Tensor,
    bias: torch.Tensor,
) -> Dict[str, float]:
    """Inspect and validate surface properties to detect bias domination or degeneration.

    Args:
        y_true: Ground truth surface tensor (B, D_A, D_B).
        y_pred: Final predicted surface (B, D_A, D_B).
        y_hill: Hill-only prediction (B, D_A, D_B).
        bias: Bias matrix (B, D_A, D_B).

    Returns:
        Dict with min, max, mean, std, and bias ratio metrics.
    """
    with torch.no_grad():
        res = y_pred - y_true
        hill_mag = torch.abs(y_hill).mean().item()
        bias_mag = torch.abs(bias).mean().item()
        bias_ratio = bias_mag / max(hill_mag, 1e-6)

        diag = {
            "true_mean": y_true.mean().item(),
            "true_std": y_true.std().item(),
            "true_min": y_true.min().item(),
            "true_max": y_true.max().item(),
            "pred_mean": y_pred.mean().item(),
            "pred_std": y_pred.std().item(),
            "pred_min": y_pred.min().item(),
            "pred_max": y_pred.max().item(),
            "hill_mean": y_hill.mean().item(),
            "hill_std": y_hill.std().item(),
            "bias_mean": bias.mean().item(),
            "bias_std": bias.std().item(),
            "bias_max_abs": torch.abs(bias).max().item(),
            "bias_to_hill_ratio": bias_ratio,
            "residual_mean": res.mean().item(),
            "residual_std": res.std().item(),
            "is_pred_finite": float(torch.all(torch.isfinite(y_pred)).item()),
        }

    return diag
