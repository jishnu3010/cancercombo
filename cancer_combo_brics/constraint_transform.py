"""
Constraint & Transform Stage Module for CancerCombo-BRICS-Symmetric.

Applies parameter-appropriate constraints and mathematical transformations:
    - e0, e1, e2, e12: Sigmoid (viability bounds in [0, 1])
    - c1, c2: Bounded log10-space parameterization:
          log_c = log_c_min + (log_c_max - log_c_min) * Sigmoid(raw_c)
          C = 10 ** log_c
      Operates smoothly in neural log-space while passing physical concentrations (c1, c2 > 0) to solver.
    - h1, h2: Softplus + 0.1 (Hill slope positivity > 0)
    - alpha: Softplus + 1e-4 (Synergy interaction > 0)
"""

from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintTransform(nn.Module):
    """
    Explicit Constraint and Transformation module with log-space EC50 parameterization.

    Args:
        log_c_min: Lower bound for log10(EC50) concentration (default: -11.0 -> 10^-11 M).
        log_c_max: Upper bound for log10(EC50) concentration (default: -3.0 -> 10^-3 M).
    """

    def __init__(self, log_c_min: float = -11.0, log_c_max: float = -3.0):
        super().__init__()
        self.log_c_min = log_c_min
        self.log_c_max = log_c_max
        self.log_range = log_c_max - log_c_min
        self.ln10 = math.log(10.0)

    def forward(self, raw_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for constraint transformation stage.

        Args:
            raw_params: Dictionary of raw unconstrained parameter tensors (B, 1).

        Returns:
            transformed_params: Dictionary of transformed parameter tensors (B, 1).
        """
        e0 = torch.sigmoid(raw_params["e0"])
        e1 = torch.sigmoid(raw_params["e1"])
        e2 = torch.sigmoid(raw_params["e2"])
        e12 = torch.sigmoid(raw_params["e12"])

        # Log-space EC50 parameterization
        log_c1 = self.log_c_min + self.log_range * torch.sigmoid(raw_params["c1"])
        log_c2 = self.log_c_min + self.log_range * torch.sigmoid(raw_params["c2"])

        c1 = torch.exp(log_c1 * self.ln10)
        c2 = torch.exp(log_c2 * self.ln10)

        h1 = F.softplus(raw_params["h1"]) + 0.1
        h2 = F.softplus(raw_params["h2"]) + 0.1

        alpha = F.softplus(raw_params["alpha"]) + 1e-4

        params = {
            "e0": e0,
            "e1": e1,
            "e2": e2,
            "e12": e12,
            "c1": c1,
            "c2": c2,
            "h1": h1,
            "h2": h2,
            "alpha": alpha,
            "log_c1": log_c1,
            "log_c2": log_c2
        }

        # Shape assertions
        first_key = list(params.keys())[0]
        B = params[first_key].size(0)
        for k, v in params.items():
            assert v.shape == (B, 1), f"Transformed parameter '{k}' shape mismatch: expected ({B}, 1), got {tuple(v.shape)}"

        return params
