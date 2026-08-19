"""
Constraint & Transform Stage Module for CancerCombo-BRICS-Symmetric.

Applies parameter-appropriate constraints and mathematical transformations:
    - Sigmoid for viability/efficacy bounds (e0, e1, e2, e12) in [0, 1]
    - Softplus for positivity on EC50s (c1, c2), Hill slopes (h1, h2), and interaction term (alpha)
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintTransform(nn.Module):
    """
    Explicit Constraint and Transformation module.

    Applies parameter-appropriate transformations:
        - e0: 0.5 + 0.6 * Sigmoid (baseline viability ~1.0)
        - e1, e2, e12: Sigmoid (efficacy bounds in [0, 1])
        - c1, c2: Softplus + 1e-4 (EC50 positivity > 0)
        - h1, h2: Softplus + 0.1 (Hill slope positivity > 0)
        - alpha: Softplus + 1e-4 (Synergy interaction > 0)
    """

    def __init__(self):
        super().__init__()

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

        c1 = F.softplus(raw_params["c1"]) + 1e-4
        c2 = F.softplus(raw_params["c2"]) + 1e-4

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
            "alpha": alpha
        }

        # Shape assertions and positivity assertions
        first_key = list(params.keys())[0]
        B = params[first_key].size(0)
        for k, v in params.items():
            assert v.shape == (B, 1), f"Transformed parameter '{k}' shape mismatch: expected ({B}, 1), got {tuple(v.shape)}"

        return params
