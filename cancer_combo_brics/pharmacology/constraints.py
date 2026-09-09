"""Pharmacological parameter constraint transformation module matching finalcheck."""

from __future__ import annotations

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintTransform(nn.Module):
    """Transforms raw neural outputs into biologically and mathematically valid pharmacological parameters.

    Target Scale Convention (matching finalcheck):
      Percentage viability where baseline e0 = 100.0 (100% viability).

    Official DeepSynBa Parameter Transformations:
      - e0: Fixed to 100.0
      - e1, e2, e3: sigmoid(raw) in (0, 1) (fractional efficacy)
      - logC1, logC2: raw (unconstrained natural log concentration in uM)
      - h1, h2: relu(raw) >= 0
      - alpha: relu(raw) >= 0
    """

    def __init__(self, e0: float = 100.0, **kwargs):
        super().__init__()
        self.e0_val = float(e0)

    def forward_tuple(
        self,
        raw_tuple: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Transforms 8-tuple of raw parameters matching finalcheck forward."""
        raw_e1, raw_e2, raw_e3, raw_log_c1, raw_log_c2, raw_h1, raw_h2, raw_alpha = raw_tuple
        e1 = torch.sigmoid(raw_e1)
        e2 = torch.sigmoid(raw_e2)
        e3 = torch.sigmoid(raw_e3)
        log_c1 = raw_log_c1
        log_c2 = raw_log_c2
        h1 = F.relu(raw_h1)
        h2 = F.relu(raw_h2)
        alpha = F.relu(raw_alpha)
        return e1, e2, e3, log_c1, log_c2, h1, h2, alpha

    def forward(
        self,
        raw_params: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Transform raw parameters into constrained domain.

        Args:
            raw_params: Dict or Tuple of 8 raw parameter tensors.

        Returns:
            constrained: Dict of constrained parameter tensors.
            diagnostics: Parameter distribution and saturation diagnostics.
        """
        if isinstance(raw_params, (list, tuple)):
            e1, e2, e3, logc1, logc2, h1, h2, alpha = self.forward_tuple(tuple(raw_params))
            B = e1.shape[0]
            device = e1.device
            e0 = torch.full((B,), self.e0_val, dtype=torch.float32, device=device)
            constrained = {
                "e0": e0,
                "e1": e1.squeeze(-1) if e1.dim() > 1 else e1,
                "e2": e2.squeeze(-1) if e2.dim() > 1 else e2,
                "e3": e3.squeeze(-1) if e3.dim() > 1 else e3,
                "logc1": logc1.squeeze(-1) if logc1.dim() > 1 else logc1,
                "logc2": logc2.squeeze(-1) if logc2.dim() > 1 else logc2,
                "h1": h1.squeeze(-1) if h1.dim() > 1 else h1,
                "h2": h2.squeeze(-1) if h2.dim() > 1 else h2,
                "alpha": alpha.squeeze(-1) if alpha.dim() > 1 else alpha,
            }
        else:
            B = next(iter(raw_params.values())).shape[0]
            device = next(iter(raw_params.values())).device
            e0 = torch.full((B,), self.e0_val, dtype=torch.float32, device=device)

            e1 = torch.sigmoid(raw_params["e1_raw"])
            e2 = torch.sigmoid(raw_params["e2_raw"])
            e3 = torch.sigmoid(raw_params["e3_raw"])

            logc1 = raw_params["logc1_raw"]
            logc2 = raw_params["logc2_raw"]

            h1 = F.relu(raw_params["h1_raw"])
            h2 = F.relu(raw_params["h2_raw"])

            alpha = F.relu(raw_params["alpha_raw"])

            constrained = {
                "e0": e0,
                "e1": e1,
                "e2": e2,
                "e3": e3,
                "logc1": logc1,
                "logc2": logc2,
                "h1": h1,
                "h2": h2,
                "alpha": alpha,
            }

        # Diagnostics
        with torch.no_grad():
            diagnostics = {}
            for name, val in constrained.items():
                if name == "e0":
                    continue
                diagnostics[f"param_{name}_mean"] = val.mean().item()
                diagnostics[f"param_{name}_std"] = val.std().item() if val.numel() > 1 else 0.0
                diagnostics[f"param_{name}_min"] = val.min().item()
                diagnostics[f"param_{name}_max"] = val.max().item()

            # Check for NaN / Inf
            for name, val in constrained.items():
                has_nan = torch.isnan(val).any().item()
                has_inf = torch.isinf(val).any().item()
                if has_nan or has_inf:
                    diagnostics[f"param_{name}_nan_or_inf"] = 1.0

        return constrained, diagnostics

