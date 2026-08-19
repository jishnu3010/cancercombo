"""
Differentiable Bivariate Hill / SynBa Surface Solver Module for CancerCombo-BRICS-Symmetric.

Computes full 2D viability surface Y in R^(B x M x N) analytically and differentiably
from predicted parameters and concentration grids (doses_A, doses_B).
"""

from typing import Dict, Union, Tuple
import torch
import torch.nn as nn


class BivariateHillSolver(nn.Module):
    """
    Closed-form, fully differentiable Bivariate Hill / SynBa dose-response solver.

    Computes the 2D viability surface Y(d_A, d_B) across dose grids:
        u_A = (d_A / c1)^{h1}
        u_B = (d_B / c2)^{h2}
        Y(d_A, d_B) = (e0 + e1 * u_A + e2 * u_B + e12 * alpha * u_A * u_B) / (1 + u_A + u_B + alpha * u_A * u_B)

    Supports backpropagation through predicted parameters (e0, e1, e2, e12, c1, c2, h1, h2, alpha).
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        params: Dict[str, torch.Tensor],
        dose_grid: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass computing the 2D viability surface Y tensor.

        Args:
            params: Dictionary containing predicted parameter tensors of shape (B, 1):
                - 'e0': Baseline viability
                - 'e1': E_max Drug A
                - 'e2': E_max Drug B
                - 'e12': E_max Combination
                - 'c1': EC50 Drug A
                - 'c2': EC50 Drug B
                - 'h1': Hill slope Drug A
                - 'h2': Hill slope Drug B
                - 'alpha': Potency synergy interaction term
            dose_grid: Dose concentrations for Drug A and Drug B.
                Can be a tuple (doses_A, doses_B):
                    - doses_A: Tensor of shape (M,) or (B, M)
                    - doses_B: Tensor of shape (N,) or (B, N)
                Or a single tensor of shape (B, M, 2) or (M, 2) containing doses.

        Returns:
            Y: Viability surface tensor of shape (B, M, N), with values in [0, 1].
        """
        e0 = params["e0"]      # Shape: (B, 1)
        e1 = params["e1"]      # Shape: (B, 1)
        e2 = params["e2"]      # Shape: (B, 1)
        e12 = params["e12"]    # Shape: (B, 1)
        c1 = params["c1"]      # Shape: (B, 1)
        c2 = params["c2"]      # Shape: (B, 1)
        h1 = params["h1"]      # Shape: (B, 1)
        h2 = params["h2"]      # Shape: (B, 1)
        alpha = params["alpha"]# Shape: (B, 1)

        B = e0.size(0)

        # Parse dose_grid inputs into doses_A of shape (B, M) and doses_B of shape (B, N)
        if isinstance(dose_grid, (tuple, list)):
            doses_A, doses_B = dose_grid[0], dose_grid[1]
        elif isinstance(dose_grid, torch.Tensor):
            if dose_grid.dim() == 3 and dose_grid.size(-1) == 2:
                doses_A = dose_grid[..., 0]
                doses_B = dose_grid[..., 1]
            else:
                raise ValueError(f"Unsupported dose_grid tensor shape: {tuple(dose_grid.shape)}")
        else:
            raise TypeError(f"dose_grid must be tuple of Tensors or Tensor, got {type(dose_grid)}")

        # Ensure doses_A is shape (B, M)
        if doses_A.dim() == 1:
            doses_A = doses_A.unsqueeze(0).expand(B, -1)
        elif doses_A.dim() == 2 and doses_A.size(0) == 1:
            doses_A = doses_A.expand(B, -1)

        # Ensure doses_B is shape (B, N)
        if doses_B.dim() == 1:
            doses_B = doses_B.unsqueeze(0).expand(B, -1)
        elif doses_B.dim() == 2 and doses_B.size(0) == 1:
            doses_B = doses_B.expand(B, -1)

        assert doses_A.size(0) == B and doses_B.size(0) == B, (
            f"Batch size mismatch in doses: expected batch size {B}, got doses_A {doses_A.size(0)}, doses_B {doses_B.size(0)}"
        )

        M = doses_A.size(1)
        N = doses_B.size(1)

        # Reshape doses for 2D grid broadcasting:
        # doses_A -> (B, M, 1), doses_B -> (B, 1, N)
        dA = doses_A.unsqueeze(2)  # Shape: (B, M, 1)
        dB = doses_B.unsqueeze(1)  # Shape: (B, 1, N)

        # Reshape parameters for broadcasting across (M, N) grid: (B, 1) -> (B, 1, 1)
        e0_b = e0.unsqueeze(2)
        e1_b = e1.unsqueeze(2)
        e2_b = e2.unsqueeze(2)
        e12_b = e12.unsqueeze(2)
        c1_b = c1.unsqueeze(2)
        c2_b = c2.unsqueeze(2)
        h1_b = h1.unsqueeze(2)
        h2_b = h2.unsqueeze(2)
        alpha_b = alpha.unsqueeze(2)

        # Differentiable computation of normalized dose effects u_A and u_B with numerical bounds
        c1_safe = c1_b.clamp(min=1e-8)
        c2_safe = c2_b.clamp(min=1e-8)
        h1_safe = h1_b.clamp(min=1e-4, max=10.0)
        h2_safe = h2_b.clamp(min=1e-4, max=10.0)

        ratio_A = (dA / c1_safe).clamp(min=self.eps, max=1e4)
        ratio_B = (dB / c2_safe).clamp(min=self.eps, max=1e4)

        u_A = torch.where(
            dA > self.eps,
            ratio_A ** h1_safe,
            torch.zeros_like(dA)
        ).clamp(max=1e6)

        u_B = torch.where(
            dB > self.eps,
            ratio_B ** h2_safe,
            torch.zeros_like(dB)
        ).clamp(max=1e6)

        # Compute Bivariate Hill surface terms: u_AB shape (B, M, N)
        u_AB = (u_A * u_B).clamp(max=1e8)

        numerator = e0_b + e1_b * u_A + e2_b * u_B + e12_b * alpha_b * u_AB
        denominator = 1.0 + u_A + u_B + alpha_b * u_AB

        # Cell Viability Surface Y bounded in [0, 1]
        Y = (numerator / denominator).clamp(min=0.0, max=1.0)

        assert Y.shape == (B, M, N), (
            f"Viability surface Y shape mismatch: expected ({B}, {M}, {N}), got {tuple(Y.shape)}"
        )

        return Y
