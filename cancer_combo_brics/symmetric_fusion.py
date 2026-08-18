"""
Symmetric Fusion Module for CancerCombo-BRICS-Symmetric.

Fuses direction-specific pooled statistics (A<-B and B<-A) using symmetric mean and absolute difference operations
to guarantee exact drug-order invariance at the feature level.
"""

import torch
import torch.nn as nn


class SymmetricFusion(nn.Module):
    """
    Symmetric A/B Fusion module for drug combination interactions.

    Computes:
        mean_sym  = (mu_A_from_B + mu_B_from_A) / 2
        mean_diff = |mu_A_from_B - mu_B_from_A|
        max_sym   = (p_A_from_B + p_B_from_A) / 2
        max_diff  = |p_A_from_B - p_B_from_A|
        r_AB_sym  = concat(mean_sym, mean_diff, max_sym, max_diff)   [shape (B, 4d)]

    Args:
        d_dim: Fragment embedding dimension d (default: 128).
    """

    def __init__(self, d_dim: int = 128):
        super().__init__()
        self.d_dim = d_dim
        self.out_dim = 4 * d_dim

    def forward(
        self,
        mu_A_from_B: torch.Tensor,
        p_A_from_B: torch.Tensor,
        mu_B_from_A: torch.Tensor,
        p_B_from_A: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for symmetric fusion.

        Args:
            mu_A_from_B: Masked mean pooled tensor for A <- B, shape (B, d).
            p_A_from_B: Masked max pooled tensor for A <- B, shape (B, d).
            mu_B_from_A: Masked mean pooled tensor for B <- A, shape (B, d).
            p_B_from_A: Masked max pooled tensor for B <- A, shape (B, d).

        Returns:
            r_AB_sym: Symmetric fused representation tensor of shape (B, 4d).
        """
        B = mu_A_from_B.size(0)
        assert mu_A_from_B.shape == (B, self.d_dim), f"mu_A_from_B shape mismatch: got {tuple(mu_A_from_B.shape)}"
        assert p_A_from_B.shape == (B, self.d_dim), f"p_A_from_B shape mismatch: got {tuple(p_A_from_B.shape)}"
        assert mu_B_from_A.shape == (B, self.d_dim), f"mu_B_from_A shape mismatch: got {tuple(mu_B_from_A.shape)}"
        assert p_B_from_A.shape == (B, self.d_dim), f"p_B_from_A shape mismatch: got {tuple(p_B_from_A.shape)}"

        # 1. Symmetric mean and absolute difference for mean statistics
        mean_sym = (mu_A_from_B + mu_B_from_A) / 2.0
        mean_diff = torch.abs(mu_A_from_B - mu_B_from_A)

        # 2. Symmetric mean and absolute difference for max statistics
        max_sym = (p_A_from_B + p_B_from_A) / 2.0
        max_diff = torch.abs(p_A_from_B - p_B_from_A)

        # 3. Concatenate all 4 symmetric statistics -> shape (B, 4d)
        r_AB_sym = torch.cat([mean_sym, mean_diff, max_sym, max_diff], dim=-1)

        assert r_AB_sym.shape == (B, self.out_dim), (
            f"r_AB_sym shape mismatch: expected ({B}, {self.out_dim}), got {tuple(r_AB_sym.shape)}"
        )

        return r_AB_sym
