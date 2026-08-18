"""
FiLM (Feature-wise Linear Modulation) Conditioning Module for CancerCombo-BRICS-Symmetric.
Conditions fragment embeddings F on the cell vector c using shared gamma and beta MLPs across drugs.
"""

import torch
import torch.nn as nn


class FiLMConditioning(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Maps cell representation vector c (512-dim) to scale gamma and shift beta parameters (d-dim)
    and applies them to fragment representations F:
        gamma_A = g_gamma(c), beta_A = g_beta(c) -> F_tilde_A = gamma_A * F_A + beta_A
        gamma_B = g_gamma(c), beta_B = g_beta(c) -> F_tilde_B = gamma_B * F_B + beta_B

    The same g_gamma and g_beta MLPs are shared across Drug A and Drug B.

    Args:
        cell_dim: Input cell vector dimension (default: 512).
        d_dim: Fragment embedding dimension d (default: 128).
    """

    def __init__(self, cell_dim: int = 512, d_dim: int = 128):
        super().__init__()
        self.cell_dim = cell_dim
        self.d_dim = d_dim

        # Generator MLP for gamma (scale factor)
        self.g_gamma = nn.Sequential(
            nn.Linear(cell_dim, d_dim),
            nn.GELU(),
            nn.Linear(d_dim, d_dim)
        )

        # Generator MLP for beta (shift factor)
        self.g_beta = nn.Sequential(
            nn.Linear(cell_dim, d_dim),
            nn.GELU(),
            nn.Linear(d_dim, d_dim)
        )

    def forward(self, F: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FiLM conditioning.

        Args:
            F: Fragment embedding tensor of shape (B, N_frag, d_dim).
            c: Cell feature representation tensor of shape (B, cell_dim).

        Returns:
            F_tilde: Cell-conditioned fragment tensor of shape (B, N_frag, d_dim).
        """
        assert F.dim() == 3, f"Expected F to be 3D tensor (B, N_frag, {self.d_dim}), got {tuple(F.shape)}"
        assert c.dim() == 2, f"Expected c to be 2D tensor (B, {self.cell_dim}), got {tuple(c.shape)}"
        assert F.size(0) == c.size(0), f"Batch size mismatch: F has {F.size(0)}, c has {c.size(0)}"
        assert F.size(2) == self.d_dim, f"F embedding dim mismatch: expected {self.d_dim}, got {F.size(2)}"
        assert c.size(1) == self.cell_dim, f"Cell vector dim mismatch: expected {self.cell_dim}, got {c.size(1)}"

        # Compute scale gamma and shift beta parameters from cell vector c
        gamma = self.g_gamma(c)  # Shape: (B, d_dim)
        beta = self.g_beta(c)    # Shape: (B, d_dim)

        # Broadcast gamma and beta across fragment dimension (B, 1, d_dim) -> (B, N_frag, d_dim)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # FiLM operation: elementwise scale + shift broadcast across sequence length
        F_tilde = gamma * F + beta  # Shape: (B, N_frag, d_dim)

        assert F_tilde.shape == F.shape, (
            f"FiLM output shape mismatch: expected {tuple(F.shape)}, got {tuple(F_tilde.shape)}"
        )

        return F_tilde
