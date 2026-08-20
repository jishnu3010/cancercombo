"""
Residual FiLM (Feature-wise Linear Modulation) Conditioning Module for CancerCombo.
Conditions fragment embeddings F_norm on the cell vector c using shared gamma and beta MLPs
with an explicit identity residual connection: F_tilde = F_norm + gamma(c) * F_norm + beta(c).
"""

import torch
import torch.nn as nn


class FiLMConditioning(nn.Module):
    """
    Residual Feature-wise Linear Modulation (FiLM) layer.

    Maps cell representation vector c (512-dim) to scale gamma and shift beta parameters (128-dim)
    and applies them to normalized fragment representations F_norm via an explicit identity residual connection:
        gamma = g_gamma(c), beta = g_beta(c)
        F_tilde = F_norm + gamma * F_norm + beta = (1 + gamma) * F_norm + beta

    Initializes the final projection layers of g_gamma and g_beta to zero weights and zero biases,
    ensuring gamma(c) ≈ 0 and beta(c) ≈ 0 at initialization, providing an exact identity mapping F_tilde ≈ F_norm.

    Args:
        cell_dim: Input cell vector dimension (default: 512).
        d_dim: Fragment embedding dimension d (default: 128).
    """

    def __init__(self, cell_dim: int = 512, d_dim: int = 128):
        super().__init__()
        self.cell_dim = cell_dim
        self.d_dim = d_dim

        # Generator MLP for gamma (scale modulation)
        self.g_gamma = nn.Sequential(
            nn.Linear(cell_dim, d_dim),
            nn.GELU(),
            nn.Linear(d_dim, d_dim)
        )

        # Generator MLP for beta (shift modulation)
        self.g_beta = nn.Sequential(
            nn.Linear(cell_dim, d_dim),
            nn.GELU(),
            nn.Linear(d_dim, d_dim)
        )

        # Zero-initialize final projection layers for exact identity initialization: F_tilde = F_norm + 0 + 0 = F_norm
        nn.init.zeros_(self.g_gamma[-1].weight)
        nn.init.zeros_(self.g_gamma[-1].bias)
        nn.init.zeros_(self.g_beta[-1].weight)
        nn.init.zeros_(self.g_beta[-1].bias)

    def forward(self, F_norm: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Residual FiLM conditioning.

        Args:
            F_norm: Normalized fragment embedding tensor of shape (B, N_frag, d_dim).
            c: Cell feature representation tensor of shape (B, cell_dim).

        Returns:
            F_tilde: Residual cell-conditioned fragment tensor of shape (B, N_frag, d_dim).
        """
        assert F_norm.dim() == 3, f"Expected F_norm to be 3D tensor (B, N_frag, {self.d_dim}), got {tuple(F_norm.shape)}"
        assert c.dim() == 2, f"Expected c to be 2D tensor (B, {self.cell_dim}), got {tuple(c.shape)}"
        assert F_norm.size(0) == c.size(0), f"Batch size mismatch: F_norm has {F_norm.size(0)}, c has {c.size(0)}"
        assert F_norm.size(2) == self.d_dim, f"F_norm embedding dim mismatch: expected {self.d_dim}, got {F_norm.size(2)}"
        assert c.size(1) == self.cell_dim, f"Cell vector dim mismatch: expected {self.cell_dim}, got {c.size(1)}"

        # Compute scale gamma and shift beta parameters from cell vector c
        gamma = self.g_gamma(c)  # Shape: (B, d_dim)
        beta = self.g_beta(c)    # Shape: (B, d_dim)

        # Broadcast gamma and beta across fragment dimension (B, 1, d_dim)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # Explicit Residual FiLM formulation: F_tilde = F_norm + gamma * F_norm + beta
        F_tilde = F_norm + gamma * F_norm + beta  # Shape: (B, N_frag, d_dim)

        assert F_tilde.shape == F_norm.shape, (
            f"Residual FiLM output shape mismatch: expected {tuple(F_norm.shape)}, got {tuple(F_tilde.shape)}"
        )

        return F_tilde
