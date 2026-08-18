"""
Cell Fusion Module for CancerCombo-BRICS-Symmetric.
Fuses the symmetric drug interaction representation r_AB_sym with cell vector c.
"""

import torch
import torch.nn as nn


class CellFusion(nn.Module):
    """
    Concatenates symmetric cross-attention drug combination vector r_AB_sym (4d-dim)
    with the cell line vector c (512-dim).

    r_final = concat(r_AB_sym, c)   [shape (B, 4d + 512)]

    Args:
        d_dim: Fragment embedding dimension d (default: 128).
        cell_dim: Cell vector dimension c (default: 512).
    """

    def __init__(self, d_dim: int = 128, cell_dim: int = 512):
        super().__init__()
        self.d_dim = d_dim
        self.cell_dim = cell_dim
        self.out_dim = 4 * d_dim + cell_dim

    def forward(self, r_AB_sym: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cell fusion.

        Args:
            r_AB_sym: Symmetric cross-attention representation tensor of shape (B, 4 * d_dim).
            c: Cell representation tensor of shape (B, cell_dim).

        Returns:
            r_final: Fused representation tensor of shape (B, 4 * d_dim + cell_dim).
        """
        assert r_AB_sym.dim() == 2, f"Expected r_AB_sym to be 2D tensor (B, {4 * self.d_dim}), got {tuple(r_AB_sym.shape)}"
        assert c.dim() == 2, f"Expected c to be 2D tensor (B, {self.cell_dim}), got {tuple(c.shape)}"
        assert r_AB_sym.size(0) == c.size(0), f"Batch size mismatch: r_AB_sym has {r_AB_sym.size(0)}, c has {c.size(0)}"
        assert r_AB_sym.size(1) == 4 * self.d_dim, f"r_AB_sym feature dim mismatch: expected {4 * self.d_dim}, got {r_AB_sym.size(1)}"
        assert c.size(1) == self.cell_dim, f"Cell vector dim mismatch: expected {self.cell_dim}, got {c.size(1)}"

        # Concatenate along feature dimension
        r_final = torch.cat([r_AB_sym, c], dim=-1)  # Shape: (B, 4 * d_dim + cell_dim)

        assert r_final.shape == (c.size(0), self.out_dim), (
            f"Fused output shape mismatch: expected ({c.size(0)}, {self.out_dim}), got {tuple(r_final.shape)}"
        )

        return r_final
