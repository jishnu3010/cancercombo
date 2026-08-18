"""
Cell Line Encoder Module for CancerCombo-BRICS-Symmetric.
Encodes 976-dim landmark gene expression profiles into a 512-dim cell vector representation c.
"""

import torch
import torch.nn as nn


class CellLineEncoder(nn.Module):
    """
    Encodes landmark gene expression profiles (e.g., NCI-60 / LINCS L1000 976 genes).

    Architecture:
        Linear(in_dim -> hidden_dim) -> GELU -> LayerNorm(hidden_dim) -> Dropout(0.2)

    Args:
        in_dim: Input gene expression dimension (default: 976).
        hidden_dim: Output cell embedding dimension c (default: 512).
        dropout_rate: Dropout probability (default: 0.2).
    """

    def __init__(
        self,
        in_dim: int = 976,
        hidden_dim: int = 512,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, cell_expr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cell line encoder.

        Args:
            cell_expr: Gene expression tensor of shape (B, 976).

        Returns:
            c: Cell feature representation tensor of shape (B, 512).
        """
        # Shape assertion for input
        assert cell_expr.dim() == 2, (
            f"Expected cell_expr to be 2D tensor (B, {self.in_dim}), got shape {tuple(cell_expr.shape)}"
        )
        assert cell_expr.size(1) == self.in_dim, (
            f"Expected gene expression dim {self.in_dim}, got {cell_expr.size(1)}"
        )

        # Stage 1: Linear(976 -> 512) -> GELU -> LayerNorm -> Dropout(0.2)
        c = self.encoder(cell_expr)  # Shape: (B, 512)

        # Shape assertion for output
        assert c.shape == (cell_expr.size(0), self.hidden_dim), (
            f"Cell encoder output shape mismatch: expected ({cell_expr.size(0)}, {self.hidden_dim}), got {tuple(c.shape)}"
        )

        return c
