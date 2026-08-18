"""
Fragment Encoder Module for CancerCombo-BRICS-Symmetric.
Encodes raw fragment descriptors (e.g., Morgan fingerprints) into d-dimensional fragment embeddings F.
Shared across Drug A and Drug B.
"""

import torch
import torch.nn as nn


class FragmentEncoder(nn.Module):
    """
    Shared fragment encoder mapping fragment fingerprints/descriptors to d-dim vector space.

    Architecture:
        Linear(in_bits -> intermediate_dim) -> GELU -> LayerNorm(intermediate_dim) -> Linear(intermediate_dim -> d_dim)

    Args:
        in_bits: Dimension of input fragment representation (e.g., Morgan FP 2048).
        d_dim: Output fragment embedding dimension d (default: 128).
        intermediate_dim: Hidden layer dimension (default: 256).
    """

    def __init__(
        self,
        in_bits: int = 2048,
        d_dim: int = 128,
        intermediate_dim: int = 256
    ):
        super().__init__()
        self.in_bits = in_bits
        self.d_dim = d_dim
        self.intermediate_dim = intermediate_dim

        self.net = nn.Sequential(
            nn.Linear(in_bits, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, d_dim)
        )

    def forward(self, frag_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for fragment encoder.

        Args:
            frag_features: Tensor of shape (B, N_frag, in_bits).

        Returns:
            F: Encoded fragment embeddings tensor of shape (B, N_frag, d_dim).
        """
        assert frag_features.dim() == 3, (
            f"Expected frag_features to be 3D tensor (B, N_frag, {self.in_bits}), got shape {tuple(frag_features.shape)}"
        )
        assert frag_features.size(2) == self.in_bits, (
            f"Expected fingerprint dim {self.in_bits}, got {frag_features.size(2)}"
        )

        B, N_frag, _ = frag_features.shape
        F = self.net(frag_features)  # Shape: (B, N_frag, d_dim)

        assert F.shape == (B, N_frag, self.d_dim), (
            f"Fragment encoder output shape mismatch: expected ({B}, {N_frag}, {self.d_dim}), got {tuple(F.shape)}"
        )

        return F
