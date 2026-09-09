"""Explicit pairwise fragment interaction module without attention."""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn


class ExplicitPairwiseFragmentInteraction(nn.Module):
    """Explicit pairwise interaction for drug functional-group fragments.

    For every pair of fragments (f_Ai, f_Bj) between Drug A (512-D) and Drug B (512-D):
      z_ij = [ f_Ai ; f_Bj ; f_Ai * f_Bj ; |f_Ai - f_Bj| ]  (2048-D)

    Passes z_ij through an explicit interaction MLP:
      2048 -> 512 -> 512

    Aggregates all valid pair representations into a fixed 512-D vector r_AB
    using masked mean pooling (permutation-invariant, non-attention).
    """

    def __init__(
        self,
        fragment_dim: int = 512,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fragment_dim = fragment_dim
        self.hidden_dim = hidden_dim

        # Pairwise interaction network (2048 -> 512 -> 512)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(4 * fragment_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fragment_dim),
            nn.LayerNorm(fragment_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.interaction_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        F_A: torch.Tensor,
        F_B: torch.Tensor,
        mask_A: torch.Tensor,
        mask_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Pairwise fragment interaction forward pass.

        Args:
            F_A: Drug A fragments tensor of shape (B, N_A, 512).
            F_B: Drug B fragments tensor of shape (B, N_B, 512).
            mask_A: Validity mask for Drug A of shape (B, N_A).
            mask_B: Validity mask for Drug B of shape (B, N_B).

        Returns:
            r_AB: Aggregated Drug-Drug interaction representation of shape (B, 512).
            diagnostics: Dictionary containing diagnostic statistics.
        """
        B, N_A, D = F_A.shape
        _, N_B, _ = F_B.shape
        device = F_A.device

        # Expand fragment sets to form all N_A x N_B pairs: (B, N_A, N_B, 512)
        F_A_exp = F_A.unsqueeze(2).expand(B, N_A, N_B, D)
        F_B_exp = F_B.unsqueeze(1).expand(B, N_A, N_B, D)

        # Pairwise feature components
        mult = F_A_exp * F_B_exp
        diff = torch.abs(F_A_exp - F_B_exp)

        # Concatenate into explicit 2048-D pair interaction feature
        z_ij = torch.cat([F_A_exp, F_B_exp, mult, diff], dim=-1)
        assert z_ij.shape == (B, N_A, N_B, 4 * D), (
            f"Expected z_ij shape ({B}, {N_A}, {N_B}, {4 * D}), got {z_ij.shape}"
        )

        # Pass through interaction MLP -> (B, N_A, N_B, 512)
        pair_repr = self.interaction_mlp(z_ij)

        # Outer product of valid fragment masks: (B, N_A, N_B, 1)
        mask_A = mask_A.to(device)
        mask_B = mask_B.to(device)
        pair_mask = (mask_A.unsqueeze(2) * mask_B.unsqueeze(1)).unsqueeze(-1)

        # Zero out padding pairs
        pair_repr = pair_repr * pair_mask

        # Permutation-invariant masked mean pooling -> r_AB in R^512
        valid_pair_counts = torch.clamp(pair_mask.sum(dim=(1, 2)), min=1.0)  # (B, 1)
        r_AB = pair_repr.sum(dim=(1, 2)) / valid_pair_counts  # (B, 512)
        assert r_AB.shape == (B, self.fragment_dim), (
            f"Expected r_AB shape ({B}, {self.fragment_dim}), got {r_AB.shape}"
        )

        diagnostics = {
            "norm_r_AB": torch.norm(r_AB, dim=-1).mean().item(),
            "mean_valid_pairs": valid_pair_counts.mean().item(),
        }

        return r_AB, diagnostics
