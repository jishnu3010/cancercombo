"""Cell-conditioned Feature-wise Linear Modulation (FiLM) for fragment embeddings."""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn


class CellConditionedFiLM(nn.Module):
    """Applies cell-conditioned FiLM modulation to fragment embeddings.

    F_tilde = (1 + gamma(c)) * F + beta(c)

    The same FiLM generator is applied to both Drug A and Drug B fragments.
    Uses identity-preserving zero-initialization on the output layer.
    """

    def __init__(
        self,
        cell_dim: int = 512,
        fragment_dim: int = 512,
        hidden_dim: int = 512,
        film_type: str = "standard",
    ):
        super().__init__()
        self.cell_dim = cell_dim
        self.fragment_dim = fragment_dim
        self.hidden_dim = hidden_dim
        self.film_type = film_type

        # MLP generating gamma and beta from cell embedding c
        self.net = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * fragment_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-initialize the final layer so gamma=0, beta=0 initially -> identity transform
        last_linear = self.net[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        F: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Modulate fragments F with cell embedding c.

        Args:
            F: Fragment embeddings of shape (B, N, 512).
            c: Cell embedding of shape (B, 512).
            mask: Fragment validity mask of shape (B, N), 1 for valid, 0 for pad.

        Returns:
            F_tilde: Modulated fragment embeddings of shape (B, N, 512).
            diagnostics: Dictionary containing modulation statistics.
        """
        assert F.shape[-1] == self.fragment_dim, (
            f"Expected fragment dim {self.fragment_dim}, got {F.shape[-1]}"
        )
        assert c.shape[-1] == self.cell_dim, (
            f"Expected cell dim {self.cell_dim}, got {c.shape[-1]}"
        )

        B, N, D = F.shape
        params = self.net(c)  # (B, 1024)
        gamma = params[:, :self.fragment_dim].unsqueeze(1)  # (B, 1, 512)
        beta = params[:, self.fragment_dim:].unsqueeze(1)   # (B, 1, 512)

        if self.film_type == "standard":
            # (1 + gamma) * F + beta
            F_tilde = (1.0 + gamma) * F + beta
        else:
            # gamma * F + beta
            F_tilde = gamma * F + beta

        # Enforce zero vector on padded fragments
        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
        F_tilde = F_tilde * mask_expanded

        # Compute diagnostics
        with torch.no_grad():
            gamma_flat = gamma.squeeze(1)
            beta_flat = beta.squeeze(1)
            diff = (F_tilde - F) * mask_expanded
            diagnostics = {
                "gamma_mean": gamma_flat.mean().item(),
                "gamma_std": gamma_flat.std().item(),
                "gamma_min": gamma_flat.min().item(),
                "gamma_max": gamma_flat.max().item(),
                "beta_mean": beta_flat.mean().item(),
                "beta_std": beta_flat.std().item(),
                "beta_min": beta_flat.min().item(),
                "beta_max": beta_flat.max().item(),
                "norm_F": torch.norm(F * mask_expanded, dim=-1).mean().item(),
                "norm_F_tilde": torch.norm(F_tilde, dim=-1).mean().item(),
                "norm_diff": torch.norm(diff, dim=-1).mean().item(),
            }

        return F_tilde, diagnostics
