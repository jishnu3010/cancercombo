"""Drug-Cell interaction module constructing z_DC (2048-D) and gated r_DC (1536-D)."""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn


class DrugCellInteraction(nn.Module):
    """Constructs explicit Drug-Cell interaction representation.

    z_DC = [r'_AB, c, r'_AB * c, |r'_AB - c|] in R^2048
    g_DC = sigmoid(MLP(z_DC)) in R^512
    r_gate = g_DC * r'_AB in R^512
    r_DC = [r'_AB ; c ; r_gate] in R^1536
    """

    def __init__(
        self,
        dim: int = 512,
        mlp_hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.mlp_hidden = mlp_hidden

        # MLP for gating: 2048 -> mlp_hidden -> 512
        # Architecture: Linear -> ReLU -> Linear -> Sigmoid
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, dim),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.gate_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        r_prime_AB: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Compute Drug-Cell interaction.

        Args:
            r_prime_AB: Drug-pair projected embedding of shape (B, 512).
            c: Cell-line embedding of shape (B, 512).

        Returns:
            r_DC: Final Drug-Cell representation of shape (B, 1536).
            r_gate: Gated drug representation of shape (B, 512).
            diagnostics: Dictionary containing gate statistics.
        """
        B = r_prime_AB.shape[0]
        assert r_prime_AB.shape == (B, self.dim), (
            f"Expected r'_AB shape ({B}, {self.dim}), got {r_prime_AB.shape}"
        )
        assert c.shape == (B, self.dim), (
            f"Expected c shape ({B}, {self.dim}), got {c.shape}"
        )

        # 1. Construct z_DC in R^2048
        mult = r_prime_AB * c
        diff = torch.abs(r_prime_AB - c)
        z_DC = torch.cat([r_prime_AB, c, mult, diff], dim=-1)
        assert z_DC.shape == (B, 4 * self.dim), (
            f"Expected z_DC shape ({B}, {4 * self.dim}), got {z_DC.shape}"
        )

        # 2. Sigmoid MLP gate
        g_DC = self.gate_mlp(z_DC)
        assert g_DC.shape == (B, self.dim), (
            f"Expected g_DC shape ({B}, {self.dim}), got {g_DC.shape}"
        )

        # 3. Gated drug representation
        r_gate = g_DC * r_prime_AB
        assert r_gate.shape == (B, self.dim), (
            f"Expected r_gate shape ({B}, {self.dim}), got {r_gate.shape}"
        )

        # 4. Final representation r_DC in R^1536
        r_DC = torch.cat([r_prime_AB, c, r_gate], dim=-1)
        assert r_DC.shape == (B, 3 * self.dim), (
            f"Expected r_DC shape ({B}, {3 * self.dim}), got {r_DC.shape}"
        )

        with torch.no_grad():
            diagnostics = {
                "gate_mean": g_DC.mean().item(),
                "gate_std": g_DC.std().item(),
                "gate_min": g_DC.min().item(),
                "gate_max": g_DC.max().item(),
                "norm_r_gate": torch.norm(r_gate, dim=-1).mean().item(),
                "norm_r_DC": torch.norm(r_DC, dim=-1).mean().item(),
            }

        return r_DC, r_gate, diagnostics
