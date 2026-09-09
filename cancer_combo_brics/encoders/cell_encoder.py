"""Cell-line encoder mapping 976-D gene expression to 512-D cell embedding."""

from __future__ import annotations

import torch
import torch.nn as nn


class CellEncoder(nn.Module):
    """Encodes 976-dimensional landmark gene expression profiles into a 512-D embedding."""

    def __init__(
        self,
        in_dim: int = 976,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Gene expression tensor of shape (B, 976).

        Returns:
            Cell embedding c of shape (B, 512).
        """
        assert x.shape[-1] == self.in_dim, (
            f"Expected input dimension {self.in_dim}, but got {x.shape[-1]}"
        )
        c = self.net(x)
        assert c.shape[-1] == self.hidden_dim, (
            f"Expected output dimension {self.hidden_dim}, but got {c.shape[-1]}"
        )
        return c
