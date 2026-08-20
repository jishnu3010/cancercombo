"""
Unified Parameter Head Module for CancerCombo-BRICS.
Predicts raw unconstrained parameter logits for all 9 dose-response parameters directly
from the concatenated representation r_final in R^1024 using ONE unified MLP.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn


class ParameterHeads(nn.Module):
    """
    Unified MLP predicting raw unconstrained logits for all 9 Bivariate Hill / SynBa surface parameters:
        r_final (1024-dim) -> parameter_mlp -> 9 raw scalar outputs

    Output Order:
        0 -> e0
        1 -> e1
        2 -> e2
        3 -> e12
        4 -> c1
        5 -> c2
        6 -> h1
        7 -> h2
        8 -> alpha

    Args:
        d_dim: Fragment embedding dimension (default: 128).
        cell_dim: Cell feature representation dimension (default: 512).
        hidden_dim: Hidden dimension for first layer (default: 512).
        in_dim: Input representation dimension (default: 1024).
        num_params: Number of dose-response parameters (default: 9).
    """

    def __init__(
        self,
        d_dim: int = 128,
        cell_dim: int = 512,
        hidden_dim: int = 512,
        in_dim: Optional[int] = None,
        num_params: int = 9
    ):
        super().__init__()
        self.d_dim = d_dim
        self.cell_dim = cell_dim

        input_dim = in_dim if in_dim is not None else (4 * d_dim + cell_dim)
        self.input_dim = input_dim

        # ONE unified Parameter MLP: 1024 -> 512 -> 256 -> 9
        self.parameter_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),

            nn.Linear(256, num_params)
        )

        # Output linear layer bias initialization for stable convergence
        final_linear = self.parameter_mlp[-1]
        with torch.no_grad():
            # 0: e0 -> 2.0
            final_linear.bias[0] = 2.0
            # 1: e1 -> -1.0
            final_linear.bias[1] = -1.0
            # 2: e2 -> -1.0
            final_linear.bias[2] = -1.0
            # 3: e12 -> -2.0
            final_linear.bias[3] = -2.0
            # 4: c1 -> 0.0
            final_linear.bias[4] = 0.0
            # 5: c2 -> 0.0
            final_linear.bias[5] = 0.0
            # 6: h1 -> 0.5
            final_linear.bias[6] = 0.5
            # 7: h2 -> 0.5
            final_linear.bias[7] = 0.5
            # 8: alpha -> 0.5
            final_linear.bias[8] = 0.5

    def forward(
        self,
        r_final: torch.Tensor,
        *args,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for unified parameter MLP.

        Args:
            r_final: Tensor of shape (B, 1024) [or (B, in_dim)].

        Returns:
            raw_params: Dictionary mapping parameter names to [B, 1] raw logit tensors.
        """
        assert r_final.dim() == 2, f"Expected 2D tensor for r_final, got shape {tuple(r_final.shape)}"
        B = r_final.size(0)

        # Pass r_final through ONE unified parameter MLP: outputs [B, 9]
        raw = self.parameter_mlp(r_final)

        # Slice 9 outputs according to explicit output order
        raw_e0    = raw[:, 0:1]
        raw_e1    = raw[:, 1:2]
        raw_e2    = raw[:, 2:3]
        raw_e12   = raw[:, 3:4]
        raw_c1    = raw[:, 4:5]
        raw_c2    = raw[:, 5:6]
        raw_h1    = raw[:, 6:7]
        raw_h2    = raw[:, 7:8]
        raw_alpha = raw[:, 8:9]

        raw_params = {
            "e0": raw_e0,
            "e1": raw_e1,
            "e2": raw_e2,
            "e12": raw_e12,
            "c1": raw_c1,
            "c2": raw_c2,
            "h1": raw_h1,
            "h2": raw_h2,
            "alpha": raw_alpha
        }

        for k, v in raw_params.items():
            assert v.shape == (B, 1), f"Raw parameter '{k}' shape mismatch: expected ({B}, 1), got {tuple(v.shape)}"

        return raw_params
