"""
Parameter Heads Module for CancerCombo-BRICS-Symmetric.
Predicts raw unconstrained parameter logits for Bivariate Hill / SynBa dose-response model.

Uses shared single-drug parameter heads for Drug A and Drug B representations to preserve exact
permutation symmetry under drug input swapping.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn


class ParameterHeads(nn.Module):
    """
    MLP heads predicting raw unconstrained logits for Bivariate Hill / SynBa surface parameters:
        - Single-drug parameters (c1/c2, h1/h2, e1/e2): predicted using a shared single-drug head
          applied to Drug A and Drug B representations respectively.
        - Combination parameters (e0, e12, alpha): predicted from symmetric fused representation r_final.

    Args:
        d_dim: Fragment embedding dimension (default: 128).
        cell_dim: Cell feature representation dimension (default: 512).
        hidden_dim: Hidden MLP dimension (default: 256).
    """

    def __init__(
        self,
        d_dim: int = 128,
        cell_dim: int = 512,
        hidden_dim: int = 256,
        in_dim: Optional[int] = None
    ):
        super().__init__()
        self.d_dim = d_dim
        self.cell_dim = cell_dim

        single_drug_in_dim = 2 * d_dim + cell_dim
        combo_in_dim = in_dim if in_dim is not None else (4 * d_dim + cell_dim)
        self.combo_in_dim = combo_in_dim

        # Shared feature trunk & heads for single-drug parameters (c, h, e)
        self.single_drug_trunk = nn.Sequential(
            nn.Linear(single_drug_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        self.head_c = nn.Linear(hidden_dim, 1)
        self.head_h = nn.Linear(hidden_dim, 1)
        self.head_e = nn.Linear(hidden_dim, 1)

        # Feature trunk & heads for combination parameters (e0, e12, alpha)
        self.combo_trunk = nn.Sequential(
            nn.Linear(combo_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        self.head_e0 = nn.Linear(hidden_dim, 1)
        self.head_e12 = nn.Linear(hidden_dim, 1)
        self.head_alpha = nn.Linear(hidden_dim, 1)

        # Initializations for fast convergence
        nn.init.constant_(self.head_e0.bias, 2.0)
        nn.init.constant_(self.head_e.bias, -1.0)
        nn.init.constant_(self.head_e12.bias, -2.0)
        nn.init.constant_(self.head_c.bias, 0.0)
        nn.init.constant_(self.head_h.bias, 0.5)
        nn.init.constant_(self.head_alpha.bias, 0.5)

    def forward(
        self,
        mu_A_from_B: torch.Tensor,
        p_A_from_B: Optional[torch.Tensor] = None,
        mu_B_from_A: Optional[torch.Tensor] = None,
        p_B_from_A: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        r_final: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for parameter heads.

        Supports calling with explicit directional representations:
            forward(mu_A_from_B, p_A_from_B, mu_B_from_A, p_B_from_A, c, r_final)

        Or fallback calling with r_final alone:
            forward(r_final)
        """
        if p_A_from_B is None and mu_B_from_A is None:
            # Fallback for single tensor call forward(r_final)
            r_final_tensor = mu_A_from_B
            B = r_final_tensor.size(0)
            feat_combo = self.combo_trunk(r_final_tensor)

            # Split r_final into r_AB_sym and c
            c_part = r_final_tensor[:, -self.cell_dim:]
            r_AB_sym = r_final_tensor[:, :-self.cell_dim]

            # Slice r_AB_sym into mean_sym, mean_diff, max_sym, max_diff
            mean_sym = r_AB_sym[:, :self.d_dim]
            mean_diff = r_AB_sym[:, self.d_dim:2*self.d_dim]
            max_sym = r_AB_sym[:, 2*self.d_dim:3*self.d_dim]
            max_diff = r_AB_sym[:, 3*self.d_dim:]

            # Reconstruct directional statistics (approximate)
            mu_A_from_B = mean_sym + mean_diff / 2.0
            mu_B_from_A = mean_sym - mean_diff / 2.0
            p_A_from_B = max_sym + max_diff / 2.0
            p_B_from_A = max_sym - max_diff / 2.0
            c = c_part
            r_final = r_final_tensor

        B = c.size(0)

        # Single-drug feature representation for Drug A
        r_A = torch.cat([mu_A_from_B, p_A_from_B, c], dim=-1)
        feat_A = self.single_drug_trunk(r_A)
        raw_c1 = self.head_c(feat_A)
        raw_h1 = self.head_h(feat_A)
        raw_e1 = self.head_e(feat_A)

        # Single-drug feature representation for Drug B (using same shared single_drug_trunk and heads)
        r_B = torch.cat([mu_B_from_A, p_B_from_A, c], dim=-1)
        feat_B = self.single_drug_trunk(r_B)
        raw_c2 = self.head_c(feat_B)
        raw_h2 = self.head_h(feat_B)
        raw_e2 = self.head_e(feat_B)

        # Combination feature representation for (e0, e12, alpha)
        feat_combo = self.combo_trunk(r_final)
        raw_e0 = self.head_e0(feat_combo)
        raw_e12 = self.head_e12(feat_combo)
        raw_alpha = self.head_alpha(feat_combo)

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
