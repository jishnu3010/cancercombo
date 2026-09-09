"""CancerCombo neural model for complete 2D dose-response surface prediction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from cancer_combo_brics.config import ModelConfig
from cancer_combo_brics.encoders.cell_encoder import CellEncoder
from cancer_combo_brics.encoders.fragment_encoder import FragmentEncoder
from cancer_combo_brics.interaction.pairwise_interaction import ExplicitPairwiseFragmentInteraction
from cancer_combo_brics.interaction.drug_cell import DrugCellInteraction
from cancer_combo_brics.pharmacology.parameter_heads import PharmacologicalParameterHeads
from cancer_combo_brics.pharmacology.constraints import ConstraintTransform
from cancer_combo_brics.pharmacology.bivariate_hill import BivariateHillSolver
from cancer_combo_brics.pharmacology.dose_bias import DoseDependentBias


class CancerComboBRICS(nn.Module):
    """CancerCombo neural model for complete 2D dose-response surface prediction.

    Target Architecture Flow:
      1. Cell-line expression (976-D) -> CellEncoder -> c (512-D)
      2. Drug A, B functional-group fragments -> Mol2Vec + Projection -> F_A (N x 512), F_B (M x 512)
      3. Explicit Pairwise Fragment Interaction -> r_AB (512-D)
      4. Drug-Cell Interaction: z_DC (2048-D) -> Sigmoid Gate -> r_gate (512-D) -> r_DC (1536-D)
      5. 8 Pharmacological parameter heads
      6. Constraint transformation (e0 = 100.0)
      7. Vectorized Bivariate Hill Solver -> Y_hill (B x 4 x 4)
      8. Dose-dependent bias -> Bias (B x 4 x 4)
      9. Final surface -> Y_pred = Y_hill + Bias
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        # 1. Cell Encoder (976 -> 512)
        self.cell_encoder = CellEncoder(
            in_dim=self.config.cell_dim,
            hidden_dim=self.config.cell_hidden_dim,
        )

        # 2. Shared Fragment Encoder (Mol2Vec -> 512-D)
        self.fragment_encoder = FragmentEncoder(
            native_dim=getattr(self.config, "mol2vec_native_dim", 300),
            fragment_dim=self.config.fragment_dim,
            dropout=0.1,
        )

        # 3. Explicit Pairwise Fragment Interaction (512-D -> r_AB in R^512)
        self.pairwise_interaction = ExplicitPairwiseFragmentInteraction(
            fragment_dim=self.config.fragment_dim,
            hidden_dim=getattr(self.config, "interaction_hidden_dim", 512),
            dropout=0.1,
        )

        # 4. Explicit Drug-Cell Interaction (512 + 512 -> r_DC in R^1536)
        self.drug_cell = DrugCellInteraction(
            dim=self.config.fragment_dim,
            mlp_hidden=self.config.drug_cell_mlp_hidden,
        )

        # 5. 8 Pharmacological Parameter Heads (matching finalcheck DeepSynBa)
        self.parameter_heads = PharmacologicalParameterHeads(
            in_dim=3 * self.config.fragment_dim,  # 1536
            emb_size=getattr(self.config, "emb_size", 1024),
            dropout=self.config.param_dropout,
        )

        # 6. Constraint Transform (matching finalcheck e0=100.0)
        self.constraint_transform = ConstraintTransform(
            e0=getattr(self.config, "hill_e0", 100.0),
        )

        # 7. Bivariate Hill Solver (matching finalcheck SynBa Log-Sum-Exp e0=100.0)
        self.hill_solver = BivariateHillSolver(
            e0=getattr(self.config, "hill_e0", 100.0),
        )

        # 8. Dose-Dependent Bias (matching finalcheck dual predictors)
        self.dose_bias = DoseDependentBias(
            context_dim=3 * self.config.fragment_dim,  # 1536
            emb_size=getattr(self.config, "emb_size", 1024),
            dropout=self.config.param_dropout,
            enabled=self.config.enable_bias,
        )

    def forward(
        self,
        cell_expr: torch.Tensor,
        fragments_A: List[List[str]],
        mask_A: torch.Tensor,
        fragments_B: List[List[str]],
        mask_B: torch.Tensor,
        doses_A: torch.Tensor,
        doses_B: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Full forward pass of target CancerCombo architecture.

        Args:
            cell_expr: Landmark gene expression tensor of shape (B, 976).
            fragments_A: List of fragment strings for Drug A across batch.
            mask_A: Validity mask for Drug A fragments of shape (B, N).
            fragments_B: List of fragment strings for Drug B across batch.
            mask_B: Validity mask for Drug B fragments of shape (B, M).
            doses_A: Dose grid for Drug A (D_A,) or (B, D_A).
            doses_B: Dose grid for Drug B (D_B,) or (B, D_B).
            return_diagnostics: If True, returns detailed diagnostic dictionary.

        Returns:
            Y_pred: Predicted 2D viability surface of shape (B, D_A, D_B).
            diagnostics: Optional dictionary of intermediate representations and stats.
        """
        B = cell_expr.shape[0]
        device = cell_expr.device

        # 1. Encode cell expression -> c in R^512
        c = self.cell_encoder(cell_expr)
        assert c.shape == (B, 512), f"c expected (B, 512), got {c.shape}"

        # 2. Encode Drug A and Drug B fragments -> F_A (B, N, 512), F_B (B, M, 512)
        F_A = self.fragment_encoder.encode_drug_fragments_batch(fragments_A, mask_A, device=device)
        F_B = self.fragment_encoder.encode_drug_fragments_batch(fragments_B, mask_B, device=device)

        # 3. Explicit pairwise fragment interaction -> r_AB in R^512
        r_AB, diag_interaction = self.pairwise_interaction(F_A, F_B, mask_A, mask_B)
        assert r_AB.shape == (B, 512), f"r_AB expected (B, 512), got {r_AB.shape}"

        # 4. Explicit Drug-Cell Interaction
        # z_DC in R^2048, g_DC in R^512, r_DC in R^1536
        r_DC, r_gate, diag_dc = self.drug_cell(r_AB, c)
        assert r_DC.shape == (B, 1536), f"r_DC expected (B, 1536), got {r_DC.shape}"

        # 5. 8 Pharmacological parameter heads
        raw_params = self.parameter_heads(r_DC)

        # 6. Constraint transformation
        constrained_params, diag_params = self.constraint_transform(raw_params)
        e1 = constrained_params["e1"]
        e2 = constrained_params["e2"]
        e3 = constrained_params["e3"]
        logc1 = constrained_params["logc1"]
        logc2 = constrained_params["logc2"]
        h1 = constrained_params["h1"]
        h2 = constrained_params["h2"]
        alpha = constrained_params["alpha"]

        # 7. Bivariate Hill Solver matching finalcheck SynBa formulation
        Y_hill = self.hill_solver(doses_A, doses_B, e1, e2, e3, logc1, logc2, h1, h2, alpha)

        # 8. Dose-dependent bias
        bias = self.dose_bias(r_DC, doses_A, doses_B)

        # 9. Complete 2D dose-response surface
        Y_pred = Y_hill + bias

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "c": c,
                "F_A": F_A,
                "F_B": F_B,
                "r_AB": r_AB,
                "r_DC": r_DC,
                "r_gate": r_gate,
                "raw_params": raw_params,
                "constrained_params": constrained_params,
                "params_tuple": (e1, e2, e3, logc1, logc2, h1, h2, alpha),
                "Y_hill": Y_hill,
                "bias": bias,
                "Y_pred": Y_pred,
                "diag_interaction": diag_interaction,
                "diag_dc": diag_dc,
                "diag_params": diag_params,
            }

        return Y_pred, diagnostics
