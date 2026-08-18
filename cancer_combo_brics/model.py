"""
Top-level CancerCombo-BRICS-Symmetric Model for Predicting 2D Dose-Response Viability Surfaces.

Integrates 9 modular pipeline stages:
    1. CellLineEncoder (Gene expression 976 -> 512 c vector)
    2. Shared FragmentEncoder (BRICS fragment descriptors -> d vector) + Padding & Masking
    3. Shared FiLMConditioning (Cell-conditioned fragment modulation F_tilde_A, F_tilde_B)
    4. MaskedBidirectionalCrossAttention (Manual MHA cross-attention + key/query masking)
    5. SymmetricFusion (Order-invariant A/B feature fusion -> 4d r_AB_sym)
    6. CellFusion (r_final = concat(r_AB_sym, c) -> 4d + 512)
    7. ParameterHeads (MLP heads predicting raw Bivariate Hill / SynBa logits with shared single-drug heads)
    8. ConstraintTransform (Softplus/Sigmoid constraints on parameters)
    9. BivariateHillSolver (Closed-form, differentiable 2D viability surface Y computation)
"""

from typing import Dict, Union, Tuple, List, Optional
import torch
import torch.nn as nn

from .cell_line_encoder import CellLineEncoder
from .fragment_encoder import FragmentEncoder
from .film_conditioning import FiLMConditioning
from .cross_attention import MaskedBidirectionalCrossAttention, BidirectionalCrossAttention
from .symmetric_fusion import SymmetricFusion
from .cell_fusion import CellFusion
from .parameter_heads import ParameterHeads
from .constraint_transform import ConstraintTransform
from .bivariate_hill_solver import BivariateHillSolver
from .brics_utils import collate_brics_fragments


class CancerComboBRICSSymmetric(nn.Module):
    """
    CancerCombo-BRICS-Symmetric PyTorch neural network model.

    Predicts full 2D dose-response viability surfaces for drug combinations with mathematical
    drug-order invariance.

    Args:
        gene_dim: Dimension of landmark gene expression input (default: 976).
        cell_dim: Cell feature representation dimension c (default: 512).
        frag_fp_dim: Dimension of fragment descriptor input (default: Morgan FP 2048).
        d_dim: Fragment embedding dimension d (default: 128).
        num_attn_heads: Number of attention heads for cross-attention (default: 4).
        dropout_rate: Dropout probability in encoders (default: 0.2).
        shared_attn_weights: Whether to share cross-attention weights across directions (default: True).
    """

    def __init__(
        self,
        gene_dim: int = 976,
        cell_dim: int = 512,
        frag_fp_dim: int = 2048,
        d_dim: int = 128,
        num_attn_heads: int = 4,
        dropout_rate: float = 0.2,
        shared_attn_weights: bool = True
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.cell_dim = cell_dim
        self.frag_fp_dim = frag_fp_dim
        self.d_dim = d_dim
        self.num_attn_heads = num_attn_heads

        # 1. Cell-Line Encoder
        self.cell_encoder = CellLineEncoder(
            in_dim=gene_dim,
            hidden_dim=cell_dim,
            dropout_rate=dropout_rate
        )

        # 2. Shared Fragment Encoder (applied to Drug A and Drug B)
        self.fragment_encoder = FragmentEncoder(
            in_bits=frag_fp_dim,
            d_dim=d_dim
        )

        # 3. Cell Conditioning (FiLM) - single shared module applied to both drugs
        self.film = FiLMConditioning(cell_dim=cell_dim, d_dim=d_dim)

        # 4. Masked Bidirectional Cross-Attention
        self.cross_attention = MaskedBidirectionalCrossAttention(
            d_dim=d_dim,
            num_heads=num_attn_heads,
            shared_weights=shared_attn_weights
        )

        # 5. Symmetric A/B Fusion
        self.symmetric_fusion = SymmetricFusion(d_dim=d_dim)

        # 6. Cell Fusion
        self.cell_fusion = CellFusion(
            d_dim=d_dim,
            cell_dim=cell_dim
        )

        # 7. Parameter Heads with Shared Single-Drug Trunk
        fusion_dim = 4 * d_dim + cell_dim
        self.parameter_heads = ParameterHeads(
            d_dim=d_dim,
            cell_dim=cell_dim,
            hidden_dim=256,
            in_dim=fusion_dim
        )

        # 8. Constraint / Transform Stage
        self.constraint_transform = ConstraintTransform()

        # 9. Differentiable Bivariate Hill Surface Solver
        self.bivariate_solver = BivariateHillSolver()

    def forward(
        self,
        cell_expr: torch.Tensor,
        drugA_frags: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[str]],
        drugA_mask: Optional[Union[torch.Tensor, List[str]]] = None,
        drugB_frags: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[str]]] = None,
        drugB_mask: Optional[torch.Tensor] = None,
        dose_grid: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        return_params: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for CancerCombo-BRICS-Symmetric.

        Supports standard signature:
            forward(cell_expr, drugA_frags, drugA_mask, drugB_frags, drugB_mask, dose_grid)

        Also supports high-level signature:
            forward(cell_expr, smiles_A, smiles_B, dose_grid)
        """
        device = cell_expr.device
        B = cell_expr.size(0)

        # Parse inputs to resolve positional or keyword usage cleanly
        fp_A, mask_A, fp_B, mask_B, grid = self._parse_inputs(
            cell_expr=cell_expr,
            drugA_frags=drugA_frags,
            drugA_mask=drugA_mask,
            drugB_frags=drugB_frags,
            drugB_mask=drugB_mask,
            dose_grid=dose_grid,
            device=device
        )

        # STAGE 1: Cell-Line Encoder
        # Output c: (B, 512)
        c = self.cell_encoder(cell_expr)
        assert c.shape == (B, self.cell_dim), f"Stage 1 cell vector shape mismatch: got {tuple(c.shape)}"

        # STAGE 2: Shared Fragment Encoder
        # Input fp_A: (B, n, fp_dim) -> Output F_A: (B, n, d)
        # Input fp_B: (B, m, fp_dim) -> Output F_B: (B, m, d)
        F_A = self.fragment_encoder(fp_A)
        F_B = self.fragment_encoder(fp_B)
        n, m = F_A.size(1), F_B.size(1)
        assert F_A.shape == (B, n, self.d_dim), f"Stage 2 F_A shape mismatch: got {tuple(F_A.shape)}"
        assert F_B.shape == (B, m, self.d_dim), f"Stage 2 F_B shape mismatch: got {tuple(F_B.shape)}"

        # STAGE 3: Cell Conditioning Module (FiLM-style with shared gamma/beta functions)
        # Input F_A, F_B and c -> Output F_tilde_A: (B, n, d), F_tilde_B: (B, m, d)
        F_tilde_A = self.film(F_A, c)
        F_tilde_B = self.film(F_B, c)
        assert F_tilde_A.shape == (B, n, self.d_dim), f"Stage 3 F_tilde_A shape mismatch: got {tuple(F_tilde_A.shape)}"
        assert F_tilde_B.shape == (B, m, self.d_dim), f"Stage 3 F_tilde_B shape mismatch: got {tuple(F_tilde_B.shape)}"

        # STAGE 4: Masked Bidirectional Cross-Attention
        # Inputs: F_tilde_A, mask_A, F_tilde_B, mask_B
        # Output: mu_A_from_B, p_A_from_B, mu_B_from_A, p_B_from_A (each shape B, d)
        mu_A_from_B, p_A_from_B, mu_B_from_A, p_B_from_A = self.cross_attention(
            F_tilde_A=F_tilde_A,
            mask_A=mask_A,
            F_tilde_B=F_tilde_B,
            mask_B=mask_B
        )

        # STAGE 5: Symmetric A/B Fusion
        # Pair corresponding statistics across directions and fuse with symmetric mean and abs diff
        # Output r_AB_sym: (B, 4d)
        r_AB_sym = self.symmetric_fusion(
            mu_A_from_B=mu_A_from_B,
            p_A_from_B=p_A_from_B,
            mu_B_from_A=mu_B_from_A,
            p_B_from_A=p_B_from_A
        )
        assert r_AB_sym.shape == (B, 4 * self.d_dim), f"Stage 5 r_AB_sym shape mismatch: got {tuple(r_AB_sym.shape)}"

        # STAGE 6: Cell Fusion
        # Input r_AB_sym and c -> Output r_final: (B, 4d + 512)
        r_final = self.cell_fusion(r_AB_sym, c)
        fusion_dim = 4 * self.d_dim + self.cell_dim
        assert r_final.shape == (B, fusion_dim), f"Stage 6 r_final shape mismatch: got {tuple(r_final.shape)}"

        # STAGE 7: Parameter Heads (Shared single-drug heads for c1/c2, h1/h2, e1/e2)
        raw_params = self.parameter_heads(
            mu_A_from_B=mu_A_from_B,
            p_A_from_B=p_A_from_B,
            mu_B_from_A=mu_B_from_A,
            p_B_from_A=p_B_from_A,
            c=c,
            r_final=r_final
        )

        # STAGE 8: Constraint / Transform Stage
        params = self.constraint_transform(raw_params)

        # STAGE 9: Differentiable Bivariate Hill Solver
        # Inputs: params, dose_grid -> Output Y_pred: (B, M, N)
        Y_pred = self.bivariate_solver(params, grid)

        if return_params:
            return Y_pred, params
        return Y_pred

    def _parse_inputs(
        self,
        cell_expr: torch.Tensor,
        drugA_frags: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[str]],
        drugA_mask: Optional[Union[torch.Tensor, List[str]]],
        drugB_frags: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[str]]],
        drugB_mask: Optional[torch.Tensor],
        dose_grid: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """Helper to parse positional/keyword inputs for SMILES or Tensor representations."""
        # Check if high-level SMILES arguments passed
        if isinstance(drugA_frags, (list, tuple)) and isinstance(drugA_frags[0], str):
            fp_A, mask_A, _ = collate_brics_fragments(drugA_frags, n_bits=self.frag_fp_dim, device=device)
            if isinstance(drugA_mask, (list, tuple)) and isinstance(drugA_mask[0], str):
                drugB_smiles = drugA_mask
                grid = drugB_frags if dose_grid is None else dose_grid
            else:
                drugB_smiles = drugB_frags
                grid = dose_grid
            fp_B, mask_B, _ = collate_brics_fragments(drugB_smiles, n_bits=self.frag_fp_dim, device=device)
            return fp_A, mask_A, fp_B, mask_B, grid

        if isinstance(drugA_frags, tuple) and len(drugA_frags) == 2 and isinstance(drugA_frags[0], torch.Tensor):
            fp_A, mask_A = drugA_frags[0], drugA_frags[1]
            if isinstance(drugA_mask, tuple) and len(drugA_mask) == 2 and isinstance(drugA_mask[0], torch.Tensor):
                fp_B, mask_B = drugA_mask[0], drugA_mask[1]
                grid = drugB_frags if dose_grid is None else dose_grid
            else:
                fp_B, mask_B = drugB_frags[0], drugB_frags[1] if isinstance(drugB_frags, tuple) else (drugB_frags, drugB_mask)
                grid = dose_grid
            return fp_A, mask_A, fp_B, mask_B, grid

        fp_A = drugA_frags
        mask_A = drugA_mask
        fp_B = drugB_frags
        mask_B = drugB_mask
        grid = dose_grid
        return fp_A, mask_A, fp_B, mask_B, grid


CancerComboBRICS = CancerComboBRICSSymmetric
