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
from .parameter_heads import ParameterHeads
from .constraint_transform import ConstraintTransform
from .bivariate_hill_solver import BivariateHillSolver
from .brics_utils import collate_brics_fragments, print_batch_drug_fragments, print_drug_fragments


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
        print_fragments: Flag to toggle debug printing of original SMILES & BRICS fragments (default: False).
    """

    def __init__(
        self,
        gene_dim: int = 976,
        cell_dim: int = 512,
        frag_fp_dim: int = 2048,
        d_dim: int = 128,
        num_attn_heads: int = 4,
        dropout_rate: float = 0.2,
        shared_attn_weights: bool = True,
        print_fragments: bool = False
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.cell_dim = cell_dim
        self.frag_fp_dim = frag_fp_dim
        self.d_dim = d_dim
        self.num_attn_heads = num_attn_heads
        self.print_fragments = print_fragments

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

        # Final Fragment LayerNorm (applied immediately before FiLM conditioning)
        self.fragment_norm = nn.LayerNorm(d_dim)

        # 3. Cell Conditioning (FiLM) - single shared module applied to both drugs
        self.film = FiLMConditioning(cell_dim=cell_dim, d_dim=d_dim)

        # 4. Masked Bidirectional Cross-Attention
        self.cross_attention = MaskedBidirectionalCrossAttention(
            d_dim=d_dim,
            num_heads=num_attn_heads,
            shared_weights=shared_attn_weights
        )

        # 5. Parameter Heads (ONE Unified 9-Output Parameter MLP: 1024 -> 512 -> 256 -> 9)
        fusion_dim = 4 * d_dim + cell_dim
        self.parameter_heads = ParameterHeads(
            d_dim=d_dim,
            cell_dim=cell_dim,
            hidden_dim=512,
            in_dim=fusion_dim,
            num_params=9
        )

        # 6. Constraint / Transform Stage
        self.constraint_transform = ConstraintTransform()

        # 7. Differentiable Bivariate Hill Surface Solver
        self.bivariate_solver = BivariateHillSolver()

    def _encode_and_condition_unpadded_fragments(
        self,
        fp_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        STAGE 1 & STAGE 2 UNPADDED EXECUTION FLOW:
            1. Extract UNPADDED real fragment fingerprints for each sample in batch.
            2. Pass real fragment fingerprints through shared FragmentEncoder -> (N_real_i, d).
            3. Pass real fragment embeddings through shared Fragment LayerNorm -> (N_real_i, d).
            4. Pass real fragment embeddings through shared Standard E2 FiLM -> (N_real_i, d).
            5. Pad conditioned fragment representations AFTER FiLM to max batch sequence length.
            6. Construct explicit boolean padding mask (B, n_max).

        Ensures FragmentEncoder, LayerNorm, and FiLM NEVER receive padded zero rows!
        """
        B = fp_tensor.size(0)
        device = fp_tensor.device

        unpadded_conditioned_list: List[torch.Tensor] = []
        max_len = 0

        for i in range(B):
            # Extract real unpadded fragment fingerprints for sample i: shape (N_real_i, 2048)
            real_mask_i = mask_tensor[i]
            real_fps_i = fp_tensor[i, real_mask_i]  # (N_real_i, 2048)

            if real_fps_i.size(0) == 0:
                # Fallback for 0-fragment edge case: single zero fingerprint
                real_fps_i = torch.zeros(1, fp_tensor.size(-1), device=device)

            # STAGE 1 (UNPADDED): FragmentEncoder -> LayerNorm -> Standard E2 FiLM
            F_real_i = self.fragment_encoder(real_fps_i)                      # (N_real_i, d)
            F_norm_i = self.fragment_norm(F_real_i)                           # (N_real_i, d)
            F_tilde_i = self.film(F_norm_i.unsqueeze(0), c[i:i+1]).squeeze(0)  # (N_real_i, d)

            unpadded_conditioned_list.append(F_tilde_i)
            if F_tilde_i.size(0) > max_len:
                max_len = F_tilde_i.size(0)

        max_len = max(max_len, fp_tensor.size(1))

        # STAGE 2 (PADDING AFTER FILM): Pad to max batch sequence length and construct boolean mask
        F_tilde_padded = torch.zeros(B, max_len, self.d_dim, device=device)
        explicit_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

        for i in range(B):
            n_real_i = unpadded_conditioned_list[i].size(0)
            F_tilde_padded[i, :n_real_i] = unpadded_conditioned_list[i]
            explicit_mask[i, :n_real_i] = True

        return F_tilde_padded, explicit_mask

    def _print_debug_fragment_embeddings(
        self,
        F_A: torch.Tensor,
        mask_A: torch.Tensor,
        F_B: torch.Tensor,
        mask_B: torch.Tensor,
        step: int
    ):
        """
        Debug hook for printing raw fragment embeddings F_A and F_B (Stage 2 output).
        Uses detached CPU tensors to ensure no effect on the autograd graph or GPU performance.
        """
        with torch.no_grad():
            FA_cpu = F_A.detach().cpu()
            FB_cpu = F_B.detach().cpu()
            mA_cpu = mask_A.detach().cpu()
            mB_cpu = mask_B.detach().cpu()

            B = FA_cpu.shape[0]

            # Sample index 0 valid fragment masks
            valid_mask_A = mA_cpu[0]
            valid_mask_B = mB_cpu[0]
            n_valid_A = int(valid_mask_A.sum().item())
            n_valid_B = int(valid_mask_B.sum().item())

            # Calculate summary stats over valid (non-padded) entries for sample 0
            if n_valid_A > 0:
                FA_valid = FA_cpu[0, :n_valid_A]
                mean_A, std_A = float(FA_valid.mean()), float(FA_valid.std())
                min_A, max_A = float(FA_valid.min()), float(FA_valid.max())
            else:
                mean_A, std_A, min_A, max_A = 0.0, 0.0, 0.0, 0.0

            if n_valid_B > 0:
                FB_valid = FB_cpu[0, :n_valid_B]
                mean_B, std_B = float(FB_valid.mean()), float(FB_valid.std())
                min_B, max_B = float(FB_valid.min()), float(FB_valid.max())
            else:
                mean_B, std_B, min_B, max_B = 0.0, 0.0, 0.0, 0.0

            # First 3 fragments, first 5 dimensions for sample 0
            slice_A = FA_cpu[0, :min(3, n_valid_A if n_valid_A > 0 else 1), :min(5, FA_cpu.shape[-1])].numpy().round(4).tolist()
            slice_B = FB_cpu[0, :min(3, n_valid_B if n_valid_B > 0 else 1), :min(5, FB_cpu.shape[-1])].numpy().round(4).tolist()

            print(f"[step {step}] F_A shape=(B={B}, n={FA_cpu.shape[1]}, d={FA_cpu.shape[2]}) valid_n={n_valid_A} mean={mean_A:.4f} std={std_A:.4f} min={min_A:.4f} max={max_A:.4f}")
            print(f"[step {step}] F_A[0, :3, :5] slice={slice_A}")
            print(f"[step {step}] F_B shape=(B={B}, m={FB_cpu.shape[1]}, d={FB_cpu.shape[2]}) valid_m={n_valid_B} mean={mean_B:.4f} std={std_B:.4f} min={min_B:.4f} max={max_B:.4f}")
            print(f"[step {step}] F_B[0, :3, :5] slice={slice_B}")

    def forward(
        self,
        cell_expr: torch.Tensor,
        drugA_frags: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[str]],
        drugA_mask: Optional[Union[torch.Tensor, List[str]]] = None,
        drugB_frags: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[str]]] = None,
        drugB_mask: Optional[torch.Tensor] = None,
        dose_grid: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        return_params: bool = False,
        debug_print_fragments: bool = False,
        print_fragments: Optional[bool] = None,
        step: Optional[int] = None,
        print_every: int = 100
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

        should_print_frags = print_fragments if print_fragments is not None else self.print_fragments

        # Parse inputs to resolve positional or keyword usage cleanly
        fp_A, mask_A, fp_B, mask_B, grid = self._parse_inputs(
            cell_expr=cell_expr,
            drugA_frags=drugA_frags,
            drugA_mask=drugA_mask,
            drugB_frags=drugB_frags,
            drugB_mask=drugB_mask,
            dose_grid=dose_grid,
            device=device,
            print_fragments=should_print_frags
        )

        # STAGE 1: Cell-Line Encoder
        # Output c: (B, 512)
        c = self.cell_encoder(cell_expr)
        n, m = fp_A.size(1), fp_B.size(1)

        # STAGE 2 & 3: FragmentEncoder (UNPADDED) -> LayerNorm (UNPADDED) -> Standard E2 FiLM (UNPADDED) -> Pad AFTER FiLM & Mask
        # Mathematical sequence:
        # 1. Unpadded real fragments -> shared FragmentEncoder -> (N_real, d)
        # 2. Unpadded real fragments -> shared LayerNorm -> (N_real, d)
        # 3. Unpadded real fragments -> shared Standard E2 FiLM -> (N_real, d)
        # 4. Pad conditioned fragment vectors AFTER FiLM to max sequence length -> (B, n_max, d)
        # 5. Generate explicit boolean mask -> (B, n_max)
        F_tilde_A, mask_A_post = self._encode_and_condition_unpadded_fragments(fp_A, mask_A, c)
        F_tilde_B, mask_B_post = self._encode_and_condition_unpadded_fragments(fp_B, mask_B, c)

        mask_A = mask_A_post
        mask_B = mask_B_post

        assert F_tilde_A.shape == (B, n, self.d_dim), f"Stage 3 F_tilde_A shape mismatch: got {tuple(F_tilde_A.shape)}"
        assert F_tilde_B.shape == (B, m, self.d_dim), f"Stage 3 F_tilde_B shape mismatch: got {tuple(F_tilde_B.shape)}"

        # DEBUG HOOK: Print raw per-fragment embeddings F_tilde_A and F_tilde_B
        if debug_print_fragments and step is not None and step % print_every == 0:
            self._print_debug_fragment_embeddings(F_tilde_A, mask_A, F_tilde_B, mask_B, step)

        # STAGE 4: Masked Bidirectional Cross-Attention
        # Inputs: F_tilde_A, mask_A, F_tilde_B, mask_B
        # Output: mu_A_from_B, p_A_from_B, mu_B_from_A, p_B_from_A (each shape B, d)
        mu_A_from_B, p_A_from_B, mu_B_from_A, p_B_from_A = self.cross_attention(
            F_tilde_A=F_tilde_A,
            mask_A=mask_A,
            F_tilde_B=F_tilde_B,
            mask_B=mask_B
        )

        # STAGE 5: Direct Concatenation of Directional Features (r_AB in R^512)
        # Directly concatenate the 4 directional cross-attention representations:
        # [mu_A_from_B, p_A_from_B, mu_B_from_A, p_B_from_A]
        r_AB = torch.cat([mu_A_from_B, p_A_from_B, mu_B_from_A, p_B_from_A], dim=-1)
        assert r_AB.shape == (B, 4 * self.d_dim), f"Stage 5 r_AB shape mismatch: got {tuple(r_AB.shape)}"

        # STAGE 6: Cell Feature Concatenation (r_final in R^1024)
        # Concatenate r_AB (512-dim) with cell line representation c (512-dim)
        r_final = torch.cat([r_AB, c], dim=-1)
        expected_r_final_dim = 4 * self.d_dim + self.cell_dim
        assert r_final.shape == (B, expected_r_final_dim), f"Stage 6 r_final shape mismatch: got {tuple(r_final.shape)}"

        # STAGE 7: ONE Unified Parameter MLP (1024 -> 512 -> 256 -> 9)
        # Predicts all 9 raw parameter logits directly from r_final
        raw_params = self.parameter_heads(r_final)

        # STAGE 8: Constraint / Transform Stage
        # Applies Softplus (positivity), Sigmoid (range [0, 1]), and Log-Space EC50 constraints
        params = self.constraint_transform(raw_params)

        # STAGE 9: Differentiable Bivariate Hill Surface Solver
        # Closed-form calculation of viability surface Y (B, M, N)
        Y_pred = self.bivariate_solver(params=params, dose_grid=grid)

        if return_params:
            return Y_pred, params
        return Y_pred

    def _parse_inputs(
        self,
        cell_expr: torch.Tensor,
        drugA_frags: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[str]],
        drugA_mask: Optional[Union[torch.Tensor, List[str]]] = None,
        drugB_frags: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[str]]] = None,
        drugB_mask: Optional[torch.Tensor] = None,
        dose_grid: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        device: Union[torch.device, str] = "cpu",
        print_fragments: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Helper to parse flexible positional and keyword input signatures."""
        if isinstance(drugA_frags, (list, tuple)) and len(drugA_frags) > 0 and isinstance(drugA_frags[0], str):
            # High-level signature: forward(cell_expr, smiles_A, smiles_B, dose_grid)
            smiles_A = drugA_frags
            smiles_B = drugA_mask if isinstance(drugA_mask, (list, tuple)) else drugB_frags
            grid = drugB_frags if isinstance(drugB_frags, tuple) else dose_grid

            fp_A, mask_A, frags_A_list = collate_brics_fragments(smiles_A, n_bits=self.frag_fp_dim, device=device)
            fp_B, mask_B, frags_B_list = collate_brics_fragments(smiles_B, n_bits=self.frag_fp_dim, device=device)

            if print_fragments:
                print_batch_drug_fragments(smiles_A, frags_A_list, smiles_B, frags_B_list)

            return fp_A, mask_A, fp_B, mask_B, grid

        return drugA_frags, drugA_mask, drugB_frags, drugB_mask, dose_grid


# Alias for backward compatibility
CancerComboBRICS = CancerComboBRICSSymmetric
