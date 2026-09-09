"""Bidirectional shared-weight multi-head fragment cross-attention and pooling."""

from __future__ import annotations

import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalFragmentCrossAttention(nn.Module):
    """Bidirectional shared-weight cross-attention for drug fragment sets.

    Drug A attends to Drug B (A -> B) and Drug B attends to Drug A (B -> A)
    using the EXACT same projection weights.

    Pooled representations:
      mean_A_from_B (512), max_A_from_B (512)
      mean_B_from_A (512), max_B_from_A (512)
      Concatenated -> r_AB in R^2048
      Projected -> r'_AB in R^512
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 512 // 4 = 128
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Shared projections for both A->B and B->A directions
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

        # Projection from 2048 to 512
        self.pooled_projection = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.pooled_projection[0].weight)
        if self.pooled_projection[0].bias is not None:
            nn.init.zeros_(self.pooled_projection[0].bias)

    def _cross_attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Compute multi-head cross-attention and masked pooling for one direction.

        Args:
            query: (B, N_q, 512)
            key: (B, N_k, 512)
            value: (B, N_k, 512)
            key_mask: (B, N_k), 1 for valid key, 0 for pad
            query_mask: (B, N_q), 1 for valid query, 0 for pad

        Returns:
            mean_pooled: (B, 512)
            max_pooled: (B, 512)
            attended: (B, N_q, 512)
            diag: Attention diagnostic metrics
        """
        B, N_q, _ = query.shape
        _, N_k, _ = key.shape

        # Linear projections & reshape to (B, num_heads, N, head_dim)
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention logits: (B, num_heads, N_q, N_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply key padding mask: (B, 1, 1, N_k)
        if key_mask is not None:
            key_mask_expanded = key_mask.view(B, 1, 1, N_k)
            # Mask out invalid keys with large negative number
            scores = scores.masked_fill(key_mask_expanded == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        # In case all keys were masked for a sample, replace NaNs with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # Attended representation: (B, num_heads, N_q, head_dim) -> (B, N_q, 512)
        context = torch.matmul(attn_weights_dropped, V)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.embed_dim)
        attended = self.out_proj(context)

        # Mask out padded query tokens
        q_mask_expanded = query_mask.unsqueeze(-1)  # (B, N_q, 1)
        attended = attended * q_mask_expanded

        # Masked mean pooling across valid query tokens
        valid_q_counts = torch.clamp(query_mask.sum(dim=1, keepdim=True), min=1.0)  # (B, 1)
        mean_pooled = attended.sum(dim=1) / valid_q_counts  # (B, 512)

        # Masked max pooling across valid query tokens
        # Set padded queries to -1e9 before taking max
        attended_for_max = attended.masked_fill(q_mask_expanded == 0, -1e9)
        max_pooled = attended_for_max.max(dim=1).values  # (B, 512)
        # If any row had 0 valid queries, reset max to 0
        max_pooled = torch.where(query_mask.sum(dim=1, keepdim=True) > 0, max_pooled, torch.zeros_like(max_pooled))

        # Diagnostics
        with torch.no_grad():
            entropy = -(attn_weights * torch.log(torch.clamp(attn_weights, min=1e-12))).sum(dim=-1).mean().item()
            max_logit = scores.max().item()
            min_logit = scores.min().item()
            diag = {
                "attn_entropy": entropy,
                "attn_max_logit": max_logit,
                "attn_min_logit": min_logit,
            }

        return mean_pooled, max_pooled, attended, diag

    def forward(
        self,
        F_A: torch.Tensor,
        F_B: torch.Tensor,
        mask_A: torch.Tensor,
        mask_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Bidirectional cross-attention forward pass.

        Args:
            F_A: Drug A modulated fragments of shape (B, N, 512).
            F_B: Drug B modulated fragments of shape (B, M, 512).
            mask_A: Validity mask for Drug A of shape (B, N).
            mask_B: Validity mask for Drug B of shape (B, M).

        Returns:
            r_prime_AB: Projected drug-pair representation of shape (B, 512).
            r_AB: Concatenated pooled representation of shape (B, 2048).
            diagnostics: Combined attention diagnostics.
        """
        B = F_A.shape[0]

        # Direction 1: A attends to B (A -> B)
        mean_A_from_B, max_A_from_B, _, diag_A2B = self._cross_attend(
            query=F_A,
            key=F_B,
            value=F_B,
            key_mask=mask_B,
            query_mask=mask_A,
        )

        # Direction 2: B attends to A (B -> A)
        mean_B_from_A, max_B_from_A, _, diag_B2A = self._cross_attend(
            query=F_B,
            key=F_A,
            value=F_A,
            key_mask=mask_A,
            query_mask=mask_B,
        )

        # Concatenate into r_AB in R^2048
        r_AB = torch.cat(
            [mean_A_from_B, max_A_from_B, mean_B_from_A, max_B_from_A],
            dim=-1,
        )
        assert r_AB.shape == (B, 2048), f"Expected r_AB shape ({B}, 2048), got {r_AB.shape}"

        # Project into r'_AB in R^512
        r_prime_AB = self.pooled_projection(r_AB)
        assert r_prime_AB.shape == (B, 512), (
            f"Expected r'_AB shape ({B}, 512), got {r_prime_AB.shape}"
        )

        diagnostics = {
            "A2B_entropy": diag_A2B["attn_entropy"],
            "B2A_entropy": diag_B2A["attn_entropy"],
            "norm_r_AB": torch.norm(r_AB, dim=-1).mean().item(),
            "norm_r_prime_AB": torch.norm(r_prime_AB, dim=-1).mean().item(),
        }

        return r_prime_AB, r_AB, diagnostics
