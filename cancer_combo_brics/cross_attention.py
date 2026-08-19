"""
Masked Bidirectional Cross-Attention Module for CancerCombo-BRICS-Symmetric.

Computes multi-head cross-attention between Drug A fragments and Drug B fragments
conditioned on cell line representation c. Key-side masks are applied before softmax;
query-side masks are applied to outputs before masked pooling.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualMultiHeadCrossAttention(nn.Module):
    """
    Manual Multi-Head Cross-Attention Module.
    Constructed without PyTorch's nn.MultiheadAttention to ensure explicit, transparent,
    and verifiable key-side mask application before softmax.
    Uses pure elementwise tensor contraction to guarantee 100% execution on native PyTorch
    CUDA C++ kernels without invoking Triton JIT C compilers.

    Args:
        d_dim: Embedding dimension d (default: 128).
        num_heads: Number of attention heads (default: 4).
    """

    def __init__(self, d_dim: int = 128, num_heads: int = 4):
        super().__init__()
        assert d_dim % num_heads == 0, f"d_dim ({d_dim}) must be divisible by num_heads ({num_heads})"
        self.d_dim = d_dim
        self.num_heads = num_heads
        self.head_dim = d_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections for Query, Key, Value, and Output
        self.W_Q = nn.Linear(d_dim, d_dim, bias=False)
        self.W_K = nn.Linear(d_dim, d_dim, bias=False)
        self.W_V = nn.Linear(d_dim, d_dim, bias=False)
        self.W_O = nn.Linear(d_dim, d_dim, bias=False)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for manual multi-head cross-attention.

        Args:
            x_q: Query sequence tensor of shape (B, L_Q, d_dim).
            x_kv: Key/Value sequence tensor of shape (B, L_KV, d_dim).
            key_padding_mask: BoolTensor of shape (B, L_KV), True = valid, False = padding.

        Returns:
            out: Cross-attended output sequence tensor of shape (B, L_Q, d_dim).
        """
        B, L_Q, _ = x_q.shape
        _, L_KV, _ = x_kv.shape

        # Linear projections -> Shape: (B, num_heads, L, head_dim)
        Q = self.W_Q(x_q).view(B, L_Q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x_kv).view(B, L_KV, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x_kv).view(B, L_KV, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute dot-product attention scores using pure elementwise contraction:
        # Q shape: (B, num_heads, L_Q, 1, head_dim), K shape: (B, num_heads, 1, L_KV, head_dim)
        scores = (Q.unsqueeze(-2) * K.unsqueeze(-3)).sum(dim=-1) * self.scale  # (B, num_heads, L_Q, L_KV)

        # Apply key-side padding mask before softmax
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_KV)
            fill_val = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, fill_val)

        # Softmax over key sequence dimension L_KV
        attn_weights = F.softmax(scores, dim=-1)  # Shape: (B, num_heads, L_Q, L_KV)

        # Compute weighted sum over values using pure elementwise contraction:
        # attn_weights shape: (B, num_heads, L_Q, L_KV, 1), V shape: (B, num_heads, 1, L_KV, head_dim)
        context = (attn_weights.unsqueeze(-1) * V.unsqueeze(-3)).sum(dim=-2)  # (B, num_heads, L_Q, head_dim)

        # Concatenate heads: (B, L_Q, d_dim)
        context = context.transpose(1, 2).contiguous().view(B, L_Q, self.d_dim)

        # Output projection
        out = self.W_O(context)  # Shape: (B, L_Q, d_dim)
        return out


class MaskedBidirectionalCrossAttention(nn.Module):
    """
    Masked Bidirectional Cross-Attention module for Drug A and Drug B fragment interactions.

    Architecture:
        - Attention blocks computing A -> B and B -> A cross attention.
        - Key-side mask applied before softmax; query-side mask applied to outputs.
        - Masked mean and max pooling over fragment dimensions.

    Args:
        d_dim: Fragment embedding dimension d (default: 128).
        num_heads: Number of attention heads (default: 4).
        shared_weights: If True, shares attention projection weights across A->B and B->A
                        for exact mathematical drug-order invariance (default: True).
    """

    def __init__(
        self,
        d_dim: int = 128,
        num_heads: int = 4,
        shared_weights: bool = True
    ):
        super().__init__()
        self.d_dim = d_dim
        self.num_heads = num_heads
        self.shared_weights = shared_weights

        if shared_weights:
            self.attn_shared = ManualMultiHeadCrossAttention(d_dim=d_dim, num_heads=num_heads)
            self.attn_a_to_b = self.attn_shared
            self.attn_b_to_a = self.attn_shared
        else:
            self.attn_a_to_b = ManualMultiHeadCrossAttention(d_dim=d_dim, num_heads=num_heads)
            self.attn_b_to_a = ManualMultiHeadCrossAttention(d_dim=d_dim, num_heads=num_heads)

    def forward(
        self,
        F_tilde_A: torch.Tensor,
        mask_A: torch.Tensor,
        F_tilde_B: torch.Tensor,
        mask_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for masked bidirectional cross-attention.

        Args:
            F_tilde_A: Cell-conditioned fragment tensor for Drug A, shape (B, n, d).
            mask_A: BoolTensor of shape (B, n), True = valid fragment, False = padding.
            F_tilde_B: Cell-conditioned fragment tensor for Drug B, shape (B, m, d).
            mask_B: BoolTensor of shape (B, m), True = valid fragment, False = padding.

        Returns:
            Tuple of pooled statistic tensors:
                - mu_A_from_B: (B, d), masked mean pool A <- B
                - p_A_from_B: (B, d), masked max pool A <- B
                - mu_B_from_A: (B, d), masked mean pool B <- A
                - p_B_from_A: (B, d), masked max pool B <- A
        """
        assert F_tilde_A.dim() == 3 and F_tilde_B.dim() == 3, "F_tilde_A and F_tilde_B must be 3D tensors"
        assert mask_A.dim() == 2 and mask_B.dim() == 2, "mask_A and mask_B must be 2D tensors"

        B, n, d = F_tilde_A.shape
        _, m, _ = F_tilde_B.shape
        assert d == self.d_dim, f"Expected fragment embedding dim {self.d_dim}, got {d}"

        # 1. Attention A -> B (Query: A, Key/Value: B, Key-side mask: mask_B)
        out_A_from_B = self.attn_a_to_b(x_q=F_tilde_A, x_kv=F_tilde_B, key_padding_mask=mask_B)  # (B, n, d)

        # 2. Attention B -> A (Query: B, Key/Value: A, Key-side mask: mask_A)
        out_B_from_A = self.attn_b_to_a(x_q=F_tilde_B, x_kv=F_tilde_A, key_padding_mask=mask_A)  # (B, m, d)

        # 3. Apply QUERY-side mask to outputs (zero out attention output rows for padded query fragments)
        mask_A_exp = mask_A.unsqueeze(-1).to(F_tilde_A.dtype)  # (B, n, 1)
        mask_B_exp = mask_B.unsqueeze(-1).to(F_tilde_B.dtype)  # (B, m, 1)

        H_A_from_B = out_A_from_B * mask_A_exp  # (B, n, d)
        H_B_from_A = out_B_from_A * mask_B_exp  # (B, m, d)

        # 4. Masked Pooling for A <- B (over fragment dimension n)
        valid_counts_A = mask_A_exp.sum(dim=1).clamp(min=1.0)  # (B, 1)
        mu_A_from_B = H_A_from_B.sum(dim=1) / valid_counts_A   # (B, d)

        fill_val_A = torch.finfo(H_A_from_B.dtype).min
        H_A_masked = H_A_from_B.masked_fill(~mask_A.unsqueeze(-1), fill_val_A)
        p_A_from_B = H_A_masked.max(dim=1)[0]                  # (B, d)
        p_A_from_B = torch.where(mask_A.any(dim=1, keepdim=True), p_A_from_B, torch.zeros_like(p_A_from_B))

        # 5. Masked Pooling for B <- A (over fragment dimension m)
        valid_counts_B = mask_B_exp.sum(dim=1).clamp(min=1.0)  # (B, 1)
        mu_B_from_A = H_B_from_A.sum(dim=1) / valid_counts_B   # (B, d)

        fill_val_B = torch.finfo(H_B_from_A.dtype).min
        H_B_masked = H_B_from_A.masked_fill(~mask_B.unsqueeze(-1), fill_val_B)
        p_B_from_A = H_B_masked.max(dim=1)[0]                  # (B, d)
        p_B_from_A = torch.where(mask_B.any(dim=1, keepdim=True), p_B_from_A, torch.zeros_like(p_B_from_A))

        assert mu_A_from_B.shape == (B, self.d_dim)
        assert p_A_from_B.shape == (B, self.d_dim)
        assert mu_B_from_A.shape == (B, self.d_dim)
        assert p_B_from_A.shape == (B, self.d_dim)

        return mu_A_from_B, p_A_from_B, mu_B_from_A, p_B_from_A


# Alias for backward compatibility
BidirectionalCrossAttention = MaskedBidirectionalCrossAttention
