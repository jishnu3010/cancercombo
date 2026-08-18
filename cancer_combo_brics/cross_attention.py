"""
Masked Bidirectional Cross-Attention Module for CancerCombo-BRICS-Symmetric.

Implements manual Multi-Head Cross-Attention from scratch without PyTorch's nn.MultiheadAttention
to avoid A100 GPU deadlock issues. Respects key-side and query-side BRICS fragment padding masks,
and performs masked mean and max pooling for both A -> B and B -> A directions.
"""

import math
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualMultiHeadCrossAttention(nn.Module):
    """
    Manual implementation of Multi-Head Cross-Attention without nn.MultiheadAttention.

    Computes:
        Q = X_query @ W_Q
        K = X_kv @ W_K
        V = X_kv @ W_V
        Attention(Q, K, V) = Softmax((Q @ K^T) / sqrt(d_k) + key_mask) @ V
        Output = Attention @ W_O

    Args:
        d_dim: Hidden dimension d (default: 128).
        num_heads: Number of attention heads (default: 4).
    """

    def __init__(self, d_dim: int = 128, num_heads: int = 4):
        super().__init__()
        assert d_dim % num_heads == 0, f"d_dim ({d_dim}) must be divisible by num_heads ({num_heads})"
        self.d_dim = d_dim
        self.num_heads = num_heads
        self.head_dim = d_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Learned linear projections for Q, K, V, and Out
        self.W_Q = nn.Linear(d_dim, d_dim, bias=False)
        self.W_K = nn.Linear(d_dim, d_dim, bias=False)
        self.W_V = nn.Linear(d_dim, d_dim, bias=False)
        self.W_O = nn.Linear(d_dim, d_dim)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for single-direction cross attention.

        Args:
            x_q: Query sequence tensor of shape (B, L_Q, d_dim).
            x_kv: Key/Value sequence tensor of shape (B, L_KV, d_dim).
            key_padding_mask: BoolTensor of shape (B, L_KV), where True = VALID keys,
                              False = PADDED keys.

        Returns:
            out: Cross-attended output sequence tensor of shape (B, L_Q, d_dim).
        """
        B, L_Q, _ = x_q.shape
        _, L_KV, _ = x_kv.shape

        # Linear projections
        # Shape: (B, L_Q, d_dim) -> (B, num_heads, L_Q, head_dim)
        Q = self.W_Q(x_q).view(B, L_Q, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (B, L_KV, d_dim) -> (B, num_heads, L_KV, head_dim)
        K = self.W_K(x_kv).view(B, L_KV, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x_kv).view(B, L_KV, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        # Q @ K^T shape: (B, num_heads, L_Q, L_KV)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply key-side padding mask before softmax
        if key_padding_mask is not None:
            # Reshape key_padding_mask for broadcasting: (B, 1, 1, L_KV)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, -1e9)

        # Softmax over key sequence dimension L_KV
        attn_weights = F.softmax(scores, dim=-1)  # Shape: (B, num_heads, L_Q, L_KV)

        # Compute weighted sum over values: (B, num_heads, L_Q, head_dim)
        context = torch.matmul(attn_weights, V)

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
        mask_A_exp = mask_A.unsqueeze(-1).float()  # (B, n, 1)
        mask_B_exp = mask_B.unsqueeze(-1).float()  # (B, m, 1)

        H_A_from_B = out_A_from_B * mask_A_exp  # (B, n, d)
        H_B_from_A = out_B_from_A * mask_B_exp  # (B, m, d)

        # 4. Masked Pooling for A <- B (over fragment dimension n)
        valid_counts_A = mask_A_exp.sum(dim=1).clamp(min=1.0)  # (B, 1)
        mu_A_from_B = H_A_from_B.sum(dim=1) / valid_counts_A   # (B, d)

        H_A_masked = H_A_from_B.masked_fill(~mask_A.unsqueeze(-1), -1e9)
        p_A_from_B = H_A_masked.max(dim=1)[0]                  # (B, d)
        p_A_from_B = torch.where(mask_A.any(dim=1, keepdim=True), p_A_from_B, torch.zeros_like(p_A_from_B))

        # 5. Masked Pooling for B <- A (over fragment dimension m)
        valid_counts_B = mask_B_exp.sum(dim=1).clamp(min=1.0)  # (B, 1)
        mu_B_from_A = H_B_from_A.sum(dim=1) / valid_counts_B   # (B, d)

        H_B_masked = H_B_from_A.masked_fill(~mask_B.unsqueeze(-1), -1e9)
        p_B_from_A = H_B_masked.max(dim=1)[0]                  # (B, d)
        p_B_from_A = torch.where(mask_B.any(dim=1, keepdim=True), p_B_from_A, torch.zeros_like(p_B_from_A))

        assert mu_A_from_B.shape == (B, self.d_dim)
        assert p_A_from_B.shape == (B, self.d_dim)
        assert mu_B_from_A.shape == (B, self.d_dim)
        assert p_B_from_A.shape == (B, self.d_dim)

        return mu_A_from_B, p_A_from_B, mu_B_from_A, p_B_from_A


# Alias for backward compatibility
BidirectionalCrossAttention = MaskedBidirectionalCrossAttention
