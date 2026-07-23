import torch
import torch.nn as nn
import math
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel, AutoTokenizer = None, None
from typing import Tuple, Optional

class SafeTransformerEncoderLayer(nn.Module):
    """Explicit Transformer encoder layer using manual attention (immune to A100 multihead deadlocks)."""
    
    def __init__(self, d_model: int = 256, nhead: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Manual projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = src.shape
        
        # 1. Project Q, K, V
        Q = self.q_proj(src)
        K = self.k_proj(src)
        V = self.v_proj(src)
        
        # 2. Reshape for multi-head attention: (B, nhead, L, head_dim)
        Q = Q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        
        # 3. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply padding mask if provided (src_key_padding_mask: True where padded)
        if src_key_padding_mask is not None:
            # Shape expansion for heads: (B, 1, 1, L)
            mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e4)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout1(attn_weights)
        
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_out = self.out_proj(attn_out)
        
        # Residual 1 & Norm 1
        x = self.norm1(src + attn_out)
        
        # Feed-forward network
        ffn_out = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class LocalTransformerEncoder(nn.Module):
    """Transformer encoder chain utilizing SafeTransformerEncoderLayers."""
    
    def __init__(self, d_model: int = 256, nhead: int = 4, dim_feedforward: int = 512, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SafeTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = src
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x