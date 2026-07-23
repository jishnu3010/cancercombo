import torch
import torch.nn as nn
import math

class AttentionMultiRepresentationFusion(nn.Module):
    """Dynamic multi-head self-attention fusion block for multi-modal drug representations."""
    
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Manual Q, K, V projections to bypass nn.MultiheadAttention A100 bug
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        
        self.pooling = nn.Linear(3 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        molformer_emb: torch.Tensor,
        morgan_emb: torch.Tensor,
        descriptor_emb: torch.Tensor
    ) -> torch.Tensor:
        """Fuses three representation modes dynamically using manual self-attention."""
        B = molformer_emb.size(0)
        
        # Stack representations: (B, seq_len=3, d_model)
        stacked = torch.stack([molformer_emb, morgan_emb, descriptor_emb], dim=1)
        seq_len = 3
        
        # 1. Project Q, K, V
        Q = self.q_proj(stacked)
        K = self.k_proj(stacked)
        V = self.v_proj(stacked)
        
        # 2. Reshape for multi-head attention: (B, n_heads, seq_len, head_dim)
        Q = Q.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3. Manual scaled dot-product attention (Immune to C++ kernel deadlocks)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, n_heads, 3, 3)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_out = torch.matmul(attn_weights, V) # (B, n_heads, 3, head_dim)
        
        # 4. Reshape back to (B, 3, d_model) and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)
        
        # 5. Flatten, pool, and apply residual
        flat_attn = attn_out.reshape(B, -1) # (B, 3 * d_model)
        fused = self.norm(self.pooling(flat_attn) + molformer_emb) # Residual to MolFormer
        return fused