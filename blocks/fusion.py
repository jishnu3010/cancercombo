import torch
import torch.nn as nn

class AttentionMultiRepresentationFusion(nn.Module):
    """Dynamic multi-head self-attention fusion block for multi-modal drug representations."""
    
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.pooling = nn.Linear(3 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        molformer_emb: torch.Tensor,
        morgan_emb: torch.Tensor,
        descriptor_emb: torch.Tensor
    ) -> torch.Tensor:
        """Fuses three representation modes dynamically using self-attention.

        Args:
            molformer_emb: Pooled MolFormer embedding of shape (B, d_model).
            morgan_emb: Projected Morgan embedding of shape (B, d_model).
            descriptor_emb: Projected Descriptors embedding of shape (B, d_model).

        Returns:
            torch.Tensor: Combined drug embedding of shape (B, d_model).
        """
        stacked = torch.stack([molformer_emb, morgan_emb, descriptor_emb], dim=1) # (B, 3, d_model)
        attn_out, _ = self.self_attn(query=stacked, key=stacked, value=stacked)
        flat_attn = attn_out.reshape(attn_out.size(0), -1) # (B, 3 * d_model)
        fused = self.norm(self.pooling(flat_attn) + molformer_emb) # Residual to MolFormer
        return fused
