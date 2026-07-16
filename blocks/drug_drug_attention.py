import torch
import torch.nn as nn
from typing import Tuple

class DrugDrugCrossAttention(nn.Module):
    """Mutually cross-attends conditioned Drug A and Drug B representations."""
    
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_a_to_b = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_b_to_a = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        
    def forward(self, cond_a: torch.Tensor, cond_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exchanges mutual interaction representations between Drug A and Drug B.

        Args:
            cond_a: Cell-conditioned Drug A of shape (B, d_model).
            cond_b: Cell-conditioned Drug B of shape (B, d_model).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Interaction-aware Drug A and Drug B embeddings.
        """
        q_a = cond_a.unsqueeze(1) # (B, 1, d_model)
        q_b = cond_b.unsqueeze(1) # (B, 1, d_model)
        
        # A queries B
        attn_a, _ = self.attn_a_to_b(query=q_a, key=q_b, value=q_b)
        aware_a = self.norm_a(q_a + attn_a).squeeze(1)
        
        # B queries A
        attn_b, _ = self.attn_b_to_a(query=q_b, key=q_a, value=q_a)
        aware_b = self.norm_b(q_b + attn_b).squeeze(1)
        
        return aware_a, aware_b
