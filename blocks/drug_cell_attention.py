import torch
import torch.nn as nn

class DrugCellCrossAttention(nn.Module):
    """Conditions drug representations on the cell line transcriptomics using cross-attention."""
    
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, drug_emb: torch.Tensor, cell_emb: torch.Tensor) -> torch.Tensor:
        """Runs cross-attention over drug embedding (query) and cell tokens (key/value).

        Args:
            drug_emb: Enhanced drug embedding of shape (B, d_model).
            cell_emb: Cell line embedding of shape (B, n_pathways, d_model).

        Returns:
            torch.Tensor: Cell-conditioned representation of shape (B, d_model).
        """
        q = drug_emb.unsqueeze(1) # (B, 1, d_model)
        attn_out, _ = self.cross_attn(query=q, key=cell_emb, value=cell_emb)
        x = self.norm(q + attn_out)
        out = self.ffn(x) + x
        return out.squeeze(1)
