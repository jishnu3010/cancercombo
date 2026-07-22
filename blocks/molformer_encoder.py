import torch
import torch.nn as nn
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel, AutoTokenizer = None, None
from typing import Tuple, Optional

class LocalTransformerEncoderLayer(nn.Module):
    """Explicit Transformer encoder layer using nn.MultiheadAttention directly (avoids nested tensor bugs)."""
    
    def __init__(self, d_model: int = 256, nhead: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # Multi-head self-attention
        attn_out, _ = self.self_attn(
            query=src, key=src, value=src,
            key_padding_mask=src_key_padding_mask
        )
        x = src + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feed-forward network
        ffn_out = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)
        return x

class LocalTransformerEncoder(nn.Module):
    """Transformer encoder chain utilizing LocalTransformerEncoderLayers."""
    
    def __init__(self, d_model: int = 256, nhead: int = 4, dim_feedforward: int = 512, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = src
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x

class MolFormerEncoder(nn.Module):
    """MolFormer feature extraction encoder wrapper with transformer fallback."""
    
    def __init__(
        self,
        d_model: int = 256,
        molformer_in_dim: int = 768,
        use_pretrained: bool = False,
        model_name: str = "ibm/MoLFormer-XL-CIMA-100M",
        vocab_size: int = 100,
        max_seq_len: int = 128
    ):
        super().__init__()
        self.use_pretrained = use_pretrained
        self.d_model = d_model
        
        if self.use_pretrained:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.molformer = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                self.proj = nn.Linear(molformer_in_dim, d_model)
            except Exception as e:
                print(f"Warning: Failed to load pre-trained MolFormer ({e}). Falling back to local transformer.")
                self.use_pretrained = False
                
        if not self.use_pretrained:
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, d_model))
            self.transformer = LocalTransformerEncoder(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                num_layers=2,
                dropout=0.1
            )
            self.proj = nn.Identity()
            
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs forward sequence encoding and token-level average pooling.

        Args:
            input_ids: Token ID sequences of shape (B, L).
            attention_mask: Attention masks of shape (B, L).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sequence embedding tokens and the pooled vector.
        """
        if self.use_pretrained:
            outputs = self.molformer(input_ids=input_ids, attention_mask=attention_mask)
            seq_feats = outputs.last_hidden_state
            seq_embeddings = self.proj(seq_feats)
        else:
            seq_feats = self.embedding(input_ids) + self.pos_encoder[:, :input_ids.size(1), :]
            key_padding_mask = ~(attention_mask.bool())
            all_masked = key_padding_mask.all(dim=-1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False
            seq_embeddings = self.transformer(seq_feats, src_key_padding_mask=key_padding_mask)
            seq_embeddings = self.proj(seq_embeddings)
            
        masked_embeddings = seq_embeddings * attention_mask.unsqueeze(-1)
        token_counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_embedding = masked_embeddings.sum(dim=1) / token_counts
        
        return seq_embeddings, pooled_embedding
