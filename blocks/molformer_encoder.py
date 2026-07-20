import torch
import torch.nn as nn
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel, AutoTokenizer = None, None
from typing import Tuple, Optional

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
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                batch_first=True,
                dropout=0.1
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
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
            seq_embeddings = self.transformer(seq_feats, src_key_padding_mask=key_padding_mask)
            seq_embeddings = self.proj(seq_embeddings)
            
        masked_embeddings = seq_embeddings * attention_mask.unsqueeze(-1)
        token_counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_embedding = masked_embeddings.sum(dim=1) / token_counts
        
        return seq_embeddings, pooled_embedding
