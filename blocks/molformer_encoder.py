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

class MolFormerEncoder(nn.Module):
    """SMILES Transformer encoder block supporting pretrained AutoModel or local SafeTransformerEncoder.
    
    Processes SMILES token sequences into token-level representations and pooled embeddings.
    Made compatible with PyTorch and device placement requirements.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        vocab_size: int = 100,
        nhead: int = 4,
        dim_feedforward: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_pretrained: bool = False,
        model_name: str = "ibm/MoLFormer-XL-CIMA-100M"
    ):
        super().__init__()
        self.d_model = d_model
        self.use_pretrained = use_pretrained and (AutoModel is not None)
        
        if self.use_pretrained:
            try:
                # Load pretrained HuggingFace MolFormer encoder model if available
                self.pretrained_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                hidden_size = getattr(self.pretrained_model.config, "hidden_size", 768)
                self.proj = nn.Linear(hidden_size, d_model) if hidden_size != d_model else nn.Identity()
            except Exception as e:
                # Fall back gracefully to local SafeTransformerEncoder on network/load failure
                self.use_pretrained = False
                
        if not self.use_pretrained:
            # Local token embedding and deadlock-free SafeTransformerEncoder
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.transformer = LocalTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_layers=num_layers,
                dropout=dropout
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes token IDs into sequence embeddings and pooled representations.
        
        Args:
            input_ids: Token ID tensor of shape (B, L).
            attention_mask: Attention mask tensor of shape (B, L) where 1 indicates valid token and 0 padded.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sequence features (B, L, d_model) and pooled features (B, d_model).
        """
        if self.use_pretrained:
            outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
            seq_feats = self.proj(outputs.last_hidden_state)
        else:
            # Embedded sequence: (B, L, d_model)
            x = self.embedding(input_ids)
            # In SafeTransformerEncoderLayer, src_key_padding_mask is True where padded (mask == 0)
            padding_mask = (attention_mask == 0)
            seq_feats = self.transformer(x, src_key_padding_mask=padding_mask)
            
        # Masked mean pooling over valid tokens to construct single drug embedding (CLS / pooled)
        mask_exp = attention_mask.unsqueeze(-1).to(dtype=seq_feats.dtype)
        sum_mask = mask_exp.sum(dim=1).clamp(min=1e-9)
        pooled_emb = (seq_feats * mask_exp).sum(dim=1) / sum_mask
        
        return seq_feats, pooled_emb