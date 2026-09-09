"""Genuine Pretrained Mol2Vec fragment embedding module mapping chemical fragments to 512-D vectors.

Flow:
  1. Fragment SMILES -> RDKit Mol -> Mol2Vec Morgan radius-1 environment tokens
  2. Token lookup in genuine pretrained Mol2Vec model (model_300dim.pkl)
  3. Pretrained vector sum-pooling -> 300-D fragment vector
  4. Trainable linear projection (300 -> 512) + LayerNorm + Dropout -> 512-D fragment embedding
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import rdkit.Chem as Chem

# NOTE: gensim and mol2vec are imported lazily inside Mol2VecEncoder.__init__
# to allow the rest of the package to be importable even when gensim is not installed.
# Tests that do not need Mol2Vec (checkpoint, pairwise, cell encoder, etc.) will still run.

logger = logging.getLogger(__name__)


class Mol2VecEncoder(nn.Module):
    """Genuine Pretrained Mol2Vec fragment embedding module.

    Uses pretrained 300-D Mol2Vec vectors (model_300dim.pkl) loaded into a frozen
    nn.Embedding lookup table, followed by a trainable 300 -> 512 projection.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        native_dim: int = 300,
        fragment_dim: int = 512,
        radius: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.native_dim = native_dim
        self.fragment_dim = fragment_dim
        self.radius = radius

        # Lazy import of gensim/mol2vec — deferred to instantiation time so the
        # rest of the package can be imported without gensim being installed.
        try:
            from gensim.models import Word2Vec as _Word2Vec
            from mol2vec.features import mol2alt_sentence as _mol2alt_sentence
        except ImportError as e:
            raise ImportError(
                f"Mol2VecEncoder requires gensim>=4.0 and mol2vec. "
                f"On Python 3.14, install Microsoft C++ Build Tools first, then: "
                f"pip install 'gensim>=4.1.0,<5.0' mol2vec\n"
                f"Original error: {e}"
            ) from e
        # Cache mol2alt_sentence for use in tokenize_fragment
        self._mol2alt_sentence = _mol2alt_sentence

        if model_path is None:
            # Check default path options
            possible_paths = [
                os.path.join("data", "model_300dim.pkl"),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "model_300dim.pkl")),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    model_path = p
                    break

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Genuine pretrained Mol2Vec model file not found at '{model_path}'. "
                "BLOCKED — genuine pretrained Mol2Vec weights unavailable."
            )

        logger.info(f"Loading genuine pretrained Mol2Vec model from '{model_path}'...")
        w2v_model = _Word2Vec.load(model_path)
        assert w2v_model.wv.vector_size == native_dim, (
            f"Expected Mol2Vec vector dimension {native_dim}, got {w2v_model.wv.vector_size}"
        )

        # Build token-to-index mapping (index 0 reserved for padding / unknown)
        self.token_to_idx: Dict[str, int] = {}
        vocab_keys = list(w2v_model.wv.key_to_index.keys())
        vocab_size = len(vocab_keys) + 1  # 1-indexed, 0 is padding/OOV

        weights_matrix = torch.zeros((vocab_size, native_dim), dtype=torch.float32)

        for i, token in enumerate(vocab_keys, start=1):
            self.token_to_idx[str(token)] = i
            weights_matrix[i] = torch.from_numpy(w2v_model.wv[token].copy())

        # Frozen pretrained embedding layer (requires_grad = False)
        self.embeddings = nn.Embedding.from_pretrained(
            weights_matrix,
            freeze=True,
            padding_idx=0,
        )
        assert not self.embeddings.weight.requires_grad, "Mol2Vec pretrained embeddings must be frozen!"

        # Trainable projection from native_dim (300) to fragment_dim (512)
        self.projection = nn.Sequential(
            nn.Linear(native_dim, fragment_dim),
            nn.LayerNorm(fragment_dim),
            nn.Dropout(dropout),
        )

        # In-memory fragment embedding cache namespace: smiles -> (512,) tensor
        self.cache_namespace = "mol2vec_pretrained_v1"
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self._init_projection_weights()

    def _init_projection_weights(self) -> None:
        for m in self.projection:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def clear_cache(self) -> None:
        """Clear cached base fragment embeddings."""
        self._embedding_cache.clear()

    def tokenize_fragment(self, smiles: str) -> List[int]:
        """Convert fragment SMILES into Mol2Vec vocabulary token indices."""
        if not smiles or not isinstance(smiles, str) or not smiles.strip():
            return [0]

        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return [0]

        try:
            tokens = self._mol2alt_sentence(mol, radius=self.radius)
            indices = [self.token_to_idx.get(t, 0) for t in tokens]
            return indices if indices else [0]
        except Exception as e:
            logger.warning(f"Error tokenizing SMILES '{smiles}': {e}")
            return [0]

    def forward_single_fragments(
        self,
        frag_smiles_list: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a flat list of fragment SMILES strings into 512-D vectors."""
        if not frag_smiles_list:
            return torch.zeros((0, self.fragment_dim), device=device)

        native_vecs_list: List[torch.Tensor] = []
        embed_device = self.embeddings.weight.device

        for s in frag_smiles_list:
            indices = self.tokenize_fragment(s)
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=embed_device)
            # Lookup pretrained vectors: (num_tokens, 300)
            vecs = self.embeddings(idx_tensor)
            # Sum-pooling across fragment tokens -> (300,)
            summed = vecs.sum(dim=0)
            native_vecs_list.append(summed)

        native_vecs = torch.stack(native_vecs_list, dim=0).to(device)
        assert native_vecs.shape == (len(frag_smiles_list), self.native_dim)

        # Trainable projection to 512-D
        projected = self.projection(native_vecs)
        return projected

    def encode_drug_fragments_batch(
        self,
        batched_fragments: List[List[str]],
        mask: torch.Tensor,
        device: torch.device,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Encode a batch of drugs, each having N fragment SMILES strings.

        Args:
            batched_fragments: List of length B, each containing N fragment strings.
            mask: Float tensor of shape (B, N) with 1.0 for valid fragments, 0.0 for padding.
            device: Target PyTorch device.
            use_cache: Whether to use in-memory cache for valid fragments.

        Returns:
            F: Tensor of shape (B, N, 512) with padded positions zeroed out.
        """
        B = len(batched_fragments)
        if B == 0:
            return torch.zeros((0, 0, self.fragment_dim), device=device)

        N = len(batched_fragments[0])
        output = torch.zeros((B, N, self.fragment_dim), device=device)
        mask = mask.to(device)

        # Correct cache criterion: use self.training mode only.
        # The Mol2Vec embedding is frozen (requires_grad=False) so its detached output
        # is safe to cache during eval. The projection layer has requires_grad=True
        # but that does NOT make caching unsafe — only training mode does.
        allow_cache = use_cache and not self.training

        unique_to_encode: List[str] = []
        frag_to_idx: Dict[str, int] = {}

        for i in range(B):
            for j in range(N):
                if mask[i, j] > 0.5:
                    frag_str = batched_fragments[i][j]
                    cache_key = f"{self.cache_namespace}:{frag_str}"
                    if allow_cache and cache_key in self._embedding_cache:
                        output[i, j] = self._embedding_cache[cache_key].to(device)
                    else:
                        if frag_str not in frag_to_idx:
                            frag_to_idx[frag_str] = len(unique_to_encode)
                            unique_to_encode.append(frag_str)

        if unique_to_encode:
            encoded_unique = self.forward_single_fragments(unique_to_encode, device=device)
            if allow_cache:
                for frag_str, idx in frag_to_idx.items():
                    cache_key = f"{self.cache_namespace}:{frag_str}"
                    self._embedding_cache[cache_key] = encoded_unique[idx].detach().cpu()

            for i in range(B):
                for j in range(N):
                    if mask[i, j] > 0.5:
                        frag_str = batched_fragments[i][j]
                        if frag_str in frag_to_idx:
                            idx = frag_to_idx[frag_str]
                            output[i, j] = encoded_unique[idx]

        output = output * mask.unsqueeze(-1)
        assert output.shape == (B, N, self.fragment_dim), (
            f"Expected shape ({B}, {N}, {self.fragment_dim}), got {output.shape}"
        )
        return output

