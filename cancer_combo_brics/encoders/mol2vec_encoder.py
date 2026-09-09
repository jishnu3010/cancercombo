"""Mol2Vec fragment encoder using Morgan fingerprint identifier extraction.

Encodes functional-group/context fragments by extracting Morgan environment identifiers
(radius 0 and radius 1), embedding them into a native D-dimensional space, and projecting
the resulting vector to a trainable 512-D fragment representation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)


def extract_morgan_subgraph_identifiers(
    smiles: str,
    radius: int = 1,
    vocab_size: int = 50000,
) -> List[int]:
    """Extract Morgan fingerprint environment identifier indices for a fragment SMILES.

    Args:
        smiles: Input fragment SMILES string.
        radius: Morgan fingerprint radius (0 and 1).
        vocab_size: Hashing vocabulary size.

    Returns:
        List of integer token indices corresponding to Morgan subgraphs.
    """
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        return [0]

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        # Fallback to character/string hash if invalid
        val = abs(hash(smiles)) % vocab_size
        return [val if val != 0 else 1]

    try:
        fp = rdMolDescriptors.GetMorganFingerprint(mol, radius)
        nonzero = fp.GetNonzeroElements()
        token_indices: List[int] = []
        for feat_id, count in nonzero.items():
            idx = (feat_id % (vocab_size - 1)) + 1  # 1-indexed, reserving 0 for pad
            token_indices.extend([idx] * count)

        return token_indices if token_indices else [0]
    except Exception as e:
        logger.warning(f"Error extracting Morgan fingerprint for '{smiles}': {e}")
        val = abs(hash(smiles)) % vocab_size
        return [val if val != 0 else 1]


class Mol2VecEncoder(nn.Module):
    """Mol2Vec encoder module mapping chemical fragments to 512-D vectors.

    Flow:
      1. Fragment SMILES -> Morgan environment identifiers (radius 0/1)
      2. Identifier lookup + sum pooling -> D-dimensional vector (default D=300)
      3. Linear projection + LayerNorm + Dropout -> 512-D fragment embedding
    """

    def __init__(
        self,
        native_dim: int = 300,
        fragment_dim: int = 512,
        vocab_size: int = 50000,
        radius: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.native_dim = native_dim
        self.fragment_dim = fragment_dim
        self.vocab_size = vocab_size
        self.radius = radius

        # Learnable Morgan identifier embedding bag (sum-pooled per fragment)
        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=native_dim,
            mode="sum",
            padding_idx=0,
        )

        # Trainable projection from native_dim (300) to fragment_dim (512)
        self.projection = nn.Sequential(
            nn.Linear(native_dim, fragment_dim),
            nn.LayerNorm(fragment_dim),
            nn.Dropout(dropout),
        )

        # In-memory fragment embedding cache: smiles -> (512,) tensor
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding_bag.weight, mean=0.0, std=0.1)
        # Ensure padding idx 0 is zeroed
        with torch.no_grad():
            self.embedding_bag.weight[0].fill_(0.0)

        for m in self.projection:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def clear_cache(self) -> None:
        """Clear cached base fragment embeddings."""
        self._embedding_cache.clear()

    def forward_single_fragments(
        self,
        frag_smiles_list: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a flat list of fragment SMILES strings into 512-D vectors."""
        if not frag_smiles_list:
            return torch.zeros((0, self.fragment_dim), device=device)

        flat_indices: List[int] = []
        offsets: List[int] = []
        curr_offset = 0

        for s in frag_smiles_list:
            ids = extract_morgan_subgraph_identifiers(
                s, radius=self.radius, vocab_size=self.vocab_size
            )
            offsets.append(curr_offset)
            flat_indices.extend(ids)
            curr_offset += len(ids)

        indices_t = torch.tensor(flat_indices, dtype=torch.long, device=device)
        offsets_t = torch.tensor(offsets, dtype=torch.long, device=device)

        # Sum-pooled native Mol2Vec embeddings: (K, native_dim)
        native_vecs = self.embedding_bag(indices_t, offsets_t)
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

        # Disable cache if gradients are required through embedding_bag
        is_training = self.training or any(p.requires_grad for p in self.parameters())
        allow_cache = use_cache and not is_training

        unique_to_encode: List[str] = []
        frag_to_idx: Dict[str, int] = {}

        for i in range(B):
            for j in range(N):
                if mask[i, j] > 0.5:
                    frag_str = batched_fragments[i][j]
                    if allow_cache and frag_str in self._embedding_cache:
                        output[i, j] = self._embedding_cache[frag_str].to(device)
                    else:
                        if frag_str not in frag_to_idx:
                            frag_to_idx[frag_str] = len(unique_to_encode)
                            unique_to_encode.append(frag_str)

        if unique_to_encode:
            encoded_unique = self.forward_single_fragments(unique_to_encode, device=device)
            if allow_cache:
                for frag_str, idx in frag_to_idx.items():
                    self._embedding_cache[frag_str] = encoded_unique[idx].detach().cpu()

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
