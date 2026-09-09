"""Fragment encoder wrapping Mol2VecEncoder and projecting to 512-D."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from cancer_combo_brics.encoders.mol2vec_encoder import Mol2VecEncoder


class FragmentEncoder(nn.Module):
    """Encodes functional-group/context fragments via shared Mol2VecEncoder into 512-D.

    Ensures that Drug A and Drug B share the exact same weights and projection layer.
    """

    def __init__(
        self,
        mol2vec: Optional[Mol2VecEncoder] = None,
        native_dim: int = 300,
        fragment_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mol2vec = mol2vec if mol2vec is not None else Mol2VecEncoder(
            native_dim=native_dim,
            fragment_dim=fragment_dim,
            dropout=dropout,
        )
        self.native_dim = self.mol2vec.native_dim
        self.fragment_dim = self.mol2vec.fragment_dim

    def clear_cache(self) -> None:
        """Clear cached fragment embeddings."""
        self.mol2vec.clear_cache()

    def forward(
        self,
        batched_fragments: List[List[str]],
        mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a batch of drug fragments."""
        return self.mol2vec.encode_drug_fragments_batch(
            batched_fragments=batched_fragments,
            mask=mask,
            device=device,
        )

    def encode_drug_fragments_batch(
        self,
        batched_fragments: List[List[str]],
        mask: torch.Tensor,
        device: torch.device,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Encode a batch of drug fragments into 512-D tensors."""
        return self.mol2vec.encode_drug_fragments_batch(
            batched_fragments=batched_fragments,
            mask=mask,
            device=device,
            use_cache=use_cache,
        )
