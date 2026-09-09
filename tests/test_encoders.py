"""Unit tests for CellEncoder and FragmentEncoder."""

import pytest
import torch
from cancer_combo_brics.encoders.cell_encoder import CellEncoder
from cancer_combo_brics.encoders.fragment_encoder import FragmentEncoder
from cancer_combo_brics.encoders.mol2vec_encoder import Mol2VecEncoder


def test_cell_encoder_dimensions():
    encoder = CellEncoder(in_dim=976, hidden_dim=512)
    x = torch.randn(4, 976)
    c = encoder(x)
    assert c.shape == (4, 512)
    assert not torch.isnan(c).any()


def test_fragment_encoder_mol2vec():
    mol2vec = Mol2VecEncoder(native_dim=300, fragment_dim=512)
    frag_encoder = FragmentEncoder(mol2vec=mol2vec, fragment_dim=512)

    batch_frags = [
        ["cNC(C)=O", "cO", "CCO", ""],
        ["c1ccccc1", "cNC(C)=O", "", ""],
    ]
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

    device = torch.device("cpu")
    F = frag_encoder.encode_drug_fragments_batch(batch_frags, mask, device=device)

    assert F.shape == (2, 4, 512)
    assert torch.all(F[0, 3] == 0.0)
    assert torch.all(F[1, 2] == 0.0)
    assert torch.all(F[1, 3] == 0.0)
    assert torch.any(F[0, 0] != 0.0)
