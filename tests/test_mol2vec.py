"""Unit tests for Mol2Vec encoder and fragment projection."""

import torch
from cancer_combo_brics.encoders.mol2vec_encoder import Mol2VecEncoder
from cancer_combo_brics.encoders.fragment_encoder import FragmentEncoder


def test_mol2vec_native_and_projected_dimensions():
    native_dim = 300
    fragment_dim = 512
    encoder = Mol2VecEncoder(model_path="data/model_300dim.pkl", native_dim=native_dim, fragment_dim=fragment_dim)

    assert encoder.native_dim == 300
    assert encoder.fragment_dim == 512

    frags = ["cNC(C)=O", "cO", "CCCC"]
    device = torch.device("cpu")
    projected = encoder.forward_single_fragments(frags, device=device)

    assert projected.shape == (3, 512)
    assert not torch.isnan(projected).any()
    assert not torch.isinf(projected).any()


def test_encode_drug_fragments_batch_dimensions():
    encoder = FragmentEncoder(model_path="data/model_300dim.pkl", native_dim=300, fragment_dim=512)
    device = torch.device("cpu")

    batched_fragments = [
        ["cNC(C)=O", "cO"],
        ["CCCC", ""],
    ]
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    F = encoder.encode_drug_fragments_batch(batched_fragments, mask=mask, device=device)
    assert F.shape == (2, 2, 512)
    assert not torch.isnan(F).any()
    assert not torch.isinf(F).any()

    # Verify zero vector for padding
    assert torch.all(F[1, 1] == 0.0)
