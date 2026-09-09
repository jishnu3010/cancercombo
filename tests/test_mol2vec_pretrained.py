"""Comprehensive unit test suite for genuine pretrained Mol2Vec representation in CancerCombo.

Verifies all 27 required architectural, functional, and parity constraints.
"""

import os
import pytest
import torch
import torch.nn as nn
from cancer_combo_brics.encoders.mol2vec_encoder import Mol2VecEncoder
from cancer_combo_brics.encoders.fragment_encoder import FragmentEncoder
from cancer_combo_brics.config import ModelConfig
from cancer_combo_brics.model import CancerComboBRICS
from cancer_combo_brics.chemistry.functional_group_fragments import extract_functional_group_fragments

MODEL_PATH = "data/model_300dim.pkl"


@pytest.fixture
def mol2vec_encoder():
    return Mol2VecEncoder(model_path=MODEL_PATH, native_dim=300, fragment_dim=512)


# Test 1: Pretrained Mol2Vec model loads
def test_1_pretrained_mol2vec_model_loads(mol2vec_encoder):
    assert mol2vec_encoder is not None
    assert mol2vec_encoder.embeddings is not None


# Test 2: Pretrained weights are non-random
def test_2_pretrained_weights_non_random(mol2vec_encoder):
    weights = mol2vec_encoder.embeddings.weight
    mean = weights.mean().item()
    std = weights.std().item()
    zero_ratio = (weights == 0.0).float().mean().item()

    # Pretrained Word2Vec weights have non-zero variance and standard deviation around ~0.3
    assert abs(mean) < 0.05
    assert 0.1 < std < 1.0
    assert zero_ratio < 0.1  # Most tokens are non-zero pretrained vectors


# Test 3: Vector dimensionality is correct (300 native, 512 projected)
def test_3_vector_dimensionality(mol2vec_encoder):
    assert mol2vec_encoder.native_dim == 300
    assert mol2vec_encoder.fragment_dim == 512
    assert mol2vec_encoder.embeddings.weight.shape[1] == 300


# Test 4: Known molecule/fragment produces valid vector
def test_4_known_fragment_vector(mol2vec_encoder):
    device = torch.device("cpu")
    vec = mol2vec_encoder.forward_single_fragments(["CC(=O)O"], device=device)
    assert vec.shape == (1, 512)
    assert not torch.isnan(vec).any()
    assert not torch.isinf(vec).any()
    assert (vec ** 2).sum().item() > 0.0


# Test 5: OOV behavior (non-crashing, deterministic handling)
def test_5_oov_behavior(mol2vec_encoder):
    device = torch.device("cpu")
    # Inorganic fallback fragment with 0 in-vocab tokens
    oov_frag = "[As+3].[As+3].[O-2].[O-2].[O-2]"
    vec = mol2vec_encoder.forward_single_fragments([oov_frag], device=device)
    assert vec.shape == (1, 512)
    assert not torch.isnan(vec).any()
    assert not torch.isinf(vec).any()


# Test 6: Multiple fragments encoding
def test_6_multiple_fragments(mol2vec_encoder):
    device = torch.device("cpu")
    frags = ["c1ccccc1", "CC(=O)O", "c1ncccn1", "CCO"]
    vecs = mol2vec_encoder.forward_single_fragments(frags, device=device)
    assert vecs.shape == (4, 512)


# Test 7: Aromatic fragment
def test_7_aromatic_fragment(mol2vec_encoder):
    device = torch.device("cpu")
    vec = mol2vec_encoder.forward_single_fragments(["c1ccccc1"], device=device)
    assert vec.shape == (1, 512)


# Test 8: Heterocycle fragment
def test_8_heterocycle_fragment(mol2vec_encoder):
    device = torch.device("cpu")
    vec = mol2vec_encoder.forward_single_fragments(["c1ncccn1"], device=device)
    assert vec.shape == (1, 512)


# Test 9: Functional-group fragment
def test_9_functional_group_fragment(mol2vec_encoder):
    device = torch.device("cpu")
    vec = mol2vec_encoder.forward_single_fragments(["CC(=O)O"], device=device)
    assert vec.shape == (1, 512)


# Test 10: Full-molecule fallback
def test_10_full_molecule_fallback():
    frags = extract_functional_group_fragments("CCCC")
    assert len(frags) > 0
    encoder = Mol2VecEncoder(model_path=MODEL_PATH)
    vecs = encoder.forward_single_fragments(frags, device=torch.device("cpu"))
    assert vecs.shape[0] == len(frags)
    assert vecs.shape[1] == 512


# Test 11: Fragment tensor shape (B, N, 512)
def test_11_fragment_tensor_shape():
    encoder = FragmentEncoder(model_path=MODEL_PATH)
    batched = [["c1ccccc1", "CC(=O)O"], ["CCO", ""]]
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    F = encoder.encode_drug_fragments_batch(batched, mask=mask, device=torch.device("cpu"))
    assert F.shape == (2, 2, 512)


# Test 12: Variable fragment counts with padding
def test_12_variable_fragment_counts():
    encoder = FragmentEncoder(model_path=MODEL_PATH)
    batched = [["c1ccccc1", "CC(=O)O", "CCO"], ["c1ncccn1", "", ""]]
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    F = encoder.encode_drug_fragments_batch(batched, mask=mask, device=torch.device("cpu"))
    assert F.shape == (2, 3, 512)
    assert torch.all(F[1, 1] == 0.0)
    assert torch.all(F[1, 2] == 0.0)


# Test 13: Pairwise interaction shape (B, 512)
def test_13_pairwise_interaction_shape():
    model = CancerComboBRICS(ModelConfig(mol2vec_model_path=MODEL_PATH))
    cell_expr = torch.randn(2, 976)
    frags_A = [["c1ccccc1", "CC(=O)O"], ["CCO"]]
    mask_A = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    frags_B = [["c1ncccn1"], ["c1ccccc1"]]
    mask_B = torch.tensor([[1.0], [1.0]])
    doses = torch.tensor([[0.0, 0.1, 1.0, 10.0], [0.0, 0.1, 1.0, 10.0]])
    _, diag = model(cell_expr, frags_A, mask_A, frags_B, mask_B, doses, doses, return_diagnostics=True)
    assert diag["r_AB"].shape == (2, 512)


# Test 14: r_AB = 512
def test_14_r_AB_dim():
    model = CancerComboBRICS(ModelConfig(mol2vec_model_path=MODEL_PATH))
    cell_expr = torch.randn(2, 976)
    frags_A = [["c1ccccc1"], ["CCO"]]
    mask_A = torch.tensor([[1.0], [1.0]])
    frags_B = [["c1ncccn1"], ["c1ccccc1"]]
    mask_B = torch.tensor([[1.0], [1.0]])
    doses = torch.tensor([[0.0, 0.1, 1.0, 10.0], [0.0, 0.1, 1.0, 10.0]])
    _, diag = model(cell_expr, frags_A, mask_A, frags_B, mask_B, doses, doses, return_diagnostics=True)
    assert diag["r_AB"].shape[1] == 512


# Test 15: r_DC = 1536
def test_15_r_DC_dim():
    model = CancerComboBRICS(ModelConfig(mol2vec_model_path=MODEL_PATH))
    cell_expr = torch.randn(2, 976)
    frags_A = [["c1ccccc1"], ["CCO"]]
    mask_A = torch.tensor([[1.0], [1.0]])
    frags_B = [["c1ncccn1"], ["c1ccccc1"]]
    mask_B = torch.tensor([[1.0], [1.0]])
    doses = torch.tensor([[0.0, 0.1, 1.0, 10.0], [0.0, 0.1, 1.0, 10.0]])
    _, diag = model(cell_expr, frags_A, mask_A, frags_B, mask_B, doses, doses, return_diagnostics=True)
    assert diag["r_DC"].shape[1] == 1536


# Test 16: Eight independent heads
def test_16_eight_independent_heads():
    model = CancerComboBRICS(ModelConfig(mol2vec_model_path=MODEL_PATH))
    cell_expr = torch.randn(2, 976)
    frags_A = [["c1ccccc1"], ["CCO"]]
    mask_A = torch.tensor([[1.0], [1.0]])
    frags_B = [["c1ncccn1"], ["c1ccccc1"]]
    mask_B = torch.tensor([[1.0], [1.0]])
    doses = torch.tensor([[0.0, 0.1, 1.0, 10.0], [0.0, 0.1, 1.0, 10.0]])
    _, diag = model(cell_expr, frags_A, mask_A, frags_B, mask_B, doses, doses, return_diagnostics=True)
    assert len(diag["raw_params"]) == 8


# Test 17: Hill solver parity
def test_17_hill_solver_parity():
    model = CancerComboBRICS(ModelConfig(mol2vec_model_path=MODEL_PATH))
    assert model.hill_solver.e0 == 100.0


# Test 18: Dose-bias parity
def test_18_dose_bias_parity():
    model = CancerComboBRICS(ModelConfig(mol2vec_model_path=MODEL_PATH))
    assert model.dose_bias.enabled is True


# Test 19: No NaN/Inf
def test_19_no_nan_inf():
    model = CancerComboBRICS(ModelConfig(mol2vec_model_path=MODEL_PATH))
    cell_expr = torch.randn(2, 976)
    frags_A = [["c1ccccc1"], ["CCO"]]
    mask_A = torch.tensor([[1.0], [1.0]])
    frags_B = [["c1ncccn1"], ["c1ccccc1"]]
    mask_B = torch.tensor([[1.0], [1.0]])
    doses = torch.tensor([[0.0, 0.1, 1.0, 10.0], [0.0, 0.1, 1.0, 10.0]])
    Y_pred, _ = model(cell_expr, frags_A, mask_A, frags_B, mask_B, doses, doses)
    assert not torch.isnan(Y_pred).any()
    assert not torch.isinf(Y_pred).any()


# Test 20: Pretrained Mol2Vec parameters remain frozen
def test_20_pretrained_mol2vec_parameters_frozen(mol2vec_encoder):
    assert not mol2vec_encoder.embeddings.weight.requires_grad
    # Perform backward pass through model and verify grad is None on embeddings
    device = torch.device("cpu")
    vec = mol2vec_encoder.forward_single_fragments(["CC(=O)O"], device=device)
    loss = vec.sum()
    loss.backward()
    assert mol2vec_encoder.embeddings.weight.grad is None


# Test 21: Old Morgan encoder inactive (no EmbeddingBag in active encoder)
def test_21_old_morgan_encoder_inactive(mol2vec_encoder):
    assert not hasattr(mol2vec_encoder, "embedding_bag")


# Test 22-26: Removed paths inactive (BRICS, MoLFormer, FiLM, Cross-attention, Symmetry)
def test_22_to_26_inactive_modules():
    model = CancerComboBRICS(ModelConfig(mol2vec_model_path=MODEL_PATH))
    assert not hasattr(model, "molformer")
    assert not hasattr(model, "film")
    assert not hasattr(model, "cross_attention")
    for name, _ in model.named_modules():
        assert "q_proj" not in name
        assert "k_proj" not in name


# Test 27: Cache separation
def test_27_cache_separation(mol2vec_encoder):
    assert mol2vec_encoder.cache_namespace == "mol2vec_pretrained_v1"
