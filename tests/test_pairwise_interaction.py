"""Unit tests for explicit pairwise fragment interaction module with dual mean + max pooling."""

import pytest
import torch
from tests.conftest import requires_gensim
from cancer_combo_brics.interaction.pairwise_interaction import ExplicitPairwiseFragmentInteraction
from cancer_combo_brics.model import CancerComboBRICS
from cancer_combo_brics.config import ModelConfig


# Test 1: Output shape (B, N_A, 512), (B, N_B, 512) -> (B, 512)
def test_pairwise_output_shape():
    interaction = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512)
    B, N_A, N_B, D = 3, 4, 6, 512
    F_A = torch.randn(B, N_A, D)
    F_B = torch.randn(B, N_B, D)
    mask_A = torch.ones(B, N_A)
    mask_B = torch.ones(B, N_B)
    mask_A[0, 3] = 0.0
    mask_B[1, 4:] = 0.0

    r_AB, diag = interaction(F_A, F_B, mask_A, mask_B)

    assert r_AB.shape == (B, 512)
    assert not torch.isnan(r_AB).any()
    assert not torch.isinf(r_AB).any()
    assert "mean_pool_norm" in diag
    assert "max_pool_norm" in diag
    assert "fused_r_AB_norm" in diag


# Test 2: Mean pooling active
def test_mean_pooling_active():
    interaction = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512)
    F_A1 = torch.ones(1, 2, 512) * 0.5
    F_B1 = torch.ones(1, 2, 512) * 0.5
    F_A2 = torch.ones(1, 2, 512) * 2.0
    F_B2 = torch.ones(1, 2, 512) * 2.0
    mask = torch.ones(1, 2)

    r_AB1, _ = interaction(F_A1, F_B1, mask, mask)
    r_AB2, _ = interaction(F_A2, F_B2, mask, mask)

    assert not torch.allclose(r_AB1, r_AB2)


# Test 3: Max pooling active (capturing unique large values)
def test_max_pooling_active():
    interaction = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512)
    interaction.eval()

    # Case A: uniform baseline inputs
    F_A = torch.ones(1, 3, 512) * 0.1
    F_B = torch.ones(1, 3, 512) * 0.1
    mask = torch.ones(1, 3)

    r_AB_base, diag_base = interaction(F_A, F_B, mask, mask)

    # Case B: introduce a single fragment pair with a large distinct feature spike
    F_A_spike = F_A.clone()
    F_A_spike[0, 1] = 50.0  # Spike at fragment index 1

    r_AB_spike, diag_spike = interaction(F_A_spike, F_B, mask, mask)

    # The outputs must differ when a spike is introduced
    # (max pool captures distinct feature extremes; result must change)
    assert not torch.allclose(r_AB_base, r_AB_spike, atol=1e-4), (
        "r_AB should differ between baseline and spiked input because max pooling "
        "should capture the large fragment feature extreme."
    )
    # The fused r_AB norm should also differ
    assert abs(diag_base["fused_r_AB_norm"] - diag_spike["fused_r_AB_norm"]) > 0.0, (
        "fused_r_AB_norm must change when a large feature spike is introduced."
    )


# Test 4: Mask correctness (padded pair values set to large values do not alter r_AB)
def test_mask_correctness():
    interaction = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512)
    interaction.eval()

    B, N_A, N_B, D = 2, 3, 3, 512
    F_A = torch.randn(B, N_A, D)
    F_B = torch.randn(B, N_B, D)

    mask_A = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])  # Item 0 has 3rd frag padded
    mask_B = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    r_AB_orig, _ = interaction(F_A, F_B, mask_A, mask_B)

    # Modify padded fragment features to extreme values (+10,000.0)
    F_A_mod = F_A.clone()
    F_B_mod = F_B.clone()
    F_A_mod[0, 2] = 10000.0
    F_B_mod[1, 2] = 10000.0

    r_AB_mod, _ = interaction(F_A_mod, F_B_mod, mask_A, mask_B)

    # Output MUST remain identical because padded positions are masked out
    torch.testing.assert_close(r_AB_mod, r_AB_orig, rtol=1e-5, atol=1e-5)


# Test 5: Permutation invariance
def test_permutation_invariance():
    interaction = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512)
    interaction.eval()

    F_A = torch.randn(1, 3, 512)
    F_B = torch.randn(1, 3, 512)
    mask = torch.ones(1, 3)

    r_AB_orig, _ = interaction(F_A, F_B, mask, mask)

    # Permute fragments of Drug A (indices 0, 1, 2 -> 2, 0, 1)
    F_A_perm = F_A[:, [2, 0, 1], :]
    r_AB_perm, _ = interaction(F_A_perm, F_B, mask, mask)

    torch.testing.assert_close(r_AB_perm, r_AB_orig, rtol=1e-5, atol=1e-5)


# Test 6: Zero-valid-pair safety
def test_zero_valid_pair_safety():
    interaction = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512)

    B, N_A, N_B, D = 2, 2, 2, 512
    F_A = torch.randn(B, N_A, D)
    F_B = torch.randn(B, N_B, D)

    # Batch item 1 has ALL masks set to 0.0 (zero valid pairs)
    mask_A = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    mask_B = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    r_AB, _ = interaction(F_A, F_B, mask_A, mask_B)

    assert r_AB.shape == (B, 512)
    assert not torch.isnan(r_AB).any()
    assert not torch.isinf(r_AB).any()
    # Batch item 1 with 0 valid pairs must produce clean zero vector
    assert torch.all(r_AB[1] == 0.0)


# Test 7: Gradient flow to both interaction MLP and fusion projection
def test_gradient_flow():
    interaction = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512)

    F_A = torch.randn(2, 2, 512, requires_grad=True)
    F_B = torch.randn(2, 2, 512, requires_grad=True)
    mask = torch.ones(2, 2)

    r_AB, _ = interaction(F_A, F_B, mask, mask)
    loss = r_AB.sum()
    loss.backward()

    # Check gradients in interaction MLP
    mlp_grad_found = False
    for p in interaction.interaction_mlp.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0:
            mlp_grad_found = True
            assert not torch.isnan(p.grad).any()
    assert mlp_grad_found, "Gradients failed to reach interaction_mlp!"

    # Check gradients in fusion projection
    fusion_grad_found = False
    for p in interaction.fusion_projection.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0:
            fusion_grad_found = True
            assert not torch.isnan(p.grad).any()
    assert fusion_grad_found, "Gradients failed to reach fusion_projection!"


# Test 8: Downstream shape compatibility
@requires_gensim
def test_downstream_shape_compatibility():
    model = CancerComboBRICS(ModelConfig(mol2vec_model_path="data/model_300dim.pkl"))
    B = 2
    cell_expr = torch.randn(B, 976)
    frags_A = [["c1ccccc1", "CC(=O)O"], ["CCO"]]
    mask_A = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    frags_B = [["c1ncccn1"], ["c1ccccc1"]]
    mask_B = torch.tensor([[1.0], [1.0]])
    doses = torch.tensor([[0.0, 0.1, 1.0, 10.0]] * B)

    Y_pred, diag = model(cell_expr, frags_A, mask_A, frags_B, mask_B, doses, doses, return_diagnostics=True)

    assert Y_pred.shape == (B, 4, 4)
    assert diag["r_AB"].shape == (B, 512)
    assert diag["r_DC"].shape == (B, 1536)
    assert len(diag["raw_params"]) == 8


# Test 9: Numerical stability on extreme inputs
def test_numerical_stability():
    interaction = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512)

    F_A = torch.randn(2, 4, 512) * 100.0
    F_B = torch.randn(2, 4, 512) * 100.0
    mask = torch.ones(2, 4)

    r_AB, _ = interaction(F_A, F_B, mask, mask)
    assert not torch.isnan(r_AB).any()
    assert not torch.isinf(r_AB).any()
