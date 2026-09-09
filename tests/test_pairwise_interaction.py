"""Unit tests for explicit pairwise fragment interaction module."""

import torch
from cancer_combo_brics.interaction.pairwise_interaction import ExplicitPairwiseFragmentInteraction


def test_pairwise_interaction_variable_counts():
    interaction = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512)

    B = 3
    N_A = 4
    N_B = 6
    D = 512

    F_A = torch.randn(B, N_A, D)
    F_B = torch.randn(B, N_B, D)

    mask_A = torch.ones(B, N_A)
    mask_B = torch.ones(B, N_B)
    mask_A[0, 3] = 0.0  # pad
    mask_B[1, 4:] = 0.0  # pad 2 items

    r_AB, diag = interaction(F_A, F_B, mask_A, mask_B)

    assert r_AB.shape == (B, 512)
    assert not torch.isnan(r_AB).any()
    assert not torch.isinf(r_AB).any()
    assert "norm_r_AB" in diag


def test_one_fragment_cases():
    interaction = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512)

    B = 2
    N_A = 1
    N_B = 1
    D = 512

    F_A = torch.randn(B, N_A, D)
    F_B = torch.randn(B, N_B, D)

    mask_A = torch.ones(B, N_A)
    mask_B = torch.ones(B, N_B)

    r_AB, _ = interaction(F_A, F_B, mask_A, mask_B)

    assert r_AB.shape == (B, 512)
    assert not torch.isnan(r_AB).any()
