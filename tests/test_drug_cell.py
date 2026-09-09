"""Unit tests for DrugCellInteraction module and dimension assertions."""

import torch
from cancer_combo_brics.interaction.drug_cell import DrugCellInteraction


def test_drug_cell_interaction_dimensions():
    dc = DrugCellInteraction(dim=512, mlp_hidden=512)

    B = 4
    r_prime_AB = torch.randn(B, 512)
    c = torch.randn(B, 512)

    r_DC, r_gate, diag = dc(r_prime_AB, c)

    # 1. Check gated representation shape (B, 512)
    assert r_gate.shape == (B, 512)

    # 2. Check final representation shape r_DC in R^1536
    assert r_DC.shape == (B, 1536)

    # 3. Check composition: [r'_AB ; c ; r_gate]
    assert torch.allclose(r_DC[:, :512], r_prime_AB)
    assert torch.allclose(r_DC[:, 512:1024], c)
    assert torch.allclose(r_DC[:, 1024:], r_gate)

    # 4. Check diagnostics
    assert "gate_mean" in diag
    assert "norm_r_DC" in diag
