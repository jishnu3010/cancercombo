"""Programmatic architecture verification tests ensuring compliance with design rules."""

import pytest
import torch
import torch.nn as nn
from tests.conftest import requires_gensim
from cancer_combo_brics.config import ModelConfig
from cancer_combo_brics.model import CancerComboBRICS

pytestmark = requires_gensim


def test_deactivation_of_removed_modules():
    config = ModelConfig()
    model = CancerComboBRICS(config)

    # 1. MoLFormer must NOT be an attribute of active model
    assert not hasattr(model, "molformer"), "MoLFormer backbone should be removed from active model."

    # 2. FiLM must NOT be an attribute of active model
    assert not hasattr(model, "film"), "FiLM module should be removed from active model."

    # 3. Cross-Attention must NOT be an attribute of active model
    assert not hasattr(model, "cross_attention"), "Cross-attention module should be removed from active model."

    # Verify no Q/K/V linear projections exist in the model's named modules
    for name, module in model.named_modules():
        assert "q_proj" not in name, f"Found Q projection in active module: {name}"
        assert "k_proj" not in name, f"Found K projection in active module: {name}"
        assert "v_proj" not in name, f"Found V projection in active module: {name}"
        assert not isinstance(module, nn.MultiheadAttention), f"Found MultiheadAttention in active module: {name}"


def test_drug_cell_dimensions_and_end_to_end_shape():
    config = ModelConfig()
    model = CancerComboBRICS(config)

    B = 2
    cell_expr = torch.randn(B, 976)
    fragments_A = [["cNC(C)=O", "cO"], ["CCCC"]]
    mask_A = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    fragments_B = [["c1ccccc1"], ["cNC(C)=O"]]
    mask_B = torch.tensor([[1.0], [1.0]])
    doses_A = torch.tensor([[0.0, 0.06, 0.6, 6.0], [0.0, 0.06, 0.6, 6.0]])
    doses_B = torch.tensor([[0.0, 0.05, 0.5, 5.0], [0.0, 0.05, 0.5, 5.0]])

    Y_pred, diag = model(
        cell_expr=cell_expr,
        fragments_A=fragments_A,
        mask_A=mask_A,
        fragments_B=fragments_B,
        mask_B=mask_B,
        doses_A=doses_A,
        doses_B=doses_B,
        return_diagnostics=True,
    )

    assert Y_pred.shape == (B, 4, 4)
    assert diag["c"].shape == (B, 512)
    assert diag["r_AB"].shape == (B, 512)
    assert diag["r_DC"].shape == (B, 1536)

    # Verify 8 prediction heads shapes
    assert len(diag["raw_params"]) == 8
    assert diag["raw_params"]["e1_raw"].shape == (B,)


def test_gradient_flow_to_all_trainable_components():
    config = ModelConfig()
    model = CancerComboBRICS(config)

    B = 2
    cell_expr = torch.randn(B, 976, requires_grad=True)
    fragments_A = [["cNC(C)=O"], ["cO"]]
    mask_A = torch.ones(B, 1)
    fragments_B = [["c1ccccc1"], ["CCCC"]]
    mask_B = torch.ones(B, 1)
    doses_A = torch.tensor([[0.0, 0.1, 1.0, 10.0], [0.0, 0.1, 1.0, 10.0]])
    doses_B = torch.tensor([[0.0, 0.1, 1.0, 10.0], [0.0, 0.1, 1.0, 10.0]])
    y_true = torch.ones(B, 4, 4) * 80.0

    Y_pred, _ = model(
        cell_expr=cell_expr,
        fragments_A=fragments_A,
        mask_A=mask_A,
        fragments_B=fragments_B,
        mask_B=mask_B,
        doses_A=doses_A,
        doses_B=doses_B,
    )

    loss = ((Y_pred - y_true) ** 2).mean()
    loss.backward()

    # Check gradients in key active components
    trainable_submodules = [
        ("cell_encoder", model.cell_encoder),
        ("fragment_encoder", model.fragment_encoder),
        ("pairwise_interaction", model.pairwise_interaction),
        ("drug_cell", model.drug_cell),
        ("parameter_heads", model.parameter_heads),
        ("dose_bias", model.dose_bias),
    ]

    for name, submod in trainable_submodules:
        has_grad = False
        for p in submod.parameters():
            if p.grad is not None and torch.norm(p.grad) > 0:
                has_grad = True
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"
        assert has_grad, f"Submodule {name} received no non-zero gradients!"
