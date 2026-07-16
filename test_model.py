import torch
import pytest
from config import ModelConfig
from cancercombo import CancerCombo

@pytest.mark.parametrize("enable_dd_attn", [True, False])
def test_full_model_forward_and_backward(enable_dd_attn):
    """Test full forward pass, bounds checking, and backpropagation gradients."""
    config = ModelConfig(
        d_model=256, n_heads=4, d_ff=512, dropout=0.1,
        molformer_in_dim=768, morgan_in_dim=2048, descriptor_in_dim=200,
        cell_in_dim=20000, use_pathway_projection=True, n_pathways=300,
        molformer_model_name="ibm/MoLFormer-XL-CIMA-100M", use_pretrained_molformer=False,
        enable_drug_drug_attention=enable_dd_attn, use_symmetric_fusion=True,
        e_min=0.0, e_max=100.0, c_min=1e-6, c_max=1e3, h_min=0.1, h_max=10.0,
        alpha_min=1e-4, alpha_max=100.0
    )
    model = CancerCombo(config)
    
    batch_size = 2
    M, N = 5, 5
    
    drug_a_ids = torch.randint(1, 10, (batch_size, 128))
    drug_a_mask = torch.ones(batch_size, 128)
    drug_a_morgan = torch.randn(batch_size, 2048)
    drug_a_desc = torch.randn(batch_size, 200)
    
    drug_b_ids = torch.randint(1, 10, (batch_size, 128))
    drug_b_mask = torch.ones(batch_size, 128)
    drug_b_morgan = torch.randn(batch_size, 2048)
    drug_b_desc = torch.randn(batch_size, 200)
    
    cell_line = torch.randn(batch_size, 20000)
    doses_a = torch.randn(batch_size, M).abs()
    doses_b = torch.randn(batch_size, N).abs()
    
    # Forward pass
    y_pred, params = model(
        drug_a_ids, drug_a_mask, drug_a_morgan, drug_a_desc,
        drug_b_ids, drug_b_mask, drug_b_morgan, drug_b_desc,
        cell_line, doses_a, doses_b
    )
    
    assert y_pred.shape == (batch_size, M, N)
    
    e1, e2, e3, c1, c2, h1, h2, alpha = params
    assert e1.shape == (batch_size, 1)
    
    assert (e1 >= config.e_min).all() and (e1 <= config.e_max).all()
    assert (c1 >= config.c_min).all() and (c1 <= config.c_max).all()
    assert (h1 >= config.h_min).all() and (h1 <= config.h_max).all()
    assert (alpha >= config.alpha_min).all() and (alpha <= config.alpha_max).all()
    
    # Backward pass
    loss = y_pred.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not enable_dd_attn and "drug_drug_attn" in name:
                continue
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient is NaN!"


def test_permutation_invariance():
    """Test that model output is permutation invariant when inputs are swapped."""
    config = ModelConfig(
        d_model=256, n_heads=4, d_ff=512, dropout=0.1,
        molformer_in_dim=768, morgan_in_dim=2048, descriptor_in_dim=200,
        cell_in_dim=20000, use_pathway_projection=True, n_pathways=300,
        molformer_model_name="ibm/MoLFormer-XL-CIMA-100M", use_pretrained_molformer=False,
        enable_drug_drug_attention=False, use_symmetric_fusion=True,
        e_min=0.0, e_max=100.0, c_min=1e-6, c_max=1e3, h_min=0.1, h_max=10.0,
        alpha_min=1e-4, alpha_max=100.0
    )
    model = CancerCombo(config)
    model.eval()
    
    batch_size = 1
    M, N = 4, 4
    
    ids_a = torch.randint(1, 10, (batch_size, 128))
    mask_a = torch.ones(batch_size, 128)
    morgan_a = torch.randn(batch_size, 2048)
    desc_a = torch.randn(batch_size, 200)
    
    ids_b = torch.randint(1, 10, (batch_size, 128))
    mask_b = torch.ones(batch_size, 128)
    morgan_b = torch.randn(batch_size, 2048)
    desc_b = torch.randn(batch_size, 200)
    
    cell_line = torch.randn(batch_size, 20000)
    doses_a = torch.randn(batch_size, M).abs()
    doses_b = torch.randn(batch_size, N).abs()
    
    with torch.no_grad():
        y_pred_ab, _ = model(
            ids_a, mask_a, morgan_a, desc_a,
            ids_b, mask_b, morgan_b, desc_b,
            cell_line, doses_a, doses_b
        )
        
        y_pred_ba, _ = model(
            ids_b, mask_b, morgan_b, desc_b,
            ids_a, mask_a, morgan_a, desc_a,
            cell_line, doses_b, doses_a
        )
        
    assert torch.allclose(y_pred_ab, y_pred_ba.transpose(1, 2), atol=1e-5)
