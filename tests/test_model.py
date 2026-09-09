"""Integration tests for end-to-end CancerComboBRICS forward and backward passes."""

import torch
from cancer_combo_brics.config import ModelConfig
from cancer_combo_brics.model import CancerComboBRICS
from cancer_combo_brics.losses import SurfaceRegressionLoss


def test_full_model_synthetic_forward_backward():
    cfg = ModelConfig(cell_dim=976, fragment_dim=512)
    model = CancerComboBRICS(config=cfg)

    B = 2
    D_A = 8
    D_B = 8

    cell_expr = torch.randn(B, 976)
    frags_A = [
        ["CC(=O)O", "c1ccccc1", "CCO"],
        ["c1ccccc1", "CC(=O)O", ""],
    ]
    mask_A = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    frags_B = [
        ["CCN(CC)CC", "c1ccccc1"],
        ["CC(C)O", "c1ccccc1"],
    ]
    mask_B = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    doses_A = torch.tensor([0.0, 0.01, 0.05, 0.2, 1.0, 3.0, 10.0, 30.0])
    doses_B = torch.tensor([0.0, 0.005, 0.02, 0.1, 0.5, 2.0, 8.0, 25.0])

    y_pred, diag = model(
        cell_expr=cell_expr,
        fragments_A=frags_A,
        mask_A=mask_A,
        fragments_B=frags_B,
        mask_B=mask_B,
        doses_A=doses_A,
        doses_B=doses_B,
        return_diagnostics=True,
    )

    assert y_pred.shape == (B, D_A, D_B), f"Expected ({B}, {D_A}, {D_B}), got {y_pred.shape}"
    assert diag["c"].shape == (B, 512), f"Expected cell (B, 512), got {diag['c'].shape}"
    assert diag["F_A"].shape == (B, 3, 512), f"Expected F_A (B, N, 512), got {diag['F_A'].shape}"
    assert diag["F_B"].shape == (B, 2, 512), f"Expected F_B (B, M, 512), got {diag['F_B'].shape}"
    assert diag["r_AB"].shape == (B, 512), f"Expected r_AB (B, 512), got {diag['r_AB'].shape}"
    assert diag["r_DC"].shape == (B, 1536), f"Expected r_DC (B, 1536), got {diag['r_DC'].shape}"
    assert len(diag["raw_params"]) == 8
    assert diag["raw_params"]["e1_raw"].shape == (B,)
    assert len(diag["params_tuple"]) == 8
    assert diag["Y_hill"].shape == (B, D_A, D_B)
    assert diag["bias"].shape == (B, D_A, D_B)

    assert torch.all(torch.isfinite(y_pred))

    y_target = torch.rand(B, D_A, D_B) * 100.0
    criterion = SurfaceRegressionLoss(loss_type="huber", delta=1.0)
    loss = criterion(y_pred, y_target)

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert torch.isfinite(grad_norm)
    assert grad_norm.item() > 0.0
