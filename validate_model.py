"""
Model Validation Command Line Diagnostic Script for CancerCombo.

Tests:
    1. Forward pass execution on CPU and GPU (if available).
    2. Autograd backward pass and finite gradients check across unified parameter MLP.
    3. Arbitrary M x N surface grid support (3x3, 4x4, 5x5, 8x8).
    4. Directional cross-attention & unpadded fragment conditioning flow.
"""

import sys
import torch
import config
from cancer_combo_brics import CancerComboBRICSSymmetric


def test_grid_shape(model, device, grid_shape=(4, 4)):
    M, N = grid_shape
    B = 2
    cell_expr = torch.randn(B, 976, device=device)
    fp_A = torch.randn(B, 3, 2048, device=device)
    mask_A = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool, device=device)
    fp_B = torch.randn(B, 2, 2048, device=device)
    mask_B = torch.tensor([[True, True], [True, False]], dtype=torch.bool, device=device)

    doses_A = torch.tensor([0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2][:M], device=device).unsqueeze(0).repeat(B, 1)
    doses_B = torch.tensor([0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2][:N], device=device).unsqueeze(0).repeat(B, 1)

    Y_pred = model(cell_expr, fp_A, mask_A, fp_B, mask_B, dose_grid=(doses_A, doses_B))

    assert Y_pred.shape == (B, M, N), f"Shape mismatch for grid {grid_shape}: expected ({B}, {M}, {N}), got {tuple(Y_pred.shape)}"
    assert torch.isfinite(Y_pred).all(), f"Non-finite values found in predicted surface for grid {grid_shape}!"
    print(f"  [PASS] Arbitrary Surface Grid {M}x{N} Output Shape: {tuple(Y_pred.shape)}")


def main():
    print("=" * 75)
    print("  CancerCombo — Model Architecture & GPU Safety Audit")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Model on Device: {device}")

    model = CancerComboBRICSSymmetric(
        gene_dim=config.GENE_DIM,
        cell_dim=config.CELL_DIM,
        frag_fp_dim=config.FRAG_FP_DIM,
        d_dim=config.D_DIM,
        num_attn_heads=config.NUM_ATTN_HEADS,
        shared_attn_weights=config.SHARED_ATTN_WEIGHTS
    ).to(device)

    # 1. Test Arbitrary Surface Grids
    print("\n[1] Testing Arbitrary Dose Grid Surfaces (M x N):")
    for shape in [(3, 3), (4, 4), (5, 5), (8, 8)]:
        test_grid_shape(model, device, grid_shape=shape)

    # 2. Test Directional Cross-Attention Forward Execution
    print("\n[2] Testing Directional Cross-Attention & Unified Parameter MLP:")
    B = 2
    cell_expr = torch.randn(B, 976, device=device)
    fp_A = torch.randn(B, 3, 2048, device=device)
    mask_A = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool, device=device)
    fp_B = torch.randn(B, 2, 2048, device=device)
    mask_B = torch.tensor([[True, True], [True, False]], dtype=torch.bool, device=device)
    doses_A = torch.tensor([[0.0, 1e-8, 1e-7, 1e-6], [0.0, 1e-8, 1e-7, 1e-6]], device=device)
    doses_B = torch.tensor([[0.0, 1e-8, 1e-7, 1e-6], [0.0, 1e-8, 1e-7, 1e-6]], device=device)

    model.eval()
    with torch.no_grad():
        Y_AB, params_AB = model(cell_expr, fp_A, mask_A, fp_B, mask_B, dose_grid=(doses_A, doses_B), return_params=True)

    assert Y_AB.shape == (B, 4, 4), f"Surface shape mismatch: expected ({B}, 4, 4), got {tuple(Y_AB.shape)}"
    assert set(params_AB.keys()) == {"e0", "e1", "e2", "e12", "c1", "c2", "h1", "h2", "alpha", "log_c1", "log_c2"}
    print("  [PASS] Directional Cross-Attention & Unified Parameter MLP Output Verified!")

    # 3. Test Backward Pass & Gradient Flow
    print("\n[3] Testing Autograd Backward Pass & Finite Gradients:")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    Y_pred = model(cell_expr, fp_A, mask_A, fp_B, mask_B, dose_grid=(doses_A, doses_B))
    loss = Y_pred.sum()
    optimizer.zero_grad()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for parameter '{name}'"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in parameter '{name}'"

    optimizer.step()
    print("  [PASS] Backward pass completed cleanly. All gradients are finite and valid!")

    print("\n" + "=" * 75)
    print("  CancerCombo Model Diagnostic Validation Successful!")
    print("=" * 75)


if __name__ == "__main__":
    main()
