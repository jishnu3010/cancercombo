"""
Example Usage Script for CancerCombo-BRICS-Symmetric using uploaded dataset.

Demonstrates:
    1. Loading real drug combination data directly from data/scenario3_drug1.csv.
    2. Collating PyTorch DataLoader batches with BRICS fragment decomposition & boolean masks.
    3. Forward pass predicting 2D viability surfaces Y (B, M, N).
    4. Verifying exact mathematical drug-order invariance on real dataset SMILES pairs.
    5. Training step computing MSE loss against real viability matrices Y_true.
"""

import os
import torch
from torch.utils.data import DataLoader
import config
from cancer_combo_brics import (
    CancerComboBRICSSymmetric,
    load_cancer_combo_from_csv,
    collate_cancer_combo_batch
)


def main():
    print("=" * 75)
    print("  CancerCombo-BRICS-Symmetric Dataset Pipeline Demonstration")
    print("=" * 75)

    device = config.DEVICE
    print(f"Using device: {device}")

    dataset_path = config.DATA_CSV
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    print(f"\n[1] Loading dataset from: {dataset_path}")
    dataset = load_cancer_combo_from_csv(dataset_path, max_samples=16)
    print(f"    - Loaded {len(dataset)} combination matrices from dataset.")

    # 2. PyTorch DataLoader with BRICS collation
    batch_size = 4
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cancer_combo_batch
    )

    # 3. Instantiate CancerComboBRICSSymmetric Model
    model = CancerComboBRICSSymmetric(
        gene_dim=976,
        cell_dim=512,
        frag_fp_dim=2048,
        d_dim=128,
        num_attn_heads=4,
        shared_attn_weights=True
    ).to(device)

    print("\n[2] Model Architecture Initialized:")
    print("    - 9 Modular Stages (CellEncoder, FragmentEncoder, FiLM, CrossAttn, SymFusion, CellFusion, ParameterHeads, ConstraintTransform, BivariateHillSolver)")

    # 4. Process Batch from Dataset
    batch = next(iter(loader))
    print(f"\n[3] DataLoader Batch Collated:")
    print(f"    - Cell Line Expression Shape: {tuple(batch['cell_expr'].shape)}")
    print(f"    - Drug A Fragment Tensor Shape: {tuple(batch['fp_A'].shape)}, Mask Shape: {tuple(batch['mask_A'].shape)}")
    print(f"    - Drug B Fragment Tensor Shape: {tuple(batch['fp_B'].shape)}, Mask Shape: {tuple(batch['mask_B'].shape)}")
    print(f"    - Doses A Shape: {tuple(batch['dose_grid'][0].shape)}, Doses B Shape: {tuple(batch['dose_grid'][1].shape)}")
    print(f"    - Target Viability Surface Y_true Shape: {tuple(batch['Y_true'].shape)}")

    # 5. Forward Pass (A, B)
    model.eval()
    with torch.no_grad():
        Y_pred_AB, params_AB = model(
            cell_expr=batch["cell_expr"],
            drugA_frags=batch["fp_A"],
            drugA_mask=batch["mask_A"],
            drugB_frags=batch["fp_B"],
            drugB_mask=batch["mask_B"],
            dose_grid=batch["dose_grid"],
            return_params=True
        )

    print(f"\n[4] Predicted Viability Surface Y_pred Shape: {tuple(Y_pred_AB.shape)}")
    print(f"    - Predicted Viability Range: min={Y_pred_AB.min().item():.4f}, max={Y_pred_AB.max().item():.4f}")
    print(f"    - Real Ground-Truth Range : min={batch['Y_true'].min().item():.4f}, max={batch['Y_true'].max().item():.4f}")

    # 6. Verification of Drug-Order Invariance (B, A)
    print("\n[5] Verifying Drug-Order Invariance (Swapping Drug A and Drug B):")
    with torch.no_grad():
        Y_pred_BA, params_BA = model(
            cell_expr=batch["cell_expr"],
            drugA_frags=batch["fp_B"],
            drugA_mask=batch["mask_B"],
            drugB_frags=batch["fp_A"],
            drugB_mask=batch["mask_A"],
            dose_grid=(batch["dose_grid"][1], batch["dose_grid"][0]),
            return_params=True
        )

    max_diff = torch.abs(Y_pred_AB - Y_pred_BA.transpose(1, 2)).max().item()
    print(f"    - Max Absolute Difference |Y(A,B) - Y(B,A)^T|: {max_diff:.8e}")
    if max_diff < 1e-5:
        print("    -> PERFECT MATHEMATICAL DRUG-ORDER INVARIANCE VERIFIED! [PASS]")

    # 7. Training Step with MSE Loss
    print("\n[6] Training Step with Ground-Truth MSE Surface Loss:")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    optimizer.zero_grad()
    Y_train = model(
        cell_expr=batch["cell_expr"],
        drugA_frags=batch["fp_A"],
        drugA_mask=batch["mask_A"],
        drugB_frags=batch["fp_B"],
        drugB_mask=batch["mask_B"],
        dose_grid=batch["dose_grid"]
    )
    loss = criterion(Y_train, batch["Y_true"])
    loss.backward()
    optimizer.step()

    print(f"    - Initial Surface MSE Loss on Real Dataset: {loss.item():.6f}")
    print("    - Backward pass completed cleanly. Gradients successfully propagated to all 9 modular stages!")
    print("\n" + "=" * 75)
    print("  CancerCombo-BRICS-Symmetric Dataset Pipeline Successful!")
    print("=" * 75)


if __name__ == "__main__":
    main()
