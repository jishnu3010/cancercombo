"""
Deterministic Inference Script for CancerCombo.

Accepts:
    - Cell Line ID (e.g. 'A549', '7860', 'HCT116')
    - Drug A SMILES string
    - Drug B SMILES string
    - Dose grid vectors for Drug A and Drug B

Predicts complete 2D dose-response viability surface Y (M x N).
Uses exact same cell expression, BRICS preprocessing, and model transformations as training.
"""

import os
import sys
import json
import torch
import numpy as np

import config
from cancer_combo_brics import (
    CancerComboBRICSSymmetric,
    collate_brics_fragments
)
from cancer_combo_brics.cell_expression import CellExpressionLoader


def predict_combination_surface(
    cell_id: str,
    smiles_A: str,
    smiles_B: str,
    doses_A: List[float],
    doses_B: List[float],
    model_checkpoint: str = config.BEST_MODEL_PATH,
    cell_expr_csv: str = os.path.join("data", "cell_line_gene_expr.csv"),
    device: str = "cpu"
) -> np.ndarray:
    """
    Predicts 2D viability surface Y for a single cell line and drug pair.
    """
    # 1. Load Cell Expression
    expr_loader = CellExpressionLoader(csv_path=cell_expr_csv, gene_dim=config.GENE_DIM)
    cell_expr = expr_loader.get_cell_expression(cell_id).unsqueeze(0).to(device)

    # 2. Collate BRICS Fragments
    fp_A, mask_A, frags_A = collate_brics_fragments([smiles_A], n_bits=config.FRAG_FP_DIM, device=device)
    fp_B, mask_B, frags_B = collate_brics_fragments([smiles_B], n_bits=config.FRAG_FP_DIM, device=device)

    dose_A_tensor = torch.tensor(doses_A, dtype=torch.float32, device=device).unsqueeze(0)
    dose_B_tensor = torch.tensor(doses_B, dtype=torch.float32, device=device).unsqueeze(0)

    # 3. Instantiate Model & Load Checkpoint
    model = CancerComboBRICSSymmetric(
        gene_dim=config.GENE_DIM,
        cell_dim=config.CELL_DIM,
        frag_fp_dim=config.FRAG_FP_DIM,
        d_dim=config.D_DIM,
        num_attn_heads=config.NUM_ATTN_HEADS,
        shared_attn_weights=config.SHARED_ATTN_WEIGHTS
    ).to(device)

    if os.path.exists(model_checkpoint):
        checkpoint = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model checkpoint from '{model_checkpoint}'.")
    else:
        print(f"Warning: Checkpoint '{model_checkpoint}' not found. Using initialized model weights.")

    model.eval()
    with torch.no_grad():
        Y_pred = model(
            cell_expr=cell_expr,
            drugA_frags=fp_A,
            drugA_mask=mask_A,
            drugB_frags=fp_B,
            drugB_mask=mask_B,
            dose_grid=(dose_A_tensor, dose_B_tensor)
        )

    return Y_pred.squeeze(0).cpu().numpy()


def main():
    print("=" * 75)
    print("  CancerCombo — Deterministic Inference Demonstration")
    print("=" * 75)

    # Example Input Sample
    cell_id = "A549"
    smiles_A = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    smiles_B = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    doses_A = [0.0, 1e-7, 1e-6, 1e-5]
    doses_B = [0.0, 1e-7, 1e-6, 1e-5]

    print(f"Cell Line ID : {cell_id}")
    print(f"Drug A SMILES: {smiles_A}")
    print(f"Drug B SMILES: {smiles_B}")
    print(f"Doses A (M)  : {doses_A}")
    print(f"Doses B (M)  : {doses_B}")

    Y_surface = predict_combination_surface(cell_id, smiles_A, smiles_B, doses_A, doses_B)

    print("\nPredicted 2D Viability Surface Y (4x4):")
    print(np.round(Y_surface, 4))
    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
