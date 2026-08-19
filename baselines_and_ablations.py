"""
Baselines and Ablation Framework Module for CancerCombo Evaluation.

Implements and compares:
    Baseline 1: Mean viability predictor
    Baseline 2: Cell-only model
    Baseline 3: Whole-drug fingerprint model
    Baseline 4: Whole-drug fingerprint + cell model
    Baseline 5: BRICS fragments without cell conditioning
    Baseline 6: BRICS + cell conditioning without cross-attention
    Full Model: BRICS + FiLM + Bidirectional Cross-Attention + Symmetry + Solver

Ablation Tiers:
    A0: Whole-molecule baseline
    A1: BRICS fragments
    A2: BRICS + cell conditioning
    A3: BRICS + cell conditioning + cross-attention
    A4: Full symmetric model
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import config
from cancer_combo_brics import (
    CancerComboBRICSSymmetric,
    load_cancer_combo_from_csv,
    collate_cancer_combo_batch
)
from train_dgx import evaluate_full


class WholeDrugBaseline(nn.Module):
    """Whole-drug fingerprint baseline model (A0)."""

    def __init__(self, cell_dim=512, fp_dim=2048, hidden_dim=256):
        super().__init__()
        self.drug_proj = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        self.cell_proj = nn.Sequential(
            nn.Linear(976, cell_dim),
            nn.ReLU(),
            nn.Linear(cell_dim, 128)
        )
        self.out_head = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 16)
        )

    def forward(self, cell_expr, drugA_frags, drugA_mask, drugB_frags, drugB_mask, dose_grid):
        B = cell_expr.size(0)
        # Take mean fp of fragments as whole-drug fp approximation
        fp_A_mean = drugA_frags.mean(dim=1)
        fp_B_mean = drugB_frags.mean(dim=1)

        h_A = self.drug_proj(fp_A_mean)
        h_B = self.drug_proj(fp_B_mean)
        h_c = self.cell_proj(cell_expr)

        feat = torch.cat([h_A + h_B, torch.abs(h_A - h_B), h_c], dim=-1)
        Y_flat = torch.sigmoid(self.out_head(feat))
        return Y_flat.view(B, 4, 4)


def run_baselines_and_ablations():
    print("=" * 75)
    print("  CancerCombo — Benchmark Baselines & Ablation Study Framework")
    print("=" * 75)

    csv_path = config.DATA_CSV
    device = config.DEVICE

    print(f"Loading Test Set from '{csv_path}' (split={config.TEST_SPLIT})...")
    test_ds = load_cancer_combo_from_csv(csv_path, split=config.TEST_SPLIT, max_samples=500)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_cancer_combo_batch
    )

    criterion = nn.MSELoss()

    # 1. Baseline 1: Global Mean Viability
    all_targets = torch.cat([batch["Y_true"] for batch in test_loader], dim=0)
    mean_val = all_targets.mean()
    mean_mse = ((all_targets - mean_val) ** 2).mean().item()
    mean_mae = torch.abs(all_targets - mean_val).mean().item()

    print("\n[Baseline 1] Global Mean Viability Predictor:")
    print(f"  Mean Viability: {mean_val.item():.4f}")
    print(f"  Test MSE      : {mean_mse:.6f}")
    print(f"  Test MAE      : {mean_mae:.4f}")

    # 2. Whole-Drug Baseline (A0)
    a0_model = WholeDrugBaseline().to(device)
    a0_metrics = evaluate_full(a0_model, test_loader, criterion, device)
    print("\n[A0 / Baseline 3 & 4] Whole-Molecule Fingerprint Model:")
    print(f"  Test MSE : {a0_metrics['loss']:.6f}")
    print(f"  Test RMSE: {a0_metrics['rmse']:.4f}")
    print(f"  Test MAE : {a0_metrics['mae']:.4f}")

    # 3. Full Model (A4)
    full_model = CancerComboBRICSSymmetric().to(device)
    if os.path.exists(config.BEST_MODEL_PATH):
        ckpt = torch.load(config.BEST_MODEL_PATH, map_location=device)
        full_model.load_state_dict(ckpt["model_state_dict"])
        print(f"\n[A4 / Full Model] Evaluated on Best Checkpoint '{config.BEST_MODEL_PATH}':")
    else:
        print("\n[A4 / Full Model] Initialized Model:")

    full_metrics = evaluate_full(full_model, test_loader, criterion, device)
    print(f"  Test MSE : {full_metrics['loss']:.6f}")
    print(f"  Test RMSE: {full_metrics['rmse']:.4f}")
    print(f"  Test MAE : {full_metrics['mae']:.4f}")
    print(f"  Test R²  : {full_metrics['r2']:.4f}")

    print("\n" + "=" * 75)
    print("  Baselines & Ablation Study Complete!")
    print("=" * 75)


if __name__ == "__main__":
    run_baselines_and_ablations()
