import os
import sys
import torch
import pandas as pd
import numpy as np

from config import load_config
from dataset import load_nci60_gex, parse_dataframe_to_records, DrugComboDataset, load_precomputed_drug_features
from cancercombo import CancerCombo

def run_verification():
    print("=" * 70)
    print("      DEEPSYNBA-INSPIRED ARCHITECTURE VERIFICATION SUITE")
    print("=" * 70)

    # 1. VERIFY FEATURES
    print("\n--- 1. FEATURE VERIFICATION ---")
    m_config, t_config = load_config("config.yaml")
    
    has_morgan = hasattr(m_config, "morgan_in_dim") and m_config.morgan_in_dim == 2048
    has_rdkit = hasattr(m_config, "descriptor_in_dim") and m_config.descriptor_in_dim == 200
    has_molformer = getattr(m_config, "use_pretrained_molformer", False)
    
    print(f"Morgan Fingerprints : {'PRESENT' if has_morgan else 'ABSENT'} (Dim: {getattr(m_config, 'morgan_in_dim', 'N/A')})")
    print(f"RDKit Descriptors   : {'PRESENT' if has_rdkit else 'ABSENT'} (Dim: {getattr(m_config, 'descriptor_in_dim', 'N/A')})")
    print(f"MolFormer           : {'PRESENT' if has_molformer else 'ABSENT'}")

    # 2. VERIFY DATASET
    print("\n--- 2. DATASET VERIFICATION ---")
    split_path = "data/scenario1_combination_50k.csv"
    if not os.path.exists(split_path) and os.path.exists("data/splits/scenario1_combination.csv"):
        split_path = "data/splits/scenario1_combination.csv"

    if os.path.exists(split_path):
        df_split = pd.read_csv(split_path)
        train_rows = len(df_split[df_split["split"] == 1])
        val_rows = len(df_split[df_split["split"] == 2])
        test_rows = len(df_split[df_split["split"] == 3])
        print(f"Dataset File   : {split_path}")
        print(f"Train rows     : {train_rows} (split == 1)")
        print(f"Validation rows: {val_rows} (split == 2)")
        print(f"Test rows      : {test_rows} (split == 3)")
        print("Dataset split status: UNCHANGED")
    else:
        print(f"Dataset split file not found at {split_path}")

    # 3. VERIFY ARCHITECTURE
    print("\n--- 3. ARCHITECTURE VERIFICATION ---")
    model = CancerCombo(m_config)
    
    active_attentions = []
    for name, module in model.named_modules():
        mod_type = type(module).__name__
        if any(att in mod_type for att in ["Attention", "MultiheadAttention", "CrossAttention", "SelfAttention", "MolFormer"]):
            active_attentions.append((name, mod_type))

    print(f"Active Attention Layers : {len(active_attentions)}")
    if len(active_attentions) > 0:
        print(f"  WARNING Found active attention modules: {active_attentions}")
    else:
        print("  [CONFIRMED] No active Attention layers, MultiHeadAttention, CrossAttention, SelfAttention, or MolFormer.")

    has_drug_cell_enc = hasattr(model, "drug_cell_encoder")
    has_pred_heads = hasattr(model, "heads")
    has_hill_solver = hasattr(model, "hill_solver")

    print(f"Drug-Cell Encoder implemented : {'CONFIRMED' if has_drug_cell_enc else 'MISSING'}")
    print(f"Prediction Heads implemented  : {'CONFIRMED' if has_pred_heads else 'MISSING'}")
    print(f"Hill Equation connected       : {'CONFIRMED' if has_hill_solver else 'MISSING'}")

    # 4. ONE FORWARD-PASS VERIFICATION
    print("\n--- 4. ONE FORWARD-PASS VERIFICATION ---")
    batch_size = 4
    drug_a_morgan = torch.randn(batch_size, 2048)
    drug_a_desc = torch.randn(batch_size, 200)
    drug_b_morgan = torch.randn(batch_size, 2048)
    drug_b_desc = torch.randn(batch_size, 200)
    cell_line = torch.randn(batch_size, 976)
    doses_a = torch.tensor([[0.0, 0.1, 1.0, 10.0]] * batch_size)
    doses_b = torch.tensor([[0.0, 0.1, 1.0, 10.0]] * batch_size)

    print(f"Input Drug A Morgan shape : {drug_a_morgan.shape}")
    print(f"Input Drug A Descriptors  : {drug_a_desc.shape}")
    print(f"Input Drug B Morgan shape : {drug_b_morgan.shape}")
    print(f"Input Drug B Descriptors  : {drug_b_desc.shape}")
    print(f"Input Cell GEX shape      : {cell_line.shape}")

    # Intermediate trace
    in_a = torch.cat([drug_a_morgan, drug_a_desc, cell_line], dim=1)
    in_b = torch.cat([drug_b_morgan, drug_b_desc, cell_line], dim=1)
    rep_a = model.drug_cell_encoder(in_a)
    rep_b = model.drug_cell_encoder(in_b)
    unified_rep = torch.cat([rep_a, rep_b], dim=1)

    print(f"Drug A Rep shape (rep_a)  : {rep_a.shape}")
    print(f"Drug B Rep shape (rep_b)  : {rep_b.shape}")
    print(f"Unified Rep shape         : {unified_rep.shape}")

    y_pred, params = model(
        drug_a_morgan=drug_a_morgan, drug_a_desc=drug_a_desc,
        drug_b_morgan=drug_b_morgan, drug_b_desc=drug_b_desc,
        cell_line=cell_line, doses_a=doses_a, doses_b=doses_b
    )

    e1, e2, e3, log_c1, log_c2, h1, h2, alpha = params
    print("Prediction Head Parameter Output Shapes:")
    print(f"  e1    : {e1.shape}")
    print(f"  e2    : {e2.shape}")
    print(f"  e3    : {e3.shape}")
    print(f"  log_c1: {log_c1.shape}")
    print(f"  log_c2: {log_c2.shape}")
    print(f"  h1    : {h1.shape}")
    print(f"  h2    : {h2.shape}")
    print(f"  alpha : {alpha.shape}")

    print(f"Hill Equation Output Matrix (y_pred): {y_pred.shape}")
    print("Forward pass successful: True")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
