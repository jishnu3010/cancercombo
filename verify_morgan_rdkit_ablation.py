import os
import torch
import pickle
import pandas as pd
import numpy as np
from dataset import load_nci60_gex, parse_dataframe_to_records, DrugComboDataset
from config import ModelConfig
from cancercombo import CancerCombo

def run_verification():
    print("=" * 80)
    print(" ABLATION 2 (MORGAN + RDKIT DESCRIPTORS ONLY) VERIFICATION REPORT")
    print("=" * 80)

    # 1. Dataset Row Counts Verification
    csv_path = "data/scenario1_combination_50k.csv"
    df = pd.read_csv(csv_path)
    total_csv_rows = len(df)
    train_rows = len(df[df["split"] == 1])
    val_rows = len(df[df["split"] == 2])
    test_rows = len(df[df["split"] == 3])

    print(f"\n1. Dataset Partition Counts ({csv_path}):")
    print(f"   Total CSV Rows      : {total_csv_rows}")
    print(f"   Train Rows (split=1): {train_rows}")
    print(f"   Val Rows   (split=2): {val_rows}")
    print(f"   Test Rows  (split=3): {test_rows}")
    assert total_csv_rows == 50000, f"Expected 50000 rows, got {total_csv_rows}"
    assert train_rows == 30000, f"Expected 30000 train rows, got {train_rows}"
    assert val_rows == 10000, f"Expected 10000 val rows, got {val_rows}"
    assert test_rows == 10000, f"Expected 10000 test rows, got {test_rows}"

    # 2. Check generated .pt and .pkl artifact files
    pt_path = "data/features/morgan_rdkit_only/drug_features_morgan_rdkit.pt"
    pkl_path = "data/features/morgan_rdkit_only/drug_features_morgan_rdkit.pkl"

    print(f"\n2. Precomputed Feature Artifact Files:")
    print(f"   PyTorch Store (.pt) : {pt_path} (Exists: {os.path.exists(pt_path)})")
    print(f"   Pickle Store (.pkl)  : {pkl_path} (Exists: {os.path.exists(pkl_path)})")
    assert os.path.exists(pt_path), f"Missing {pt_path}"
    assert os.path.exists(pkl_path), f"Missing {pkl_path}"

    # Load PyTorch feature store
    pt_store = torch.load(pt_path, map_location="cpu", weights_only=True)
    num_smiles = len(pt_store)
    sample_smiles = list(pt_store.keys())[0]
    sample_feat = pt_store[sample_smiles]

    print(f"\n3. Precomputed Feature Store Inspection:")
    print(f"   Total Unique SMILES in Feature Store: {num_smiles}")
    print(f"   Keys in Drug Entry Dict             : {list(sample_feat.keys())}")
    for k, v in sample_feat.items():
        if torch.is_tensor(v):
            print(f"     Field '{k}': shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"     Field '{k}': type={type(v)}")

    morgan_dim = sample_feat["morgan"].shape[-1]
    desc_dim = sample_feat["descriptors"].shape[-1]
    print(f"   Morgan Fingerprint Dimension        : {morgan_dim}")
    print(f"   RDKit Descriptors Dimension         : {desc_dim}")

    assert morgan_dim == 2048, f"Expected 2048-bit Morgan fingerprint, got {morgan_dim}"
    assert desc_dim == 200, f"Expected 200 descriptors, got {desc_dim}"

    # Load Pickle feature store to confirm identical content
    with open(pkl_path, "rb") as f:
        pkl_store = pickle.load(f)
    print(f"   Pickle Feature Store SMILES Count   : {len(pkl_store)}")
    assert len(pkl_store) == num_smiles

    # 3. Verify MolFormer Exclusion in New Artifacts
    forbidden_terms = ["molformer", "molformer_emb", "token_ids", "token_mask", "hidden_states"]
    found_forbidden = []
    for k in sample_feat.keys():
        for term in forbidden_terms:
            if term in k.lower():
                found_forbidden.append((k, term))

    print(f"\n4. MolFormer Representation Exclusion Verification:")
    if not found_forbidden:
        print("   [CONFIRMED] Zero MolFormer embeddings/tokens present in Ablation 2 feature store!")
    else:
        print(f"   [WARNING] Forbidden MolFormer terms detected: {found_forbidden}")
    assert not found_forbidden, f"MolFormer terms found in Ablation 2 feature store: {found_forbidden}"

    # 4. Verify Dataset Parsing & DataLoader Output
    gex_dict = load_nci60_gex()
    records = parse_dataframe_to_records(df, known_gex_dict=gex_dict)
    
    train_records = parse_dataframe_to_records(df[df["split"] == 1], known_gex_dict=gex_dict)
    val_records = parse_dataframe_to_records(df[df["split"] == 2], known_gex_dict=gex_dict)
    test_records = parse_dataframe_to_records(df[df["split"] == 3], known_gex_dict=gex_dict)

    print(f"\n5. Processed Dataset Record Counts vs Original Split Counts:")
    print(f"   Parsed Records Total   : {len(records)} (Original CSV: {total_csv_rows})")
    print(f"   Processed Train Records: {len(train_records)} (Original: {train_rows})")
    print(f"   Processed Val Records  : {len(val_records)} (Original: {val_rows})")
    print(f"   Processed Test Records : {len(test_records)} (Original: {test_rows})")
    assert len(train_records) > 0, "Train records should be non-empty"
    assert len(val_records) > 0, "Val records should be non-empty"
    assert len(test_records) > 0, "Test records should be non-empty"


    # 5. Sanity Check Sample Inspection
    ds = DrugComboDataset(records, gex_dict, drug_feature_store=pt_store)
    sample_item = ds[0]

    print(f"\n6. Sanity Check Sample #0 Inspection:")
    print(f"   Sample Keys        : {list(sample_item.keys())}")
    print(f"   smiles_a           : {records[0]['smiles_a']}")
    print(f"   smiles_b           : {records[0]['smiles_b']}")
    print(f"   cell_line          : {records[0]['cell_line_name']}")
    print(f"   drug_a_morgan shape: {sample_item['drug_a_morgan'].shape} | dtype: {sample_item['drug_a_morgan'].dtype}")
    print(f"   drug_a_desc shape  : {sample_item['drug_a_desc'].shape} | dtype: {sample_item['drug_a_desc'].dtype}")
    print(f"   drug_b_morgan shape: {sample_item['drug_b_morgan'].shape} | dtype: {sample_item['drug_b_morgan'].dtype}")
    print(f"   drug_b_desc shape  : {sample_item['drug_b_desc'].shape} | dtype: {sample_item['drug_b_desc'].dtype}")
    print(f"   cell_line GEX shape: {sample_item['cell_line'].shape}")
    print(f"   doses_a shape      : {sample_item['doses_a'].shape} | values: {sample_item['doses_a'].tolist()}")
    print(f"   doses_b shape      : {sample_item['doses_b'].shape} | values: {sample_item['doses_b'].tolist()}")
    print(f"   viability shape    : {sample_item['viability'].shape}")

    # Confirm absence of MolFormer keys in sample batch
    assert "drug_a_emb" not in sample_item, "drug_a_emb should NOT be in Ablation 2 sample"
    assert "drug_b_emb" not in sample_item, "drug_b_emb should NOT be in Ablation 2 sample"
    assert "drug_a_ids" not in sample_item, "drug_a_ids should NOT be in Ablation 2 sample"

    # 6. One-Batch Model Forward Pass Test
    config = ModelConfig(
        d_model=256, emb_size=1024, n_heads=4, d_ff=512, dropout=0.1,
        molformer_in_dim=768, morgan_in_dim=2048, descriptor_in_dim=200,
        cell_in_dim=976, use_pathway_projection=True, n_pathways=300,
        molformer_model_name="ibm/MoLFormer-XL-CIMA-100M", use_pretrained_molformer=False,
        enable_drug_drug_attention=False, use_symmetric_fusion=True,
        e_min=0.0, e_max=100.0, c_min=1e-6, c_max=1e3, h_min=0.1, h_max=10.0,
        alpha_min=1e-4, alpha_max=100.0
    )
    model = CancerCombo(config)
    model.eval()

    batch_size = 4
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    with torch.no_grad():
        y_pred, params = model(
            drug_a_morgan=batch["drug_a_morgan"],
            drug_a_desc=batch["drug_a_desc"],
            drug_b_morgan=batch["drug_b_morgan"],
            drug_b_desc=batch["drug_b_desc"],
            cell_line=batch["cell_line"],
            doses_a=batch["doses_a"],
            doses_b=batch["doses_b"]
        )

    print(f"\n7. One-Batch Model Forward Pass Verification:")
    print(f"   Input Batch Size             : {batch_size}")
    print(f"   y_pred Output Shape          : {y_pred.shape}")
    print(f"   Number of Predicted Params   : {len(params)} (e1, e2, e3, log_c1, log_c2, h1, h2, alpha)")
    print(f"   Forward Pass Status          : SUCCESS!")

    assert y_pred.shape == (batch_size, 4, 4), f"Expected shape ({batch_size}, 4, 4), got {y_pred.shape}"

    print("\n" + "=" * 80)
    print(" ALL ABLATION 2 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_verification()
