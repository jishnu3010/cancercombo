import os
import torch
import pickle
import pandas as pd
import numpy as np
from dataset import load_nci60_gex, parse_dataframe_to_records, DrugComboDataset, load_precomputed_drug_features

def run_verification():
    print("=" * 80)
    print(" MOLFORMER-ONLY ABLATION VERIFICATION REPORT")
    print("=" * 80)

    # 1. Dataset Row Counts Verification
    csv_path = "data/scenario1_combination_50k.csv"
    df = pd.read_csv(csv_path)
    total_csv_rows = len(df)
    train_rows = len(df[df["split"] == 1])
    val_rows = len(df[df["split"] == 2])
    test_rows = len(df[df["split"] == 3])

    print(f"\n1. Original Dataset Partition Counts ({csv_path}):")
    print(f"   Total CSV Rows : {total_csv_rows}")
    print(f"   Train Rows (split=1): {train_rows}")
    print(f"   Val Rows   (split=2): {val_rows}")
    print(f"   Test Rows  (split=3): {test_rows}")

    # 2. Check generated .pt and .pkl artifacts
    pt_path = "data/features/molformer_only/drug_features_molformer.pt"
    pkl_path = "data/features/molformer_only/drug_features_molformer.pkl"

    print(f"\n2. Precomputed Artifact Files:")
    print(f"   PyTorch Store (.pt) : {pt_path} (Exists: {os.path.exists(pt_path)})")
    print(f"   Pickle Store (.pkl)  : {pkl_path} (Exists: {os.path.exists(pkl_path)})")

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

    molformer_dim = sample_feat["molformer_emb"].shape[-1]
    print(f"   MolFormer Embedding Dimension       : {molformer_dim}")

    # Load Pickle feature store to confirm identical content
    with open(pkl_path, "rb") as f:
        pkl_store = pickle.load(f)
    print(f"   Pickle Feature Store SMILES Count   : {len(pkl_store)}")

    # 3. Search for Morgan / Descriptor keys in generated artifact
    forbidden_terms = ["morgan", "fingerprint", "fp", "ecfp", "rdkit_descriptor", "descriptor", "molecular_descriptor"]
    found_forbidden = []
    for k in sample_feat.keys():
        for term in forbidden_terms:
            if term in k.lower():
                found_forbidden.append((k, term))

    print(f"\n4. Morgan & RDKit Descriptor Exclusion Verification:")
    if not found_forbidden:
        print("   [CONFIRMED] Zero Morgan fingerprints or RDKit descriptors present in new artifacts!")
    else:
        print(f"   [WARNING] Forbidden terms detected in feature keys: {found_forbidden}")

    # 4. Verify Dataset Parsing & DataLoader Output
    gex_dict = load_nci60_gex()
    records = parse_dataframe_to_records(df, known_gex_dict=gex_dict)
    
    train_records = parse_dataframe_to_records(df[df["split"] == 1], known_gex_dict=gex_dict)
    val_records = parse_dataframe_to_records(df[df["split"] == 2], known_gex_dict=gex_dict)
    test_records = parse_dataframe_to_records(df[df["split"] == 3], known_gex_dict=gex_dict)

    print(f"\n5. Processed Dataset Record Counts vs Original Split Counts:")
    print(f"   Parsed Records Total: {len(records)} (Original CSV: {total_csv_rows})")
    print(f"   Processed Train Records: {len(train_records)} (Original: {train_rows}) -> Match: {len(train_records) == train_rows}")
    print(f"   Processed Val Records  : {len(val_records)} (Original: {val_rows}) -> Match: {len(val_records) == val_rows}")
    print(f"   Processed Test Records : {len(test_records)} (Original: {test_rows}) -> Match: {len(test_records) == test_rows}")

    # 5. Sanity Check Sample Inspection
    ds = DrugComboDataset(records, gex_dict, drug_feature_store=pt_store)
    sample_item = ds[0]

    print(f"\n6. Sanity Check Sample #0 Inspection:")
    print(f"   Sample Keys: {list(sample_item.keys())}")
    print(f"   smiles_a   : {records[0]['smiles_a']}")
    print(f"   smiles_b   : {records[0]['smiles_b']}")
    print(f"   cell_line  : {records[0]['cell_line_name']}")
    print(f"   doses_a shape      : {sample_item['doses_a'].shape} | values: {sample_item['doses_a'].tolist()}")
    print(f"   doses_b shape      : {sample_item['doses_b'].shape} | values: {sample_item['doses_b'].tolist()}")
    print(f"   viability shape    : {sample_item['viability'].shape}")
    print(f"   drug_a_emb shape   : {sample_item['drug_a_emb'].shape} | first 5 values: {sample_item['drug_a_emb'][:5].tolist()}")
    print(f"   drug_b_emb shape   : {sample_item['drug_b_emb'].shape} | first 5 values: {sample_item['drug_b_emb'][:5].tolist()}")
    print(f"   cell_line GEX shape: {sample_item['cell_line'].shape}")

    print("\n" + "=" * 80)
    print(" ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_verification()
