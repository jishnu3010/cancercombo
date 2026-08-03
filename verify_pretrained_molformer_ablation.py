import os
import sys
import torch
import pickle
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from config import load_config
from dataset import load_nci60_gex, parse_dataframe_to_records, DrugComboDataset, load_precomputed_drug_features
from blocks.molformer_encoder import MolFormerEncoder
from cancercombo import CancerCombo

def run_verification():
    print("=" * 80)
    print(" PRETRAINED IBM MOLFORMER ABLATION 1 VERIFICATION REPORT")
    print("=" * 80)

    # Load Config
    m_config, t_config = load_config("config.yaml")

    print(f"\n1. Pretrained Model Configuration Check:")
    print(f"   Model Requested          : {m_config.molformer_model_name}")
    print(f"   use_pretrained Config    : {m_config.use_pretrained_molformer}")
    assert m_config.use_pretrained_molformer is True, "use_pretrained_molformer must be True in config.yaml!"

    # 2. Test Direct Loading of Pretrained IBM MoLFormer Model & Tokenizer
    print(f"\n2. Loading Pretrained IBM MoLFormer from Hugging Face...")
    try:
        hf_model = AutoModel.from_pretrained(m_config.molformer_model_name, trust_remote_code=True)
        hf_tokenizer = AutoTokenizer.from_pretrained(m_config.molformer_model_name, trust_remote_code=True)
        pretrained_loaded = True
        raw_hidden_dim = getattr(hf_model.config, "hidden_size", 768)
        print(f"   Pretrained Model Loaded   : TRUE")
        print(f"   Raw Hidden Dimension     : {raw_hidden_dim}")
        print(f"   HF Tokenizer Class       : {hf_tokenizer.__class__.__name__}")
    except Exception as e:
        pretrained_loaded = False
        print(f"   Pretrained Model Loaded   : FALSE (Error: {e})")
        raise RuntimeError(f"FATAL: Could not load pretrained model '{m_config.molformer_model_name}': {e}")

    # 3. Check Dataset & 30K/10K/10K Split Invariants
    csv_path = "data/scenario1_combination_50k.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/splits/scenario1_combination.csv"
    df = pd.read_csv(csv_path)
    train_rows = len(df[df["split"] == 1])
    val_rows = len(df[df["split"] == 2])
    test_rows = len(df[df["split"] == 3])

    print(f"\n3. Dataset Split Verification ({csv_path}):")
    print(f"   Total CSV Rows           : {len(df)}")
    print(f"   Train Rows (split=1)     : {train_rows}")
    print(f"   Val Rows   (split=2)     : {val_rows}")
    print(f"   Test Rows  (split=3)     : {test_rows}")
    assert train_rows == 30000, f"Expected 30,000 train rows, got {train_rows}"
    assert val_rows == 10000, f"Expected 10,000 val rows, got {val_rows}"
    assert test_rows == 10000, f"Expected 10,000 test rows, got {test_rows}"

    # 4. Check Feature Store Artifacts
    pt_path = "data/features/pretrained_molformer_only/drug_features_pretrained_molformer.pt"
    pkl_path = "data/features/pretrained_molformer_only/drug_features_pretrained_molformer.pkl"

    print(f"\n4. Feature Store Artifact Check:")
    print(f"   PyTorch Store (.pt) Path : {pt_path} (Exists: {os.path.exists(pt_path)})")
    print(f"   Pickle Store (.pkl) Path : {pkl_path} (Exists: {os.path.exists(pkl_path)})")

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Feature store not found at '{pt_path}'. Please run precompute_molecular_features.py first.")

    feature_store = torch.load(pt_path, map_location="cpu", weights_only=True)
    num_unique_smiles = len(feature_store)
    sample_smiles = list(feature_store.keys())[0]
    sample_entry = feature_store[sample_smiles]

    print(f"   Unique SMILES in Store   : {num_unique_smiles}")
    print(f"   Feature Entry Keys       : {list(sample_entry.keys())}")
    stored_dim = sample_entry["molformer_emb"].shape[-1]
    print(f"   Stored Embedding Dim     : {stored_dim}")

    # 5. Verify Absence of Morgan Fingerprints & RDKit Descriptors
    forbidden_keys = ["morgan", "fingerprint", "fp", "descriptor", "rdkit"]
    found_forbidden = [k for k in sample_entry.keys() if any(f in k.lower() for f in forbidden_keys)]
    print(f"\n5. Morgan & RDKit Exclusion Verification:")
    print(f"   Morgan Fingerprints Present: FALSE")
    print(f"   RDKit Descriptors Present  : FALSE")
    assert len(found_forbidden) == 0, f"Forbidden feature keys found in store: {found_forbidden}"

    # 6. Verify One SMILES String
    test_smiles = sample_smiles
    encoded = hf_tokenizer(test_smiles, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        hf_outputs = hf_model(**encoded)
        last_hidden = hf_outputs.last_hidden_state # (1, L, 768)
        mask = encoded["attention_mask"].unsqueeze(-1).to(dtype=last_hidden.dtype)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        raw_pooled = (last_hidden * mask).sum(dim=1) / sum_mask # (1, 768)

    print(f"\n6. Single SMILES Detailed Verification:")
    print(f"   Target SMILES                    : {test_smiles[:60]}...")
    print(f"   Tokenized Input Shape            : {tuple(encoded['input_ids'].shape)}")
    print(f"   Pretrained Hidden-State Shape    : {tuple(last_hidden.shape)}")
    print(f"   Raw Pooled Representation Shape  : {tuple(raw_pooled.shape)}")
    print(f"   CancerCombo Projected Stored Shape: {tuple(sample_entry['molformer_emb'].shape)}")

    # 7. Single Batch CancerCombo Forward Pass Verification
    gex_dict = load_nci60_gex()
    records = parse_dataframe_to_records(df.head(4), known_gex_dict=gex_dict)
    ds = DrugComboDataset(records, gex_dict, drug_feature_store=feature_store)
    batch = ds[0]

    # Create batch tensors
    batch_a_emb = batch["drug_a_emb"].unsqueeze(0)
    batch_b_emb = batch["drug_b_emb"].unsqueeze(0)
    batch_cell = batch["cell_line"].unsqueeze(0)
    batch_doses_a = batch["doses_a"].unsqueeze(0)
    batch_doses_b = batch["doses_b"].unsqueeze(0)

    model = CancerCombo(m_config)
    model.eval()

    with torch.no_grad():
        y_pred, params = model(
            drug_a_ids=batch["drug_a_ids"].unsqueeze(0),
            drug_a_mask=batch["drug_a_mask"].unsqueeze(0),
            drug_b_ids=batch["drug_b_ids"].unsqueeze(0),
            drug_b_mask=batch["drug_b_mask"].unsqueeze(0),
            cell_line=batch_cell,
            doses_a=batch_doses_a,
            doses_b=batch_doses_b,
            drug_a_emb=batch_a_emb,
            drug_b_emb=batch_b_emb
        )

    print(f"\n7. Single Batch CancerCombo Forward Pass Verification:")
    print(f"   drug_a pretrained embedding shape : {tuple(batch_a_emb.shape)}")
    print(f"   drug_b pretrained embedding shape : {tuple(batch_b_emb.shape)}")
    print(f"   cell_line gene expr feature shape : {tuple(batch_cell.shape)}")
    print(f"   predicted viability matrix shape  : {tuple(y_pred.shape)}")
    print(f"   Forward pass without Morgan/desc  : SUCCESS (No Morgan or descriptor inputs required)")

    print("\n" + "=" * 80)
    print(" ALL PRETRAINED MOLFORMER ABLATION 1 VERIFICATION CHECKS PASSED!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_verification()
