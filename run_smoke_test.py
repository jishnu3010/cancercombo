import os
import sys
import torch
import pandas as pd
from dataset import load_nci60_gex, load_synergy_dataset, DrugComboDataset
from torch.utils.data import DataLoader
from cancercombo import CancerCombo
from losses import CancerComboLoss
from config import load_config

def run_smoke_test():
    print("="*60)
    print("1. Loading NCI-60 GEX and Matching Cell Lines...")
    gex_dict = load_nci60_gex()
    print(f"Loaded {len(gex_dict)} gene expression vectors.")
    
    print("\n2. Loading Synergy Dataset (First 100 rows)...")
    # Instead of reading the full 4.5M rows for the smoke test, we'll read just a chunk of the raw CSV.
    import zipfile
    z = zipfile.ZipFile("data/DrugCombination_with_SMILES.zip")
    csv_file = [f for f in z.namelist() if f.endswith('.csv')][0]
    with z.open(csv_file) as f:
        df = pd.read_csv(f, nrows=1000)
        
    print("3. Preprocessing into records...")
    from dataset import parse_dataframe_to_records
    records = parse_dataframe_to_records(df, known_cells=set(gex_dict.keys()))
    
    if not records:
        print("ERROR: No matched records found!")
        sys.exit(1)
        
    dataset = DrugComboDataset(records, gex_dict)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    
    print("\n4. Tensor Shapes from DataLoader:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
            
    m_config, _ = load_config()
    model = CancerCombo(m_config)
    
    print("\n5. Forward Pass Check...")
    y_pred, params = model(
        batch["drug_a_ids"], batch["drug_a_mask"], batch["drug_a_morgan"], batch["drug_a_desc"],
        batch["drug_b_ids"], batch["drug_b_mask"], batch["drug_b_morgan"], batch["drug_b_desc"],
        batch["cell_line"], batch["doses_a"], batch["doses_b"]
    )
    
    print(f"  y_pred shape: {y_pred.shape}")
    for i, v in enumerate(params):
        print(f"  param {i} shape: {v.shape}")
        
    print("\n6. Loss Check...")
    loss_fn = CancerComboLoss()
    loss = loss_fn(y_pred, batch["viability"], params)
    print(f"  Loss Value: {loss.item()}")
    
    if torch.isnan(loss):
        print("ERROR: Loss is NaN!")
        sys.exit(1)
        
    print("\n7. Backward Pass & Optimizer Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  Optimizer step succeeded.")
    
    print("\n8. Checking for NaN/Inf in outputs...")
    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
        print("  WARNING: NaN or Inf found in y_pred")
    else:
        print("  y_pred is clean.")
        
    print("\nSMOKE TEST COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    run_smoke_test()
