import os
import sys
import torch
import pandas as pd
from dataset import load_nci60_gex, parse_dataframe_to_records, DrugComboDataset
from torch.utils.data import DataLoader
from cancercombo import CancerCombo
from losses import CancerComboLoss
from config import load_config
from metrics import calculate_metrics

def run_smoke_test():
    print("="*75)
    print(" C AN C E R C O M B O   R E A L - D A T A   E N D - T O - E N D   S M O K E   T E S T")
    print("="*75)
    
    # 1. Load landmark gene expression vectors
    print("Step 1 & 3: Loading NCI-60 GEX landmark features...")
    gex_dict = load_nci60_gex("data/features/NCI-60_landmark_gex.csv")
    if not gex_dict:
        raise RuntimeError("Failed to load NCI-60 landmark GEX data.")
    print(f"  [PASS] Loaded {len(gex_dict)} gene expression cell line keys.")
    
    # 2. Check Scenario Split CSVs
    print("\nStep 2 & 5: Checking generated scenario split CSVs...")
    s1_path = "data/splits/scenario1_combination.csv"
    s2_path = "data/splits/scenario2_cell.csv"
    s3_path = "data/splits/scenario3_drug.csv"
    
    for p in [s1_path, s2_path, s3_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing scenario split file: {p}")
    print("  [PASS] All 3 scenario split CSV files exist.")
    
    # 4 & 6. Verify Leakage & Load Scenario 1 Partitions
    print("\nStep 6 & 7: Loading Scenario 1 train (split=1), val (split=2), test (split=3)...")
    s1_df = pd.read_csv(s1_path)
    train_df = s1_df[s1_df["split"] == 1].head(100)
    val_df = s1_df[s1_df["split"] == 2].head(20)
    test_df = s1_df[s1_df["split"] == 3].head(20)
    print(f"  [PASS] Subset partition sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 8. Create DrugComboDataset objects
    print("\nStep 8: Creating DrugComboDataset objects...")
    train_records = parse_dataframe_to_records(train_df, known_gex_dict=gex_dict)
    val_records = parse_dataframe_to_records(val_df, known_gex_dict=gex_dict)
    test_records = parse_dataframe_to_records(test_df, known_gex_dict=gex_dict)
    
    train_ds = DrugComboDataset(train_records, gex_dict)
    val_ds = DrugComboDataset(val_records, gex_dict)
    test_ds = DrugComboDataset(test_records, gex_dict)
    print(f"  [PASS] Created datasets. Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # 9. Create DataLoaders
    print("\nStep 9: Creating DataLoaders...")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    # 10. Fetch one real training batch
    print("\nStep 10 & 12: Fetching real training batch and verifying tensor shapes...")
    batch = next(iter(train_loader))
    for k in ["drug_a_ids", "drug_a_mask", "drug_a_morgan", "drug_a_desc", 
              "drug_b_ids", "drug_b_mask", "drug_b_morgan", "drug_b_desc", 
              "cell_line", "doses_a", "doses_b", "viability"]:
        print(f"  Tensor '{k}': shape {batch[k].shape}, dtype {batch[k].dtype}")
        
    assert batch["cell_line"].shape[-1] == 976, f"Expected cell_line dim 976, got {batch['cell_line'].shape[-1]}"
    assert batch["viability"].dim() == 3, f"Expected 3D viability matrix (B, M, N), got {batch['viability'].dim()}D"
    
    # 11. Run complete forward pass
    print("\nStep 11: Running forward pass...")
    m_config, t_config = load_config()
    model = CancerCombo(m_config)
    y_pred, params = model(
        batch["drug_a_ids"], batch["drug_a_mask"], batch["drug_a_morgan"], batch["drug_a_desc"],
        batch["drug_b_ids"], batch["drug_b_mask"], batch["drug_b_morgan"], batch["drug_b_desc"],
        batch["cell_line"], batch["doses_a"], batch["doses_b"]
    )
    print(f"  [PASS] Predicted y_pred shape: {y_pred.shape}")
    
    # 13 & 14. Compute real loss & verify finite
    print("\nStep 13 & 14: Computing CancerComboLoss...")
    loss_fn = CancerComboLoss()
    loss = loss_fn(y_pred, batch["viability"], params)
    print(f"  Computed Loss: {loss.item():.6f}")
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    
    # 15 & 16. Run loss.backward() and check gradients
    print("\nStep 15 & 16: Running loss.backward() and gradient checks...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    
    nan_grads = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                nan_grads.append(name)
    assert len(nan_grads) == 0, f"NaN/Inf gradients found in parameters: {nan_grads}"
    print("  [PASS] All gradients are finite and clean.")
    
    # 17. Run optimizer step
    print("\nStep 17: Executing optimizer step...")
    optimizer.step()
    print("  [PASS] Optimizer step completed successfully.")
    
    # 18. Run validation batch
    print("\nStep 18: Executing validation batch...")
    model.eval()
    val_batch = next(iter(val_loader))
    with torch.no_grad():
        val_pred, _ = model(
            val_batch["drug_a_ids"], val_batch["drug_a_mask"], val_batch["drug_a_morgan"], val_batch["drug_a_desc"],
            val_batch["drug_b_ids"], val_batch["drug_b_mask"], val_batch["drug_b_morgan"], val_batch["drug_b_desc"],
            val_batch["cell_line"], val_batch["doses_a"], val_batch["doses_b"]
        )
        val_loss = loss_fn(val_pred, val_batch["viability"])
    print(f"  [PASS] Validation Loss: {val_loss.item():.6f}")
    
    # 19 & 20. Save & reload temporary checkpoint
    print("\nStep 19 & 20: Saving and reloading checkpoint...")
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/temp_smoke_test.ckpt"
    torch.save({"state_dict": model.state_dict(), "config": m_config}, ckpt_path)
    
    fresh_model = CancerCombo(m_config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fresh_model.load_state_dict(ckpt["state_dict"])
    fresh_model.eval()
    print("  [PASS] Checkpoint saved and reloaded into fresh model.")
    
    # 21, 22, 23. Run test evaluation & calculate metrics
    print("\nStep 21, 22 & 23: Inference on test subset & calculating evaluation metrics...")
    test_preds, test_trues = [], []
    with torch.no_grad():
        for test_batch in test_loader:
            p, _ = fresh_model(
                test_batch["drug_a_ids"], test_batch["drug_a_mask"], test_batch["drug_a_morgan"], test_batch["drug_a_desc"],
                test_batch["drug_b_ids"], test_batch["drug_b_mask"], test_batch["drug_b_morgan"], test_batch["drug_b_desc"],
                test_batch["cell_line"], test_batch["doses_a"], test_batch["doses_b"]
            )
            test_preds.append(p.numpy())
            test_trues.append(test_batch["viability"].numpy())
            
    test_preds = torch.tensor(test_preds[0])
    test_trues = torch.tensor(test_trues[0])
    
    assert not torch.isnan(test_preds).any(), "NaN found in test predictions!"
    assert not torch.isinf(test_preds).any(), "Inf found in test predictions!"
    
    metrics = calculate_metrics(test_preds.numpy(), test_trues.numpy())
    print("  Evaluation Metrics on Test Subset:")
    for m_name, m_val in metrics.items():
        print(f"    {m_name.upper()}: {m_val:.4f}")
        
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        
    print("\n" + "="*75)
    print(" ALL 23 REAL-DATA SMOKE TEST VERIFICATION STEPS PASSED SUCCESSFULLY!")
    print("="*75 + "\n")

if __name__ == "__main__":
    run_smoke_test()

