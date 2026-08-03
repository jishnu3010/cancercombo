#!/usr/bin/env python3
import torch
import torch.nn as nn
from config import load_config
from cancercombo import CancerCombo

def verify_forward_pass():
    print("=" * 60)
    print("VERIFYING DRUG-DRUG ATTENTION FORWARD PASS")
    print("=" * 60)
    
    config, _ = load_config("config.yaml")
    
    model = CancerCombo(config)
    model.eval()
    
    batch_size = 8
    drug_a_morgan = torch.randn(batch_size, 2048)
    drug_a_desc = torch.randn(batch_size, 200)
    drug_b_morgan = torch.randn(batch_size, 2048)
    drug_b_desc = torch.randn(batch_size, 200)
    cell_line = torch.randn(batch_size, 976)
    doses_a = torch.randn(batch_size, 4)
    doses_b = torch.randn(batch_size, 4)
    
    # 1. Compute Drug-Cell Encoder representations
    in_a = torch.cat([drug_a_morgan, drug_a_desc, cell_line], dim=1) # (B, 3224)
    in_b = torch.cat([drug_b_morgan, drug_b_desc, cell_line], dim=1) # (B, 3224)
    rep_a = model.drug_cell_encoder(in_a)
    rep_b = model.drug_cell_encoder(in_b)
    
    print(f"Drug A Representation Shape        : {tuple(rep_a.shape)}")
    print(f"Drug B Representation Shape        : {tuple(rep_b.shape)}")
    print(f"Drug-Drug Attention Input Shape    : Drug A {tuple(rep_a.shape)}, Drug B {tuple(rep_b.shape)}")
    
    # 2. Compute Drug-Drug Attention
    aware_a, aware_b = model.drug_drug_attn(rep_a, rep_b)
    print(f"Drug-Drug Attention Output Shape   : Drug A {tuple(aware_a.shape)}, Drug B {tuple(aware_b.shape)}")
    
    # 3. Concatenate pair representation
    enhanced_pair_rep = torch.cat([aware_a, aware_b], dim=1)
    print(f"Prediction Head Input Shape        : {tuple(enhanced_pair_rep.shape)}")
    
    # 4. Full model forward pass
    y_pred, params = model(
        drug_a_morgan=drug_a_morgan,
        drug_a_desc=drug_a_desc,
        drug_b_morgan=drug_b_morgan,
        drug_b_desc=drug_b_desc,
        cell_line=cell_line,
        doses_a=doses_a,
        doses_b=doses_b
    )
    
    print(f"Prediction Output Shape (y_pred)   : {tuple(y_pred.shape)}")
    print("-" * 60)
    print("Forward Pass Successful")
    print("=" * 60)

if __name__ == "__main__":
    verify_forward_pass()
