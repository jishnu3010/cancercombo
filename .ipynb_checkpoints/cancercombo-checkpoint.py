import torch
import torch.nn as nn
from typing import Tuple

class CancerCombo(nn.Module):
    """A clean, robust, and deadlock-free architecture for Dose-Response Prediction."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        
        # 1. Drug Feature Projections (Standard MLPs)
        self.morgan_proj = nn.Sequential(
            nn.Linear(config.morgan_in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.desc_proj = nn.Sequential(
            nn.Linear(config.descriptor_in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        # Assuming MolFormer output is already d_model, else add a projection
        
        # 2. Robust Feature Fusion (Bypassing MultiheadAttention bugs)
        # Concatenate MolFormer, Morgan, Descriptor -> pass through MLP
        self.drug_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 3. Cell Line Encoder
        self.cell_enc = nn.Sequential(
            nn.Linear(config.cell_in_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 4. Drug-Cell Interaction (Replacing buggy Cross-Attention with Bilinear/MLP Fusion)
        self.drug_cell_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 5. Prediction Heads (Generating Pharmacological Params)
        self.heads = nn.Sequential(
            nn.Linear(d_model * 2, d_model), # Takes concatenated Drug A & Drug B
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 8) # e1, e2, e3, log_c1, log_c2, h1, h2, alpha
        )

    def forward(
        self,
        drug_a_ids, drug_a_mask, drug_a_morgan, drug_a_desc,
        drug_b_ids, drug_b_mask, drug_b_morgan, drug_b_desc,
        cell_line, doses_a, doses_b
    ):
        # 1. Dummy MolFormer Pooling (Replace with actual MolFormer call if integrated)
        # seq_a, pooled_a = self.molformer_enc(...)
        # For now, assuming pooled_a is available. If not, replace this block with your actual MolFormer output.
        pooled_a = torch.zeros(drug_a_morgan.size(0), self.config.d_model, device=drug_a_morgan.device)
        pooled_b = torch.zeros(drug_b_morgan.size(0), self.config.d_model, device=drug_b_morgan.device)
        
        # 2. Encode & Fuse Drug A
        morgan_a = self.morgan_proj(drug_a_morgan)
        desc_a = self.desc_proj(drug_a_desc)
        fused_a = self.drug_fusion(torch.cat([pooled_a, morgan_a, desc_a], dim=-1))
        
        # 3. Encode & Fuse Drug B
        morgan_b = self.morgan_proj(drug_b_morgan)
        desc_b = self.desc_proj(drug_b_desc)
        fused_b = self.drug_fusion(torch.cat([pooled_b, morgan_b, desc_b], dim=-1))
        
        # 4. Encode Cell Line
        cell_features = self.cell_enc(cell_line)
        
        # 5. Drug-Cell Interaction
        cond_a = self.drug_cell_fusion(torch.cat([fused_a, cell_features], dim=-1))
        cond_b = self.drug_cell_fusion(torch.cat([fused_b, cell_features], dim=-1))
        
        # 6. Predict Parameters
        combo_features = torch.cat([cond_a, cond_b], dim=-1)
        params_raw = self.heads(combo_features)
        
        # Split into individual parameters
        e1, e2, e3 = params_raw[:, 0], params_raw[:, 1], params_raw[:, 2]
        log_c1, log_c2 = params_raw[:, 3], params_raw[:, 4]
        h1, h2, alpha = params_raw[:, 5], params_raw[:, 6], params_raw[:, 7]
        
        # 7. Clean, Safe Hill Equation (No Masks, No Zero Gradients)
        # Using simple continuous functions to guarantee gradient flow
        doses_a_safe = torch.clamp(doses_a, min=1e-6)
        doses_b_safe = torch.clamp(doses_b, min=1e-6)
        
        log_x1 = torch.log(doses_a_safe)
        log_x2 = torch.log(doses_b_safe)
        
        h1_u, h2_u = h1.unsqueeze(-1).unsqueeze(-1), h2.unsqueeze(-1).unsqueeze(-1)
        log_c1_u, log_c2_u = log_c1.unsqueeze(-1).unsqueeze(-1), log_c2.unsqueeze(-1).unsqueeze(-1)
        
        if log_x1.dim() == 2: log_x1 = log_x1.unsqueeze(-1)
        if log_x2.dim() == 2: log_x2 = log_x2.unsqueeze(1)
            
        exp_A = log_c1_u * h1_u + log_c2_u * h2_u
        exp_B = log_x1 * h1_u + log_c2_u * h2_u
        exp_C = log_c1_u * h1_u + log_x2 * h2_u
        exp_D = log_x1 * h1_u + log_x2 * h2_u
        
        def safe_exp(x):
            return torch.exp(torch.clamp(x, min=-20.0, max=20.0))
            
        val_A, val_B, val_C, val_D = safe_exp(exp_A), safe_exp(exp_B), safe_exp(exp_C), safe_exp(exp_D)
        
        e1_u, e2_u, e3_u, alpha_u = e1.view(-1, 1, 1), e2.view(-1, 1, 1), e3.view(-1, 1, 1), alpha.view(-1, 1, 1)
        
        numerator = 100.0 * val_A + e1_u * val_B + e2_u * val_C + e3_u * alpha_u * val_D
        denominator = val_A + val_B + val_C + alpha_u * val_D
        
        y_pred = numerator / (denominator + 1e-8)
        
        return y_pred, (e1, e2, e3, log_c1, log_c2, h1, h2, alpha)