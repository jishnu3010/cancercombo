import torch
import torch.nn as nn
from typing import Tuple

from blocks.molformer_encoder import MolFormerEncoder
from blocks.morgan_encoder import MorganEncoder
from blocks.descriptor_encoder import DescriptorEncoder
from blocks.fusion import AttentionMultiRepresentationFusion
from blocks.cell_encoder import CellLineEncoder
from blocks.drug_cell_attention import DrugCellCrossAttention
from blocks.drug_drug_attention import DrugDrugCrossAttention
from blocks.shared_feature import SymmetricComboFusion
from blocks.prediction_heads import CancerComboPredictionHeads
from blocks.hill_equation import BivariateHillSolver

class CancerCombo(nn.Module):
    """Complete, modular CancerCombo architecture for dose-response prediction.
    
    Integrates SMILES MolFormer sequence encoding, Morgan and continuous Descriptor encoders,
    Attention Multi-Representation Fusion, Pathway Cell Line Encoder, Drug-Cell Cross Attention,
    Symmetric Combo Fusion for exact permutation invariance, constrained Prediction Heads,
    and numerically stable Bivariate Hill Solver.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        
        # 1. SMILES & Molecular Encoders
        # Integrated MolFormer encoder to process SMILES token IDs (Phase 3)
        self.molformer_enc = MolFormerEncoder(
            d_model=d_model,
            vocab_size=100,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            use_pretrained=getattr(config, "use_pretrained_molformer", False),
            model_name=getattr(config, "molformer_model_name", "ibm/MoLFormer-XL-CIMA-100M")
        )
        # 1. SMILES & Molecular Encoders
        # Integrated MolFormer encoder to process SMILES token IDs (Phase 3)
        self.molformer_enc = MolFormerEncoder(
            d_model=d_model,
            vocab_size=100,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            use_pretrained=getattr(config, "use_pretrained_molformer", False),
            model_name=getattr(config, "molformer_model_name", "ibm/MoLFormer-XL-CIMA-100M")
        )

# ============================================================
# OLD CODE - DISABLED FOR MOLFORMER-ONLY ABLATION
# ============================================================
#         # Morgan fingerprint encoder block
#         self.morgan_enc = MorganEncoder(
#             in_dim=config.morgan_in_dim,
#             d_model=d_model,
#             dropout=config.dropout
#         )
#         # continuous molecular descriptor encoder block
#         self.descriptor_enc = DescriptorEncoder(
#             in_dim=config.descriptor_in_dim,
#             d_model=d_model,
#             dropout=config.dropout
#         )

# ============================================================
# NEW CODE - MOLFORMER-ONLY ABLATION
# ============================================================
        # 2. Multi-Head Self-Attention Fusion Block (Pass-through for MolFormer-only)
        self.fusion = AttentionMultiRepresentationFusion(
            d_model=d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        
        # 3. Transcriptomic Pathway Cell Line Encoder
        self.cell_enc = CellLineEncoder(
            in_dim=config.cell_in_dim,
            d_model=d_model,
            n_pathways=config.n_pathways,
            use_pathway_projection=config.use_pathway_projection,
            dropout=config.dropout
        )
        
        # 4. Drug-Cell Cross-Attention Block (Phase 5)
        # Drug embedding attends to pathway cell tokens
        self.drug_cell_attn = DrugCellCrossAttention(
            d_model=d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        
        # 5. Optional Mutual Drug-Drug Attention
        if getattr(config, "enable_drug_drug_attention", False):
            self.drug_drug_attn = DrugDrugCrossAttention(
                d_model=d_model,
                n_heads=config.n_heads,
                dropout=config.dropout
            )
            
        # 6. Symmetric Combination Fusion Block (Phase 6)
        # Enforces mathematical permutation invariance when swapping Drug A and Drug B
        self.symmetric_fusion = SymmetricComboFusion(
            d_model=d_model,
            dropout=config.dropout
        )
        
        # 7. Constrained Biophysical Prediction Heads (Phase 7)
        # Sigmoid transforms project outputs to biological parameter boundaries
        self.heads = CancerComboPredictionHeads(config)
        
        # 8. Numerically Stable Bivariate Hill Solver with Masking (Phase 1 & Phase 8)
        # Masked Hill equation solver preserving zero-dose controls and non-zero gradients
        self.hill_solver = BivariateHillSolver(e0=100.0)

    def forward(
        self,
        drug_a_ids=None, drug_a_mask=None, drug_a_morgan=None, drug_a_desc=None,
        drug_b_ids=None, drug_b_mask=None, drug_b_morgan=None, drug_b_desc=None,
        cell_line=None, doses_a=None, doses_b=None,
        drug_a_emb=None, drug_b_emb=None
    ):
# ============================================================
# OLD CODE - DISABLED FOR MOLFORMER-ONLY ABLATION
# ============================================================
#         seq_a, pooled_a = self.molformer_enc(drug_a_ids, drug_a_mask)
#         morgan_a = self.morgan_enc(drug_a_morgan)
#         desc_a = self.descriptor_enc(drug_a_desc)
#         fused_a = self.fusion(pooled_a, morgan_a, desc_a)
#         
#         seq_b, pooled_b = self.molformer_enc(drug_b_ids, drug_b_mask)
#         morgan_b = self.morgan_enc(drug_b_morgan)
#         desc_b = self.descriptor_enc(drug_b_desc)
#         fused_b = self.fusion(pooled_b, morgan_b, desc_b)

# ============================================================
# NEW CODE - MOLFORMER-ONLY ABLATION
# ============================================================
        # 1. Encode Drug A using MolFormer ONLY (from precomputed emb or tokens)
        if drug_a_emb is not None:
            pooled_a = drug_a_emb
        else:
            _, pooled_a = self.molformer_enc(drug_a_ids, drug_a_mask)
        fused_a = self.fusion(pooled_a)

        # 2. Encode Drug B using MolFormer ONLY (from precomputed emb or tokens)
        if drug_b_emb is not None:
            pooled_b = drug_b_emb
        else:
            _, pooled_b = self.molformer_enc(drug_b_ids, drug_b_mask)
        fused_b = self.fusion(pooled_b)
        
        # 3. Encode Cell Line pathway embeddings
        cell_features = self.cell_enc(cell_line)
        
        # 4. Cross-attend Drug representations onto Cell Line tokens
        cond_a = self.drug_cell_attn(fused_a, cell_features)
        cond_b = self.drug_cell_attn(fused_b, cell_features)
        
        # 5. Mutual Drug-Drug Attention (if enabled)
        if hasattr(self, "drug_drug_attn") and getattr(self.config, "enable_drug_drug_attention", False):
            aware_a, aware_b = self.drug_drug_attn(cond_a, cond_b)
        else:
            aware_a, aware_b = cond_a, cond_b
            
        # 6. Symmetric Fusion for Permutation Invariance
        z_combo = self.symmetric_fusion(aware_a, aware_b)
        
        # 7. Predict Constrained Pharmacological Parameters
        e1, e2, e3, log_c1, log_c2, h1, h2, alpha = self.heads(aware_a, aware_b, z_combo)
        
        # 8. Solve 2D Dose-Response Matrix with Bivariate Hill Equation Solver
        y_pred = self.hill_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
        
        return y_pred, (e1, e2, e3, log_c1, log_c2, h1, h2, alpha)