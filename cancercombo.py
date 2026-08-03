import torch
import torch.nn as nn
from typing import Tuple

from blocks.drug_cell_encoder import DeepSynBaDrugCellEncoder
from blocks.prediction_heads import CancerComboPredictionHeads
from blocks.hill_equation import BivariateHillSolver

class CancerCombo(nn.Module):
    """DeepSynBa-inspired architecture for dose-response surface prediction.
    
    Processes multi-modal drug features (Morgan fingerprints + RDKit continuous descriptors)
    concatenated with landmark transcriptomic cell features via DeepSynBa Drug-Cell Encoder MLPs,
    concatenates dual drug representations into a unified feature representation, predicts biophysical
    Hill parameters using DeepSynBa Prediction Heads, and computes dose-response matrices via
    the Bivariate Hill Equation Solver.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = getattr(config, "d_model", 256)
        
        morgan_dim = getattr(config, "morgan_in_dim", 2048)
        descriptor_dim = getattr(config, "descriptor_in_dim", 200)
        cell_dim = getattr(config, "cell_in_dim", 976)
        in_dim = morgan_dim + descriptor_dim + cell_dim # 2048 + 200 + 976 = 3224
        
        # 1. DeepSynBa Drug-Cell Encoder MLP
        self.drug_cell_encoder = DeepSynBaDrugCellEncoder(
            in_dim=in_dim,
            d_model=d_model,
            dropout=getattr(config, "dropout", 0.2)
        )

        # 2. DeepSynBa Prediction Heads for Bivariate Hill parameters
        self.heads = CancerComboPredictionHeads(config)
        
        # 3. Bivariate Hill Solver
        self.hill_solver = BivariateHillSolver(e0=100.0)

#######################################################
# OLD CODE - CANCERCOMBO ATTENTION
#######################################################
#         self.morgan_enc = MorganEncoder(...)
#         self.descriptor_enc = DescriptorEncoder(...)
#         self.fusion = AttentionMultiRepresentationFusion(...)
#         self.cell_enc = CellLineEncoder(...)
#         self.drug_cell_attn = DrugCellCrossAttention(...)
#         if getattr(config, "enable_drug_drug_attention", False):
#             self.drug_drug_attn = DrugDrugCrossAttention(...)
#         self.symmetric_fusion = SymmetricComboFusion(...)
#######################################################

    def forward(
        self,
        drug_a_ids=None, drug_a_mask=None, drug_a_morgan=None, drug_a_desc=None,
        drug_b_ids=None, drug_b_mask=None, drug_b_morgan=None, drug_b_desc=None,
        cell_line=None, doses_a=None, doses_b=None,
        drug_a_emb=None, drug_b_emb=None
    ):
#######################################################
# OLD CODE - CANCERCOMBO ATTENTION
#######################################################
#         morgan_a = self.morgan_enc(drug_a_morgan)
#         desc_a = self.descriptor_enc(drug_a_desc)
#         fused_a = self.fusion(morgan_a, desc_a)
#
#         morgan_b = self.morgan_enc(drug_b_morgan)
#         desc_b = self.descriptor_enc(drug_b_desc)
#         fused_b = self.fusion(morgan_b, desc_b)
#         
#         cell_features = self.cell_enc(cell_line)
#         cond_a = self.drug_cell_attn(fused_a, cell_features)
#         cond_b = self.drug_cell_attn(fused_b, cell_features)
#         
#         if hasattr(self, "drug_drug_attn") and getattr(self.config, "enable_drug_drug_attention", False):
#             aware_a, aware_b = self.drug_drug_attn(cond_a, cond_b)
#         else:
#             aware_a, aware_b = cond_a, cond_b
#             
#         z_combo = self.symmetric_fusion(aware_a, aware_b)
#         e1, e2, e3, log_c1, log_c2, h1, h2, alpha = self.heads(aware_a, aware_b, z_combo)
#######################################################

        # 1. Concatenate Drug A (Morgan + RDKit Descriptors) and Cell Line Gene Expression
        in_a = torch.cat([drug_a_morgan, drug_a_desc, cell_line], dim=1) # (B, 3224)

        # 2. Concatenate Drug B (Morgan + RDKit Descriptors) and Cell Line Gene Expression
        in_b = torch.cat([drug_b_morgan, drug_b_desc, cell_line], dim=1) # (B, 3224)

        # 3. Pass through DeepSynBa Drug-Cell Encoder MLPs
        rep_a = self.drug_cell_encoder(in_a) # (B, d_model)
        rep_b = self.drug_cell_encoder(in_b) # (B, d_model)

        # 4. Concatenate Drug A + Drug B into Unified Representation
        unified_rep = torch.cat([rep_a, rep_b], dim=1) # (B, 2 * d_model) = (B, 512)

        # 5. Predict Biophysical Parameters with DeepSynBa Prediction Heads
        e1, e2, e3, log_c1, log_c2, h1, h2, alpha = self.heads(unified_rep)

        # 6. Solve 2D Dose-Response Matrix with Bivariate Hill Equation Solver
        y_pred = self.hill_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)

        return y_pred, (e1, e2, e3, log_c1, log_c2, h1, h2, alpha)