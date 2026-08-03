import torch
import torch.nn as nn
from typing import Tuple

from blocks.morgan_encoder import MorganEncoder
from blocks.descriptor_encoder import DescriptorEncoder
from blocks.fusion import AttentionMultiRepresentationFusion
from blocks.cell_encoder import CellLineEncoder
from blocks.drug_cell_attention import DrugCellCrossAttention
from blocks.drug_drug_attention import DrugDrugCrossAttention
from blocks.prediction_heads import CancerComboPredictionHeads
from blocks.hill_equation import BivariateHillSolver

class CancerCombo(nn.Module):
    """CancerCombo Ablation 2 architecture with Drug-Drug Cross Attention.
    
    Integrates Morgan fingerprints, RDKit physical descriptors, Attention Multi-Representation Fusion,
    Pathway Cell Line Encoder, Drug-Cell Cross Attention, Mutual Drug-Drug Cross Attention,
    Prediction Heads, and numerically stable Bivariate Hill Solver.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = getattr(config, "d_model", 256)
        
        # 1. Morgan Fingerprint Encoder
        self.morgan_enc = MorganEncoder(
            in_dim=getattr(config, "morgan_in_dim", 2048),
            d_model=d_model,
            dropout=getattr(config, "dropout", 0.1)
        )
        
        # 2. RDKit Continuous Descriptor Encoder
        self.descriptor_enc = DescriptorEncoder(
            in_dim=getattr(config, "descriptor_in_dim", 200),
            d_model=d_model,
            dropout=getattr(config, "dropout", 0.1)
        )

        # 3. Multi-Head Self-Attention Fusion Block (Fuses Morgan + Descriptors)
        self.fusion = AttentionMultiRepresentationFusion(
            d_model=d_model,
            n_heads=getattr(config, "n_heads", 4),
            dropout=getattr(config, "dropout", 0.1)
        )
        
        # 4. Transcriptomic Pathway Cell Line Encoder
        self.cell_enc = CellLineEncoder(
            in_dim=getattr(config, "cell_in_dim", 976),
            d_model=d_model,
            n_pathways=getattr(config, "n_pathways", 300),
            use_pathway_projection=getattr(config, "use_pathway_projection", True),
            dropout=getattr(config, "dropout", 0.1)
        )
        
        # 5. Drug-Cell Cross-Attention Block
        self.drug_cell_attn = DrugCellCrossAttention(
            d_model=d_model,
            n_heads=getattr(config, "n_heads", 4),
            dropout=getattr(config, "dropout", 0.1)
        )
        
        # 6. Mutual Drug-Drug Cross-Attention Block
        self.drug_drug_attn = DrugDrugCrossAttention(
            d_model=d_model,
            n_heads=getattr(config, "n_heads", 4),
            dropout=getattr(config, "dropout", 0.1)
        )
        
        # 7. Prediction Heads
        self.heads = CancerComboPredictionHeads(config)
        
        # 8. Bivariate Hill Solver
        self.hill_solver = BivariateHillSolver(e0=100.0)

    def forward(
        self,
        drug_a_ids=None, drug_a_mask=None, drug_a_morgan=None, drug_a_desc=None,
        drug_b_ids=None, drug_b_mask=None, drug_b_morgan=None, drug_b_desc=None,
        cell_line=None, doses_a=None, doses_b=None,
        drug_a_emb=None, drug_b_emb=None
    ):
        # 1. Encode Drug A (Morgan + RDKit Descriptors)
        morgan_a = self.morgan_enc(drug_a_morgan)
        desc_a = self.descriptor_enc(drug_a_desc)
        fused_a = self.fusion(morgan_a, desc_a)

        # 2. Encode Drug B (Morgan + RDKit Descriptors)
        morgan_b = self.morgan_enc(drug_b_morgan)
        desc_b = self.descriptor_enc(drug_b_desc)
        fused_b = self.fusion(morgan_b, desc_b)
        
        # 3. Encode Cell Line pathway embeddings
        cell_features = self.cell_enc(cell_line)
        
        # 4. Drug-Cell Cross-Attention (Conditioning)
        cond_a = self.drug_cell_attn(fused_a, cell_features) # (B, d_model)
        cond_b = self.drug_cell_attn(fused_b, cell_features) # (B, d_model)

########################################################
# OLD CODE
# Simple Concatenation
########################################################
#       unified_rep = torch.cat([cond_a, cond_b], dim=1) # (B, 2 * d_model)

########################################################
# NEW CODE
# Drug–Drug Cross Attention
########################################################
        aware_a, aware_b = self.drug_drug_attn(cond_a, cond_b) # (B, d_model), (B, d_model)
        enhanced_pair_rep = torch.cat([aware_a, aware_b], dim=1) # (B, 2 * d_model)

        # 6. Predict Biophysical Parameters
        e1, e2, e3, log_c1, log_c2, h1, h2, alpha = self.heads(enhanced_pair_rep)

        # 7. Solve 2D Dose-Response Matrix with Bivariate Hill Equation Solver
        y_pred = self.hill_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)

        return y_pred, (e1, e2, e3, log_c1, log_c2, h1, h2, alpha)