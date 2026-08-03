import torch
import torch.nn as nn
from typing import Tuple

from blocks.morgan_encoder import MorganEncoder
from blocks.descriptor_encoder import DescriptorEncoder
from blocks.fusion import AttentionMultiRepresentationFusion
from blocks.drug_cell_encoder import DeepSynBaDrugCellEncoder
from blocks.prediction_heads import CancerComboPredictionHeads
from blocks.hill_equation import BivariateHillSolver
# Disabled for Multi-Representation Fusion architecture.
# from blocks.drug_drug_attention import DrugDrugCrossAttention

class CancerCombo(nn.Module):
    """DeepSynBa-inspired architecture for dose-response surface prediction with Attention-Based Multi-Representation Fusion.
    
    Fuses multi-modal drug features (Morgan fingerprints + RDKit continuous descriptors) using
    Attention-Based Multi-Representation Fusion, concatenates enhanced drug embeddings with landmark
    transcriptomic cell features, processes through DeepSynBa Drug-Cell Encoder MLPs, concatenates
    dual drug representations into a shared pair representation, predicts biophysical Hill parameters using
    DeepSynBa Prediction Heads, and computes dose-response matrices via the Bivariate Hill Equation Solver.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = getattr(config, "d_model", 256)
        
        morgan_dim = getattr(config, "morgan_in_dim", 2048)
        descriptor_dim = getattr(config, "descriptor_in_dim", 200)
        cell_dim = getattr(config, "cell_in_dim", 976)
        
        # 1. Encoders for Morgan Fingerprints and RDKit Descriptors
        self.morgan_enc = MorganEncoder(
            in_dim=morgan_dim,
            d_model=d_model,
            dropout=getattr(config, "dropout", 0.1)
        )
        self.descriptor_enc = DescriptorEncoder(
            in_dim=descriptor_dim,
            d_model=d_model,
            dropout=getattr(config, "dropout", 0.1)
        )

        # 2. Attention-Based Multi-Representation Fusion (Morgan + RDKit Descriptors)
        self.fusion = AttentionMultiRepresentationFusion(
            d_model=d_model,
            n_heads=getattr(config, "n_heads", 4),
            dropout=getattr(config, "dropout", 0.1)
        )

        # 3. DeepSynBa Drug-Cell Encoder MLP (Fused Drug Embedding + Cell Features)
        in_dim = d_model + cell_dim # 256 + 976 = 1232
        self.drug_cell_encoder = DeepSynBaDrugCellEncoder(
            in_dim=in_dim,
            d_model=d_model,
            dropout=getattr(config, "dropout", 0.2)
        )

        # Disabled for Multi-Representation Fusion architecture.
        # self.drug_drug_attn = DrugDrugCrossAttention(
        #     d_model=d_model,
        #     n_heads=getattr(config, "n_heads", 4),
        #     dropout=getattr(config, "dropout", 0.1)
        # )

        # 4. DeepSynBa Prediction Heads for Bivariate Hill parameters
        self.heads = CancerComboPredictionHeads(config)
        
        # 5. Bivariate Hill Solver
        self.hill_solver = BivariateHillSolver(e0=100.0)

    def forward(
        self,
        drug_a_ids=None, drug_a_mask=None, drug_a_morgan=None, drug_a_desc=None,
        drug_b_ids=None, drug_b_mask=None, drug_b_morgan=None, drug_b_desc=None,
        cell_line=None, doses_a=None, doses_b=None,
        drug_a_emb=None, drug_b_emb=None
    ):
        # 1. Encode & Fuse Drug A representations (Morgan + RDKit Descriptors)
        morgan_a = self.morgan_enc(drug_a_morgan) # (B, d_model)
        desc_a = self.descriptor_enc(drug_a_desc) # (B, d_model)
        fused_a = self.fusion(morgan_a, desc_a)   # (B, d_model)

        # 2. Encode & Fuse Drug B representations (Morgan + RDKit Descriptors)
        morgan_b = self.morgan_enc(drug_b_morgan) # (B, d_model)
        desc_b = self.descriptor_enc(drug_b_desc) # (B, d_model)
        fused_b = self.fusion(morgan_b, desc_b)   # (B, d_model)

        # 3. Concatenate Enhanced Drug Embeddings with Cell Line Gene Expression
        in_a = torch.cat([fused_a, cell_line], dim=1) # (B, d_model + cell_dim) = (B, 1232)
        in_b = torch.cat([fused_b, cell_line], dim=1) # (B, d_model + cell_dim) = (B, 1232)

        # 4. Pass through DeepSynBa Drug-Cell Encoder MLPs
        rep_a = self.drug_cell_encoder(in_a) # (B, d_model)
        rep_b = self.drug_cell_encoder(in_b) # (B, d_model)

        # Disabled for Multi-Representation Fusion architecture.
        # aware_a, aware_b = self.drug_drug_attn(rep_a, rep_b) # (B, d_model), (B, d_model)
        # enhanced_pair_rep = torch.cat([aware_a, aware_b], dim=1) # (B, 2 * d_model) = (B, 512)

        # 5. Concatenate Drug A and Drug B conditioned representations into shared feature vector
        shared_feature_vec = torch.cat([rep_a, rep_b], dim=1) # (B, 2 * d_model) = (B, 512)

        # 6. Predict Biophysical Parameters with DeepSynBa Prediction Heads
        e1, e2, e3, log_c1, log_c2, h1, h2, alpha = self.heads(shared_feature_vec)

        # 7. Solve 2D Dose-Response Matrix with Bivariate Hill Equation Solver
        y_pred = self.hill_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)

        return y_pred, (e1, e2, e3, log_c1, log_c2, h1, h2, alpha)