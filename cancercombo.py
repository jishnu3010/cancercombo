import torch
import torch.nn as nn
from typing import Tuple

from blocks.morgan_encoder import MorganEncoder
from blocks.descriptor_encoder import DescriptorEncoder
from blocks.cell_encoder import CellLineEncoder
from blocks.drug_cell_attention import DrugCellCrossAttention
from blocks.prediction_heads import CancerComboPredictionHeads
from blocks.hill_equation import BivariateHillSolver

class CancerCombo(nn.Module):
    """CancerCombo Architecture (Ablation 8).
    
    Replaces DeepSynBa Drug-Cell Encoder MLP with Morgan Encoder, Descriptor Encoder,
    CellLineEncoder, and original Drug-Cell Cross Attention.
    Conditioned Drug A and Drug B representations are directly concatenated into a pair
    representation, passed to Prediction Heads to predict Hill parameters, and solved
    via the Bivariate Hill Equation Solver.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = getattr(config, "d_model", 256)
        n_heads = getattr(config, "n_heads", 4)
        dropout = getattr(config, "dropout", 0.1)
        
        morgan_dim = getattr(config, "morgan_in_dim", 2048)
        descriptor_dim = getattr(config, "descriptor_in_dim", 200)
        cell_dim = getattr(config, "cell_in_dim", 976)
        n_pathways = getattr(config, "n_pathways", 300)
        use_pathway_proj = getattr(config, "use_pathway_projection", True)
        
        # 1. Encoders
        self.morgan_enc = MorganEncoder(in_dim=morgan_dim, d_model=d_model, dropout=dropout)
        self.descriptor_enc = DescriptorEncoder(in_dim=descriptor_dim, d_model=d_model, dropout=dropout)
        self.cell_enc = CellLineEncoder(
            in_dim=cell_dim, d_model=d_model, n_pathways=n_pathways,
            use_pathway_projection=use_pathway_proj, dropout=dropout
        )

        # 2. Drug-Cell Cross Attention Module
        self.drug_cell_attn = DrugCellCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

        # 3. Prediction Heads for Bivariate Hill parameters
        self.heads = CancerComboPredictionHeads(config)
        
        # 4. Bivariate Hill Solver
        self.hill_solver = BivariateHillSolver(e0=100.0)

    def forward(
        self,
        drug_a_ids=None, drug_a_mask=None, drug_a_morgan=None, drug_a_desc=None,
        drug_b_ids=None, drug_b_mask=None, drug_b_morgan=None, drug_b_desc=None,
        cell_line=None, doses_a=None, doses_b=None,
        drug_a_emb=None, drug_b_emb=None
    ):
        # 1. Pass Drug A features through Morgan & Descriptor Encoders
        morgan_a = self.morgan_enc(drug_a_morgan) # (B, d_model)
        desc_a = self.descriptor_enc(drug_a_desc) # (B, d_model)
        drug_emb_a = morgan_a + desc_a # (B, d_model)

        # 2. Pass Drug B features through Morgan & Descriptor Encoders
        morgan_b = self.morgan_enc(drug_b_morgan) # (B, d_model)
        desc_b = self.descriptor_enc(drug_b_desc) # (B, d_model)
        drug_emb_b = morgan_b + desc_b # (B, d_model)

        # 3. Encode Cell Line transcriptomics
        cell_emb = self.cell_enc(cell_line) # (B, n_pathways, d_model)

        # 4. Condition Drug A and Drug B on Cell Line via Drug-Cell Cross Attention
        cond_a = self.drug_cell_attn(drug_emb_a, cell_emb) # (B, d_model)
        cond_b = self.drug_cell_attn(drug_emb_b, cell_emb) # (B, d_model)

        # 5. Direct Concatenation of Conditioned Drug Representations
        pair_rep = torch.cat([cond_a, cond_b], dim=1) # (B, 2 * d_model) = (B, 512)

        # 6. Predict Biophysical Parameters with Prediction Heads
        e1, e2, e3, log_c1, log_c2, h1, h2, alpha = self.heads(pair_rep)

        # 7. Solve 2D Dose-Response Matrix with Bivariate Hill Equation Solver
        y_pred = self.hill_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)

        return y_pred, (e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
