import torch
import torch.nn as nn
from config import ModelConfig
from blocks.molformer_encoder import MolFormerEncoder
from blocks.morgan_encoder import MorganEncoder
from blocks.descriptor_encoder import DescriptorEncoder
from blocks.fusion import AttentionMultiRepresentationFusion
from blocks.cell_encoder import CellLineEncoder
from blocks.drug_cell_attention import DrugCellCrossAttention
# DISABLED: Drug–Drug Attention
# from blocks.drug_drug_attention import DrugDrugCrossAttention
from blocks.shared_feature import SymmetricComboFusion
from blocks.prediction_heads import CancerComboPredictionHeads
from blocks.hill_equation import BivariateHillSolver
from typing import Tuple, Optional

class CancerCombo(nn.Module):
    """CancerCombo End-to-End Deep Learning Architecture for Dose-Response Prediction."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Drug encoders
        self.molformer_enc = MolFormerEncoder(
            model_name=config.molformer_model_name,
            d_model=config.d_model,
            use_pretrained=config.use_pretrained_molformer
        )
        self.morgan_enc = MorganEncoder(
            in_dim=config.morgan_in_dim,
            d_model=config.d_model,
            dropout=config.dropout
        )
        self.descriptor_enc = DescriptorEncoder(
            in_dim=config.descriptor_in_dim,
            d_model=config.d_model,
            dropout=config.dropout
        )
        
        # Feature fusion
        self.fusion = AttentionMultiRepresentationFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        
        # Cell profile encoder
        self.cell_enc = CellLineEncoder(
            in_dim=config.cell_in_dim,
            d_model=config.d_model,
            n_pathways=config.n_pathways,
            use_pathway_projection=config.use_pathway_projection,
            dropout=config.dropout
        )
        
        # Cross-attention layers
        self.drug_cell_attn = DrugCellCrossAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        # DISABLED: Drug–Drug Attention
        # self.drug_drug_attn = DrugDrugCrossAttention(
        #     d_model=config.d_model,
        #     n_heads=config.n_heads,
        #     dropout=config.dropout
        # )
        
        # Combo fusion mapping (permutation invariance)
        self.symmetric_fusion = SymmetricComboFusion(
            d_model=config.d_model,
            dropout=config.dropout
        )
        if not config.use_symmetric_fusion:
            self.asym_linear = nn.Linear(config.d_model * 2, config.d_model)
        
        # Prediction heads block
        self.heads = CancerComboPredictionHeads(config)
        
        # Bivariate Hill equation solver
        self.hill_solver = BivariateHillSolver(e0=100.0)
        
    def forward(
        self,
        drug_a_ids: torch.Tensor,
        drug_a_mask: torch.Tensor,
        drug_a_morgan: torch.Tensor,
        drug_a_desc: torch.Tensor,
        drug_b_ids: torch.Tensor,
        drug_b_mask: torch.Tensor,
        drug_b_morgan: torch.Tensor,
        drug_b_desc: torch.Tensor,
        cell_line: torch.Tensor,
        doses_a: torch.Tensor,
        doses_b: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Runs complete combination prediction forward pass.

        Args:
            drug_a_ids: Drug A token IDs (B, L).
            drug_a_mask: Drug A mask (B, L).
            drug_a_morgan: Drug A fingerprints (B, morgan_in_dim).
            drug_a_desc: Drug A physical descriptors (B, descriptor_in_dim).
            drug_b_ids: Drug B token IDs (B, L).
            drug_b_mask: Drug B mask (B, L).
            drug_b_morgan: Drug B fingerprints (B, morgan_in_dim).
            drug_b_desc: Drug B physical descriptors (B, descriptor_in_dim).
            cell_line: Cell line expressions (B, cell_in_dim).
            doses_a: Doses grid for drug A (B, M).
            doses_b: Doses grid for drug B (B, N).

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]: Viability predicted matrix and parameter tuple.
        """
        # Step 1: Encode Drug A chemical representations
        seq_a, pooled_a = self.molformer_enc(drug_a_ids, drug_a_mask)
        morgan_a = self.morgan_enc(drug_a_morgan)
        desc_a = self.descriptor_enc(drug_a_desc)
        
        # Step 2: Encode Drug B chemical representations
        seq_b, pooled_b = self.molformer_enc(drug_b_ids, drug_b_mask)
        morgan_b = self.morgan_enc(drug_b_morgan)
        desc_b = self.descriptor_enc(drug_b_desc)
        
        # Step 3: Fused Drug representations
        fused_a = self.fusion(pooled_a, morgan_a, desc_a)
        fused_b = self.fusion(pooled_b, morgan_b, desc_b)
        
        # Step 4: Encode Cell line profile
        cell_features = self.cell_enc(cell_line)
        
        # Step 5: Drug-Cell Cross Attention
        cond_a = self.drug_cell_attn(fused_a, cell_features)
        cond_b = self.drug_cell_attn(fused_b, cell_features)
        
        # Step 6: DISABLED: Drug–Drug Attention
        # if self.config.enable_drug_drug_attention:
        #     aware_a, aware_b = self.drug_drug_attn(cond_a, cond_b)
        # else:
        #     aware_a, aware_b = cond_a, cond_b
        aware_a, aware_b = cond_a, cond_b
            
        # Step 7: Symmetric Combination Fusion (ensures permutation invariance)
        if self.config.use_symmetric_fusion:
            z_combo = self.symmetric_fusion(aware_a, aware_b)
        else:
            z_combo = torch.cat([aware_a, aware_b], dim=-1)
            z_combo = self.asym_linear(z_combo)
            
        # Step 8: Predict Pharmacological parameters
        e1, e2, e3, log_c1, log_c2, h1, h2, alpha = self.heads(z_combo)
        
        # Step 9: Differentiable Hill Solver
        y_pred = self.hill_solver(
            doses_a=doses_a,
            doses_b=doses_b,
            e1=e1,
            e2=e2,
            e3=e3,
            log_c1=log_c1,
            log_c2=log_c2,
            h1=h1,
            h2=h2,
            alpha=alpha
        )
        
        params = (e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
        return y_pred, params
