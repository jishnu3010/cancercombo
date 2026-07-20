import torch
try:
    import pytest
except ImportError:
    pytest = None
from config import ModelConfig
from blocks.molformer_encoder import MolFormerEncoder
from blocks.morgan_encoder import MorganEncoder
from blocks.descriptor_encoder import DescriptorEncoder
from blocks.fusion import AttentionMultiRepresentationFusion
from blocks.cell_encoder import CellLineEncoder
from blocks.drug_cell_attention import DrugCellCrossAttention

def test_encoders_and_fusion_shapes():
    """Verify that individual intermediate dimensions match exactly across blocks."""
    config = ModelConfig(
        d_model=256, n_heads=4, d_ff=512, dropout=0.1,
        molformer_in_dim=768, morgan_in_dim=2048, descriptor_in_dim=200,
        cell_in_dim=20000, use_pathway_projection=True, n_pathways=300,
        molformer_model_name="ibm/MoLFormer-XL-CIMA-100M", use_pretrained_molformer=False,
        enable_drug_drug_attention=False, use_symmetric_fusion=True,
        e_min=0.0, e_max=100.0, c_min=1e-6, c_max=1e3, h_min=0.1, h_max=10.0,
        alpha_min=1e-4, alpha_max=100.0
    )
    batch_size = 4
    
    molformer = MolFormerEncoder(d_model=config.d_model, vocab_size=100)
    morgan = MorganEncoder(in_dim=config.morgan_in_dim, d_model=config.d_model)
    descriptor = DescriptorEncoder(in_dim=config.descriptor_in_dim, d_model=config.d_model)
    fusion = AttentionMultiRepresentationFusion(d_model=config.d_model)
    cell = CellLineEncoder(in_dim=config.cell_in_dim, d_model=config.d_model, n_pathways=config.n_pathways)
    drug_cell = DrugCellCrossAttention(d_model=config.d_model)
    
    ids = torch.randint(1, 20, (batch_size, 128))
    mask = torch.ones(batch_size, 128)
    morgan_in = torch.randn(batch_size, config.morgan_in_dim)
    desc_in = torch.randn(batch_size, config.descriptor_in_dim)
    cell_in = torch.randn(batch_size, config.cell_in_dim)
    
    seq_emb, pooled_emb = molformer(ids, mask)
    assert pooled_emb.shape == (batch_size, config.d_model)
    
    morgan_emb = morgan(morgan_in)
    assert morgan_emb.shape == (batch_size, config.d_model)
    
    desc_emb = descriptor(desc_in)
    assert desc_emb.shape == (batch_size, config.d_model)
    
    fused = fusion(pooled_emb, morgan_emb, desc_emb)
    assert fused.shape == (batch_size, config.d_model)
    
    cell_emb = cell(cell_in)
    assert cell_emb.shape == (batch_size, config.n_pathways, config.d_model)
    
    cond_drug = drug_cell(fused, cell_emb)
    assert cond_drug.shape == (batch_size, config.d_model)


if __name__ == "__main__":
    test_encoders_and_fusion_shapes()
    print("ALL FORWARD SHAPE TESTS PASSED SUCCESSFULLY!")

