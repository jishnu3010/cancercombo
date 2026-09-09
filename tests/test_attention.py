"""Verification test ensuring Cross-Attention is deactivated from the active model architecture."""

from cancer_combo_brics.config import ModelConfig
from cancer_combo_brics.model import CancerComboBRICS


def test_cross_attention_deactivated_in_model():
    model = CancerComboBRICS(ModelConfig())
    assert not hasattr(model, "cross_attention"), "Cross-attention module must be removed from active model instance."
