"""Verification test ensuring FiLM is deactivated from the active model architecture."""

import pytest
from tests.conftest import requires_gensim
from cancer_combo_brics.config import ModelConfig
from cancer_combo_brics.model import CancerComboBRICS

pytestmark = requires_gensim


def test_film_deactivated_in_model():
    model = CancerComboBRICS(ModelConfig())
    assert not hasattr(model, "film"), "FiLM module must be removed from active model instance."
