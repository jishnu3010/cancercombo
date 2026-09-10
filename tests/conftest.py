"""Shared pytest configuration and fixtures for CancerCombo test suite.

Handles graceful skipping of tests that require gensim/mol2vec when those
libraries are not installed (e.g. on Python 3.14 without C++ Build Tools).
"""

from __future__ import annotations

import pytest


import os
import pytest


def _mol2vec_functional() -> bool:
    """Return True if genuine pretrained Mol2Vec can be loaded and tokenized."""
    try:
        possible_paths = [
            "data/model_300dim.pkl",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "model_300dim.pkl")),
        ]
        path = next((p for p in possible_paths if os.path.exists(p)), None)
        if path is None:
            return False
        from cancer_combo_brics.encoders.mol2vec_encoder import Mol2VecEncoder
        _ = Mol2VecEncoder(model_path=path)
        return True
    except Exception:
        return False


GENSIM_AVAILABLE = _mol2vec_functional()
MOL2VEC_AVAILABLE = GENSIM_AVAILABLE

#: Pytest skip mark for tests that require Mol2Vec
requires_gensim = pytest.mark.skipif(
    not GENSIM_AVAILABLE,
    reason="Pretrained Mol2Vec model or dependencies not functional.",
)

#: Pytest skip mark for tests that require the pretrained Mol2Vec model file
requires_mol2vec_model = pytest.mark.skipif(
    not GENSIM_AVAILABLE,
    reason="Pretrained Mol2Vec model or dependencies not functional.",
)
