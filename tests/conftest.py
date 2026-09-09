"""Shared pytest configuration and fixtures for CancerCombo test suite.

Handles graceful skipping of tests that require gensim/mol2vec when those
libraries are not installed (e.g. on Python 3.14 without C++ Build Tools).
"""

from __future__ import annotations

import pytest


def _gensim_available() -> bool:
    """Return True if gensim >= 4.0 is importable and functional."""
    try:
        import gensim
        from gensim.models import Word2Vec  # noqa: F401
        # Ensure it is gensim >= 4.x (has key_to_index API)
        major = int(gensim.__version__.split(".")[0])
        return major >= 4
    except Exception:
        return False


def _mol2vec_available() -> bool:
    """Return True if mol2vec is importable."""
    try:
        from mol2vec.features import mol2alt_sentence  # noqa: F401
        return True
    except Exception:
        return False


GENSIM_AVAILABLE = _gensim_available()
MOL2VEC_AVAILABLE = _mol2vec_available()

#: Pytest skip mark for tests that require gensim >= 4.0
requires_gensim = pytest.mark.skipif(
    not GENSIM_AVAILABLE,
    reason=(
        "gensim>=4.0 not installed. "
        "On Python 3.14, install Microsoft C++ Build Tools then: "
        "pip install 'gensim>=4.1.0,<5.0' mol2vec"
    ),
)

#: Pytest skip mark for tests that require the pretrained Mol2Vec model file
requires_mol2vec_model = pytest.mark.skipif(
    not GENSIM_AVAILABLE,
    reason=(
        "gensim>=4.0 not installed — cannot load Mol2Vec pretrained model. "
        "On Python 3.14, install Microsoft C++ Build Tools then: "
        "pip install 'gensim>=4.1.0,<5.0' mol2vec"
    ),
)
