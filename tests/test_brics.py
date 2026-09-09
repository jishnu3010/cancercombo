"""Unit tests verifying FunctionalGroupCache and functional-group fragmentations."""

import os
import pytest
from cancer_combo_brics.chemistry.functional_group_fragments import extract_functional_group_fragments
from cancer_combo_brics.chemistry.cache import FunctionalGroupCache


def test_functional_group_extraction_determinism():
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    frags_1 = extract_functional_group_fragments(aspirin)
    frags_2 = extract_functional_group_fragments(aspirin)

    assert frags_1 == frags_2
    assert len(frags_1) > 0
    assert frags_1 == sorted(frags_1)


def test_functional_group_cache():
    db_path = "./data/test_fg_cache.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    cache = FunctionalGroupCache(db_path=db_path, radius=1)
    smiles = "CC(=O)Oc1ccccc1C(=O)O"

    assert cache.get(smiles) is None

    frags = cache.get_or_decompose(smiles)
    assert len(frags) > 0

    cached = cache.get(smiles)
    assert cached == frags

    del cache
    import gc
    gc.collect()

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            pass
