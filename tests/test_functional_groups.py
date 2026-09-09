"""Unit tests for functional group fragmentation and context preservation."""

import pytest
from cancer_combo_brics.chemistry.functional_group_fragments import (
    extract_functional_group_fragments,
    extract_fragment_with_context,
)
import rdkit.Chem as Chem


def test_valid_smiles_extraction():
    # Acetaminophen
    smiles = "CC(=O)Nc1ccc(O)cc1"
    frags = extract_functional_group_fragments(smiles)
    assert isinstance(frags, list)
    assert len(frags) > 0
    # Every returned fragment must be a valid non-empty SMILES string
    for f in frags:
        assert isinstance(f, str) and len(f) > 0


def test_multiple_and_repeated_functional_groups():
    # Aspirin: carboxylic acid + ester + aromatic ring
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    frags = extract_functional_group_fragments(smiles)
    assert len(frags) >= 2


def test_adjacent_functional_groups():
    # Oxamide: adjacent amide groups
    smiles = "NC(=O)C(=O)N"
    frags = extract_functional_group_fragments(smiles)
    assert len(frags) > 0


def test_no_recognized_functional_group_fallback():
    # Simple alkane with no standard functional groups
    smiles = "CCCC"
    frags = extract_functional_group_fragments(smiles)
    assert len(frags) == 1
    # Canonical fallback
    mol = Chem.MolFromSmiles(smiles)
    expected = Chem.MolToSmiles(mol, canonical=True)
    assert frags[0] == expected


def test_invalid_smiles():
    smiles = "INVALID_SMILES_123"
    frags = extract_functional_group_fragments(smiles)
    assert len(frags) == 1
    assert frags[0] == smiles


def test_context_preservation():
    # Acetaminophen amide match
    smiles = "CC(=O)Nc1ccc(O)cc1"
    mol = Chem.MolFromSmiles(smiles)
    # Core amide indices (C, O, N)
    match_indices = (1, 2, 3)
    frag = extract_fragment_with_context(mol, match_indices, radius=1)
    assert frag is not None
    # Context should retain neighboring atoms (e.g. methyl C or aromatic c)
    assert len(frag) > 4  # More than just C(=O)N
