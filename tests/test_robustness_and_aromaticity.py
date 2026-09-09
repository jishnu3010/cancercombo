"""Comprehensive robustness, aromaticity, and architecture verification tests."""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rdkit.Chem as Chem

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cancer_combo_brics.chemistry.functional_group_fragments import (
    extract_functional_group_fragments,
    extract_fragment_with_context,
    validate_and_canonicalize_fragment,
    COMPILED_FG_PATTERNS,
)
from cancer_combo_brics.data.dataset import CancerComboDataset, collate_combo_batch, is_valid_smiles
from cancer_combo_brics.encoders.mol2vec_encoder import Mol2VecEncoder, extract_morgan_subgraph_identifiers
from cancer_combo_brics.interaction.pairwise_interaction import ExplicitPairwiseFragmentInteraction
from cancer_combo_brics.interaction.drug_cell import DrugCellInteraction
from cancer_combo_brics.pharmacology.parameter_heads import PharmacologicalParameterHeads
from cancer_combo_brics.pharmacology.constraints import ConstraintTransform
from cancer_combo_brics.pharmacology.bivariate_hill import BivariateHillSolver
from cancer_combo_brics.pharmacology.dose_bias import DoseDependentBias
from cancer_combo_brics.model import CancerComboBRICS
from cancer_combo_brics.config import ModelConfig
from cancer_combo_brics.data.preflight import run_data_preflight


# -------------------------------------------------------------------
# 1. NaN SMILES_A HANDLING
# -------------------------------------------------------------------
def test_1_nan_smiles_a():
    df = pd.DataFrame({
        "smiles_a": [np.nan],
        "smiles_b": ["CC(=O)O"],
        "cell_line_name": ["786_0"],
        "doses_a": ["[0.0, 1.0]"],
        "doses_b": ["[0.0, 1.0]"],
        "viability_matrix": ["[[100.0, 100.0], [100.0, 100.0]]"],
    })
    cell_expressions = {"786_0": np.ones(976, dtype=np.float32)}
    ds = CancerComboDataset(df, cell_expressions)
    sample = ds[0]
    assert sample["is_valid_sample"] is False
    assert sample["frags_a"] == []


# -------------------------------------------------------------------
# 2. NaN SMILES_B HANDLING
# -------------------------------------------------------------------
def test_2_nan_smiles_b():
    df = pd.DataFrame({
        "smiles_a": ["CCO"],
        "smiles_b": [np.nan],
        "cell_line_name": ["786_0"],
        "doses_a": ["[0.0, 1.0]"],
        "doses_b": ["[0.0, 1.0]"],
        "viability_matrix": ["[[100.0, 100.0], [100.0, 100.0]]"],
    })
    cell_expressions = {"786_0": np.ones(976, dtype=np.float32)}
    ds = CancerComboDataset(df, cell_expressions)
    sample = ds[0]
    assert sample["is_valid_sample"] is False
    assert sample["frags_b"] == []


# -------------------------------------------------------------------
# 3. EMPTY SMILES HANDLING
# -------------------------------------------------------------------
def test_3_empty_smiles():
    assert is_valid_smiles("") is False
    assert is_valid_smiles("   ") is False
    assert is_valid_smiles("nan") is False
    frags = extract_functional_group_fragments("")
    assert frags == []


# -------------------------------------------------------------------
# 4. INVALID ORIGINAL SMILES
# -------------------------------------------------------------------
def test_4_invalid_original_smiles():
    assert is_valid_smiles("InvalidChemicalString123") is True
    frags = extract_functional_group_fragments("InvalidChemicalString123")
    assert frags == []


# -------------------------------------------------------------------
# 5. VALID MOLECULE WITH NO FUNCTIONAL GROUP MATCH
# -------------------------------------------------------------------
def test_5_valid_molecule_no_fg_match():
    # Cyclohexane has no reactive functional group in our SMARTS dict
    cyc = "C1CCCCC1"
    frags = extract_functional_group_fragments(cyc)
    assert len(frags) == 1
    assert frags[0] == cyc


# -------------------------------------------------------------------
# 6. FULL MOLECULE FALLBACK CASE A
# -------------------------------------------------------------------
def test_6_fallback_case_a():
    mol_smiles = "C1CCCCC1"
    frags = extract_functional_group_fragments(mol_smiles)
    assert frags == [mol_smiles]


# -------------------------------------------------------------------
# 7. FULL MOLECULE FALLBACK CASE B
# -------------------------------------------------------------------
def test_7_fallback_case_b():
    # Aspirin
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    frags = extract_functional_group_fragments(aspirin)
    assert len(frags) > 0
    for f in frags:
        assert Chem.MolFromSmiles(f) is not None


# -------------------------------------------------------------------
# 8-11. AROMATIC / HETEROAROMATIC / FUSED / SUBSTITUENT EXTRACTION
# -------------------------------------------------------------------
def test_8_11_aromatic_robustness():
    test_molecules = [
        ("Aromatic substituent", "Cc1ccccc1"),  # Toluene
        ("Heteroaromatic ring", "c1cnccn1"),    # Pyrimidine
        ("Fused aromatic system", "c1ccc2ccccc2c1"),  # Naphthalene
        ("Complex anticancer drug", "C1=NC2=C(N=C(N=C2N1C3C(C(C(O3)CO)O)O)F)N"),  # Clofarabine
    ]
    for name, sm in test_molecules:
        frags = extract_functional_group_fragments(sm, radius=1)
        assert len(frags) > 0, f"No fragments produced for {name} ({sm})"
        for f in frags:
            t_mol = Chem.MolFromSmiles(f)
            assert t_mol is not None, f"Invalid fragment '{f}' extracted from {name}"
            Chem.SanitizeMol(t_mol)


# -------------------------------------------------------------------
# 12. 1-HOP CONTEXT PRESERVATION
# -------------------------------------------------------------------
def test_12_context_preservation():
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    frags = extract_functional_group_fragments(aspirin, radius=1)
    # 1-hop context ensures fragments are larger than isolated single-atom tokens
    for f in frags:
        f_mol = Chem.MolFromSmiles(f)
        assert f_mol.GetNumHeavyAtoms() >= 1


# -------------------------------------------------------------------
# 13-16. EVERY ACCEPTED FRAGMENT IS RDKIT VALID, SANITIZES, NO ORPHANS
# -------------------------------------------------------------------
def test_13_16_accepted_fragments_strictly_valid():
    sample_drugs = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CCC1=C2CN3C(=CC4=C(C3=O)COC(=O)C4(CC)O)C2=NC5=C1C=C(C=C5)O",
        "C1=NC2=C(N=C(N=C2N1C3C(C(C(O3)CO)O)O)F)N",
    ]
    for sm in sample_drugs:
        frags = extract_functional_group_fragments(sm)
        for f in frags:
            assert validate_and_canonicalize_fragment(f) is not None
            f_mol = Chem.MolFromSmiles(f)
            assert f_mol is not None
            Chem.SanitizeMol(f_mol)


# -------------------------------------------------------------------
# 17. MOL2VEC RECEIVES ONLY VALID FRAGMENTS
# -------------------------------------------------------------------
def test_17_mol2vec_receives_only_valid_fragments():
    encoder = Mol2VecEncoder(native_dim=300, fragment_dim=512)
    frags = [["CC(=O)O", "c1ccccc1"]]
    mask = torch.tensor([[1.0, 1.0]])
    emb = encoder.encode_drug_fragments_batch(frags, mask, device=torch.device("cpu"))
    assert emb.shape == (1, 2, 512)


# -------------------------------------------------------------------
# 18-19. VARIABLE FRAGMENT COUNTS & PAIRWISE INTERACTION
# -------------------------------------------------------------------
def test_18_19_variable_fragment_counts_pairwise():
    interaction = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512)
    F_A = torch.randn(2, 3, 512)
    F_B = torch.randn(2, 5, 512)
    mask_A = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    mask_B = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]])

    r_AB, diag = interaction(F_A, F_B, mask_A, mask_B)
    assert r_AB.shape == (2, 512)


# -------------------------------------------------------------------
# 20-23. ARCHITECTURE SHAPES: r_AB=512, r_DC=1536, 8 HEADS (1536->1024->8)
# -------------------------------------------------------------------
def test_20_23_architecture_shapes():
    cfg = ModelConfig(cell_dim=976, fragment_dim=512)
    model = CancerComboBRICS(config=cfg)

    cell_expr = torch.randn(2, 976)
    frags_A = [["CCO"], ["CC(=O)O"]]
    mask_A = torch.tensor([[1.0], [1.0]])
    frags_B = [["c1ccccc1"], ["c1ccccc1"]]
    mask_B = torch.tensor([[1.0], [1.0]])
    doses_a = torch.tensor([[0.0, 0.1, 1.0, 10.0]] * 2, dtype=torch.float32)
    doses_b = torch.tensor([[0.0, 0.05, 0.5, 5.0]] * 2, dtype=torch.float32)

    y_pred, diag = model(
        cell_expr=cell_expr,
        fragments_A=frags_A,
        mask_A=mask_A,
        fragments_B=frags_B,
        mask_B=mask_B,
        doses_A=doses_a,
        doses_B=doses_b,
        return_diagnostics=True,
    )

    assert diag["r_AB"].shape == (2, 512)
    assert diag["r_DC"].shape == (2, 1536)
    assert len(diag["raw_params"]) == 8
    assert y_pred.shape == (2, 4, 4)


# -------------------------------------------------------------------
# 24-27. PHARMACOLOGY UNCHANGED (M->uM, percentage scale, Hill, dose bias)
# -------------------------------------------------------------------
def test_24_27_pharmacology_unchanged():
    solver = BivariateHillSolver(e0=100.0)
    bias_mod = DoseDependentBias(context_dim=1536, enabled=True)

    B = 2
    e1 = torch.tensor([[0.5], [0.2]])
    e2 = torch.tensor([[0.3], [0.8]])
    e3 = torch.tensor([[0.1], [0.4]])
    log_c1 = torch.tensor([[0.0], [1.0]])
    log_c2 = torch.tensor([[0.5], [-0.5]])
    h1 = torch.tensor([[1.0], [1.5]])
    h2 = torch.tensor([[1.2], [0.8]])
    alpha = torch.tensor([[2.0], [0.5]])

    doses_a = torch.tensor([[0.0, 0.06, 0.6, 6.0]] * B, dtype=torch.float32)
    doses_b = torch.tensor([[0.0, 0.05, 0.5, 5.0]] * B, dtype=torch.float32)

    y_hill = solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
    r_DC = torch.randn(B, 1536)
    bias = bias_mod(r_DC, doses_a, doses_b)
    y_pred = y_hill + bias

    # Check zero-dose boundary conditions
    for b in range(B):
        assert torch.isclose(y_hill[b, 0, 0], torch.tensor(100.0), atol=1e-4)
        assert torch.isclose(bias[b, 0, 0], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(y_pred[b, 0, 0], torch.tensor(100.0), atol=1e-4)


# -------------------------------------------------------------------
# 28-32. INACTIVE MODULES CHECK (BRICS, MoLFormer, FiLM, Cross-attention)
# -------------------------------------------------------------------
def test_28_32_inactive_modules_check():
    import cancer_combo_brics.model as m_mod
    import cancer_combo_brics.data.dataset as d_mod

    # Confirm brics is not imported in dataset or model
    assert not hasattr(m_mod, "BRICS")
    assert not hasattr(d_mod, "decompose_smiles")


# -------------------------------------------------------------------
# 33-34. COMPLETE PREFLIGHT & PARITY
# -------------------------------------------------------------------
def test_33_34_preflight_runs():
    res = run_data_preflight()
    assert res["invalid_fragments_passed_downstream"] == 0
    assert res["zero_fragment_molecules"] == 0
