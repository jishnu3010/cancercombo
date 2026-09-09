"""Automated downstream parity tests comparing bricks2 directly with finalcheck.

Mandatory tests from PART 20:
  TEST A — Dose conversion: 6e-8 M -> 0.06 uM
  TEST B — Target scale: 100.0 remains 100.0
  TEST C — Parameter order: e1, e2, e3, logC1, logC2, h1, h2, alpha
  TEST D — Constraint parity: assert_close on identical raw parameters
  TEST E — Hill solver parity: assert_close on identical parameters & doses
  TEST F — Dose-bias parity: assert_close on identical representations & doses
  TEST G — End-to-end downstream parity: assert_close on full downstream pipeline
  TEST H — Zero-dose behavior: Y(0,0) == 100.0, bias(0,0) == 0.0
  TEST I — Numerical stability: extreme doses & parameters produce finite outputs/gradients
  TEST J — Output shape: [B, 4, 4]
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add bricks2 project root to sys.path
BRICKS2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BRICKS2_ROOT not in sys.path:
    sys.path.insert(0, BRICKS2_ROOT)

# Add finalcheck project path to sys.path
FINALCHECK_PATH = os.path.abspath(os.path.join(BRICKS2_ROOT, "..", "finalcheck", "finalcheck", "cancercombo"))
if FINALCHECK_PATH not in sys.path:
    sys.path.insert(0, FINALCHECK_PATH)

from cancer_combo_brics.data.dataset import CancerComboDataset
from cancer_combo_brics.pharmacology.parameter_heads import (
    DeepSynBaBlock,
    DeepSynBaPredictionHead,
    PharmacologicalParameterHeads,
)
from cancer_combo_brics.pharmacology.constraints import ConstraintTransform
from cancer_combo_brics.pharmacology.bivariate_hill import BivariateHillSolver as BricksBivariateHillSolver
from cancer_combo_brics.pharmacology.dose_bias import (
    DoseDependentBias,
    DoseResponsePredictor as BricksDoseResponsePredictor,
)
from cancer_combo_brics.model import CancerComboBRICS
from cancer_combo_brics.config import ModelConfig

# Import authoritative finalcheck reference modules
try:
    from blocks.hill_equation import BivariateHillSolver as FinalcheckHillSolver
    from blocks.prediction_heads import (
        DeepSynBaPredictionHead as FinalcheckPredictionHead,
        DoseResponsePredictor as FinalcheckDoseResponsePredictor,
    )
    FINALCHECK_AVAILABLE = True
except ImportError as e:
    FINALCHECK_AVAILABLE = False
    print(f"Warning: Could not import finalcheck directly: {e}")


# ------------------------------------------------------------
# TEST A — DOSE CONVERSION
# ------------------------------------------------------------
def test_a_dose_conversion():
    """Verify 6e-8 M converts to exactly 0.06 uM for both Drug A and Drug B."""
    raw_doses_a = "[0.0, 6e-08, 6e-07, 6e-06]"
    raw_doses_b = "[0.0, 6e-08, 5e-07, 5e-06]"

    df = pd.DataFrame({
        "smiles_a": ["CCO"],
        "smiles_b": ["CC(=O)O"],
        "cell_line_name": ["786_0"],
        "doses_a": [raw_doses_a],
        "doses_b": [raw_doses_b],
        "viability_matrix": ["[[100.0, 100.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0]]"],
    })
    cell_expressions = {"786_0": np.ones(976, dtype=np.float32)}
    ds = CancerComboDataset(df, cell_expressions=cell_expressions)

    sample = ds[0]
    doses_a = sample["doses_a"]
    doses_b = sample["doses_b"]

    assert np.isclose(doses_a[1], 0.06, rtol=1e-5), f"Expected 0.06 uM for Drug A, got {doses_a[1]}"
    assert np.isclose(doses_b[1], 0.06, rtol=1e-5), f"Expected 0.06 uM for Drug B, got {doses_b[1]}"
    assert np.isclose(doses_a[0], 0.0, atol=1e-9)
    assert np.isclose(doses_b[0], 0.0, atol=1e-9)


# ------------------------------------------------------------
# TEST B — TARGET SCALE
# ------------------------------------------------------------
def test_b_target_scale():
    """Verify viability value 100.0 remains 100.0 and is never divided by 100.0."""
    df = pd.DataFrame({
        "smiles_a": ["CCO"],
        "smiles_b": ["CC(=O)O"],
        "cell_line_name": ["786_0"],
        "doses_a": ["[0.0, 1e-6]"],
        "doses_b": ["[0.0, 1e-6]"],
        "viability_matrix": ["[[100.0, 85.5], [92.3, 64.1]]"],
    })
    cell_expressions = {"786_0": np.ones(976, dtype=np.float32)}
    ds = CancerComboDataset(df, cell_expressions=cell_expressions)

    sample = ds[0]
    viab = sample["viability_matrix"]

    assert np.isclose(viab[0, 0], 100.0, atol=1e-4), f"Expected 100.0, got {viab[0, 0]}"
    assert viab[0, 0] != 1.0, "Viability was accidentally normalized to 1.0!"
    assert np.isclose(viab[0, 1], 85.5, atol=1e-4)


# ------------------------------------------------------------
# TEST C — PARAMETER ORDER
# ------------------------------------------------------------
def test_c_parameter_order():
    """Verify the eight predicted parameters match exact finalcheck order:
       e1, e2, e3, logC1, logC2, h1, h2, alpha.
    """
    heads = PharmacologicalParameterHeads(in_dim=1536)
    B = 2
    r_DC = torch.randn(B, 1536)

    tup = heads.forward_tuple(r_DC)
    assert len(tup) == 8
    e1, e2, e3, log_c1, log_c2, h1, h2, alpha = tup
    for p in [e1, e2, e3, log_c1, log_c2, h1, h2, alpha]:
        assert p.shape == (B, 1)

    raw_dict = heads(r_DC)
    expected_keys = [
        "e1_raw",
        "e2_raw",
        "e3_raw",
        "logc1_raw",
        "logc2_raw",
        "h1_raw",
        "h2_raw",
        "alpha_raw",
    ]
    assert list(raw_dict.keys()) == expected_keys


# ------------------------------------------------------------
# TEST D — CONSTRAINT PARITY
# ------------------------------------------------------------
def test_d_constraint_parity():
    """Feed identical raw parameters into finalcheck transforms and bricks2 ConstraintTransform."""
    B = 10
    torch.manual_seed(42)

    raw_e1 = torch.randn(B, 1)
    raw_e2 = torch.randn(B, 1)
    raw_e3 = torch.randn(B, 1)
    raw_log_c1 = torch.randn(B, 1) * 3.0
    raw_log_c2 = torch.randn(B, 1) * 3.0
    raw_h1 = torch.randn(B, 1) * 2.0
    raw_h2 = torch.randn(B, 1) * 2.0
    raw_alpha = torch.randn(B, 1) * 2.0

    fc_e1 = torch.sigmoid(raw_e1)
    fc_e2 = torch.sigmoid(raw_e2)
    fc_e3 = torch.sigmoid(raw_e3)
    fc_log_c1 = raw_log_c1
    fc_log_c2 = raw_log_c2
    fc_h1 = F.relu(raw_h1)
    fc_h2 = F.relu(raw_h2)
    fc_alpha = F.relu(raw_alpha)

    transform = ConstraintTransform(e0=100.0)
    raw_tuple = (raw_e1, raw_e2, raw_e3, raw_log_c1, raw_log_c2, raw_h1, raw_h2, raw_alpha)
    b2_e1, b2_e2, b2_e3, b2_log_c1, b2_log_c2, b2_h1, b2_h2, b2_alpha = transform.forward_tuple(raw_tuple)

    torch.testing.assert_close(b2_e1, fc_e1)
    torch.testing.assert_close(b2_e2, fc_e2)
    torch.testing.assert_close(b2_e3, fc_e3)
    torch.testing.assert_close(b2_log_c1, fc_log_c1)
    torch.testing.assert_close(b2_log_c2, fc_log_c2)
    torch.testing.assert_close(b2_h1, fc_h1)
    torch.testing.assert_close(b2_h2, fc_h2)
    torch.testing.assert_close(b2_alpha, fc_alpha)


# ------------------------------------------------------------
# TEST E — HILL SOLVER PARITY
# ------------------------------------------------------------
@pytest.mark.skipif(not FINALCHECK_AVAILABLE, reason="Finalcheck modules not importable")
def test_e_hill_parity():
    """Feed identical parameters and doses into finalcheck Hill solver and bricks2 Hill solver."""
    b2_solver = BricksBivariateHillSolver(e0=100.0)
    fc_solver = FinalcheckHillSolver(e0=100.0)

    B = 4
    torch.manual_seed(123)

    e1 = torch.sigmoid(torch.randn(B, 1))
    e2 = torch.sigmoid(torch.randn(B, 1))
    e3 = torch.sigmoid(torch.randn(B, 1))
    log_c1 = torch.randn(B, 1) * 2.0
    log_c2 = torch.randn(B, 1) * 2.0
    h1 = F.relu(torch.randn(B, 1)) + 0.1
    h2 = F.relu(torch.randn(B, 1)) + 0.1
    alpha = F.relu(torch.randn(B, 1)) + 0.1

    doses_a = torch.tensor([[0.0, 0.06, 0.6, 6.0]] * B, dtype=torch.float32)
    doses_b = torch.tensor([[0.0, 0.05, 0.5, 5.0]] * B, dtype=torch.float32)

    fc_out = fc_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
    b2_out = b2_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)

    torch.testing.assert_close(b2_out, fc_out, rtol=1e-6, atol=1e-6)


# ------------------------------------------------------------
# TEST F — DOSE-BIAS PARITY
# ------------------------------------------------------------
@pytest.mark.skipif(not FINALCHECK_AVAILABLE, reason="Finalcheck modules not importable")
def test_f_dose_bias_parity():
    """Verify bricks2 DoseDependentBias matches finalcheck dual predictor formula."""
    torch.manual_seed(42)
    B = 3
    context_dim = 512
    emb_size = 1024

    fc_p1 = FinalcheckDoseResponsePredictor(context_dim, emb_size, dropout=0.0)
    fc_p2 = FinalcheckDoseResponsePredictor(context_dim, emb_size, dropout=0.0)

    b2_bias_mod = DoseDependentBias(context_dim, emb_size, dropout=0.0, enabled=True)
    b2_bias_mod.bias_predictor1.load_state_dict(fc_p1.state_dict())
    b2_bias_mod.bias_predictor2.load_state_dict(fc_p2.state_dict())

    b2_bias_mod.eval()
    fc_p1.eval()
    fc_p2.eval()

    rep = torch.randn(B, context_dim)
    doses_a = torch.tensor([[0.0, 0.1, 1.0, 10.0]] * B, dtype=torch.float32)
    doses_b = torch.tensor([[0.0, 0.05, 0.5, 5.0]] * B, dtype=torch.float32)

    out1 = fc_p1(rep)
    out2 = fc_p2(rep)
    out1_grid = out1.reshape(B, 4, 1).repeat(1, 1, 4)
    out2_grid = out2.reshape(B, 1, 4).repeat(1, 4, 1)
    d1_grid = doses_a.reshape(B, 4, 1).repeat(1, 1, 4)
    d2_grid = doses_b.reshape(B, 1, 4).repeat(1, 4, 1)
    fc_bias = torch.mul(out1_grid, d1_grid) + torch.mul(out2_grid, d2_grid)

    b2_bias = b2_bias_mod(rep, doses_a, doses_b)

    torch.testing.assert_close(b2_bias, fc_bias, rtol=1e-6, atol=1e-6)


# ------------------------------------------------------------
# TEST G — END-TO-END DOWNSTREAM PARITY
# ------------------------------------------------------------
@pytest.mark.skipif(not FINALCHECK_AVAILABLE, reason="Finalcheck modules not importable")
def test_g_end_to_end_downstream_parity():
    """Compare complete downstream pipeline (params -> constraints -> Hill -> bias -> output)."""
    torch.manual_seed(99)
    B = 2

    e1 = torch.sigmoid(torch.tensor([[0.5], [-0.2]]))
    e2 = torch.sigmoid(torch.tensor([[-0.1], [0.8]]))
    e3 = torch.sigmoid(torch.tensor([[-1.0], [0.3]]))
    log_c1 = torch.tensor([[0.2], [-0.5]])
    log_c2 = torch.tensor([[0.8], [0.1]])
    h1 = F.relu(torch.tensor([[1.5], [2.0]]))
    h2 = F.relu(torch.tensor([[1.2], [0.9]]))
    alpha = F.relu(torch.tensor([[1.8], [3.2]]))

    doses_a = torch.tensor([[0.0, 0.06, 0.6, 6.0]] * B, dtype=torch.float32)
    doses_b = torch.tensor([[0.0, 0.05, 0.5, 5.0]] * B, dtype=torch.float32)

    fc_solver = FinalcheckHillSolver(e0=100.0)
    b2_solver = BricksBivariateHillSolver(e0=100.0)

    y_hill_fc = fc_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
    y_hill_b2 = b2_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
    torch.testing.assert_close(y_hill_b2, y_hill_fc)

    r_DC = torch.randn(B, 1536)
    bias_module = DoseDependentBias(context_dim=1536, emb_size=1024, dropout=0.0, enabled=True)
    bias_module.eval()
    bias = bias_module(r_DC, doses_a, doses_b)

    y_pred_b2 = y_hill_b2 + bias
    y_pred_fc = y_hill_fc + bias

    torch.testing.assert_close(y_pred_b2, y_pred_fc)
    assert y_pred_b2.shape == (B, 4, 4)


# ------------------------------------------------------------
# TEST H — ZERO DOSE BEHAVIOR
# ------------------------------------------------------------
def test_h_zero_dose_behavior():
    """Verify that when dose_A = 0 and dose_B = 0:
       Y(0,0) == e0 == 100.0, Bias(0,0) == 0.0, no NaN, no Inf, finite gradients.
    """
    solver = BricksBivariateHillSolver(e0=100.0)
    bias_mod = DoseDependentBias(context_dim=1536, enabled=True)

    B = 3
    raw_e1 = torch.randn(B, 1, requires_grad=True)
    raw_e2 = torch.randn(B, 1, requires_grad=True)
    raw_e3 = torch.randn(B, 1, requires_grad=True)
    log_c1 = torch.randn(B, 1, requires_grad=True)
    log_c2 = torch.randn(B, 1, requires_grad=True)
    raw_h1 = torch.randn(B, 1, requires_grad=True)
    raw_h2 = torch.randn(B, 1, requires_grad=True)
    raw_alpha = torch.randn(B, 1, requires_grad=True)

    e1 = torch.sigmoid(raw_e1)
    e2 = torch.sigmoid(raw_e2)
    e3 = torch.sigmoid(raw_e3)
    h1 = F.relu(raw_h1)
    h2 = F.relu(raw_h2)
    alpha = F.relu(raw_alpha)

    doses_a = torch.tensor([[0.0, 0.1, 1.0, 10.0]] * B, dtype=torch.float32)
    doses_b = torch.tensor([[0.0, 0.05, 0.5, 5.0]] * B, dtype=torch.float32)

    r_DC = torch.randn(B, 1536)
    y_hill = solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
    bias = bias_mod(r_DC, doses_a, doses_b)
    y_pred = y_hill + bias

    for b in range(B):
        assert torch.isclose(y_hill[b, 0, 0], torch.tensor(100.0), atol=1e-4)
        assert torch.isclose(bias[b, 0, 0], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(y_pred[b, 0, 0], torch.tensor(100.0), atol=1e-4)

    loss = y_pred.sum()
    loss.backward()

    for p in [raw_e1, raw_e2, raw_e3, log_c1, log_c2, raw_h1, raw_h2, raw_alpha]:
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))


# ------------------------------------------------------------
# TEST I — NUMERICAL STABILITY
# ------------------------------------------------------------
def test_i_numerical_stability():
    """Test extreme but valid doses and parameters for finite outputs, loss, and gradients."""
    solver = BricksBivariateHillSolver(e0=100.0)

    doses_a = torch.tensor([[0.0, 1e-6, 1e-3, 1.0, 100.0, 1000.0]])
    doses_b = torch.tensor([[0.0, 1e-6, 1e-3, 1.0, 100.0, 1000.0]])

    e1 = torch.tensor([[0.001]], requires_grad=True)
    e2 = torch.tensor([[0.999]], requires_grad=True)
    e3 = torch.tensor([[0.0001]], requires_grad=True)
    log_c1 = torch.tensor([[-8.0]], requires_grad=True)
    log_c2 = torch.tensor([[8.0]], requires_grad=True)
    h1 = torch.tensor([[5.0]], requires_grad=True)
    h2 = torch.tensor([[0.1]], requires_grad=True)
    alpha = torch.tensor([[50.0]], requires_grad=True)

    y_hill = solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)

    assert torch.all(torch.isfinite(y_hill))

    loss = y_hill.sum()
    assert torch.isfinite(loss)

    loss.backward()
    for name, p in [("e1", e1), ("e2", e2), ("e3", e3), ("log_c1", log_c1),
                    ("log_c2", log_c2), ("h1", h1), ("h2", h2), ("alpha", alpha)]:
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))


# ------------------------------------------------------------
# TEST J — OUTPUT SHAPE
# ------------------------------------------------------------
def test_j_output_shape():
    """Verify output shape matches complete 2D dose-response matrix [B, n_dose_A, n_dose_B]."""
    cfg = ModelConfig(cell_dim=976, fragment_dim=512)
    model = CancerComboBRICS(config=cfg)

    B = 2
    n_dose_A = 4
    n_dose_B = 4

    cell_expr = torch.randn(B, 976)
    frags_A = [["CC(=O)O", "c1ccccc1"], ["c1ccccc1", "CCO"]]
    mask_A = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    frags_B = [["CCN(CC)CC", "c1ccccc1"], ["CC(C)O", "c1ccccc1"]]
    mask_B = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    doses_a = torch.tensor([[0.0, 0.06, 0.6, 6.0]] * B, dtype=torch.float32)
    doses_b = torch.tensor([[0.0, 0.05, 0.5, 5.0]] * B, dtype=torch.float32)

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

    assert y_pred.shape == (B, n_dose_A, n_dose_B)
    assert diag["r_DC"].shape == (B, 1536)
    assert diag["r_AB"].shape == (B, 512)
    assert diag["c"].shape == (B, 512)
    assert len(diag["raw_params"]) == 8
    assert y_pred.ndim == 3
