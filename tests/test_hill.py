"""Unit tests for finalcheck-aligned pharmacological parameter heads, constraints, and BivariateHillSolver."""

import torch
from cancer_combo_brics.pharmacology.parameter_heads import PharmacologicalParameterHeads
from cancer_combo_brics.pharmacology.constraints import ConstraintTransform
from cancer_combo_brics.pharmacology.bivariate_hill import BivariateHillSolver
from cancer_combo_brics.pharmacology.dose_bias import DoseDependentBias


def test_parameter_heads_and_constraints():
    heads = PharmacologicalParameterHeads(in_dim=1536)
    constraints = ConstraintTransform(e0=100.0)

    B = 3
    r_DC = torch.randn(B, 1536)
    raw = heads(r_DC)

    # Exactly 8 raw parameters
    assert len(raw) == 8
    for name, tensor in raw.items():
        assert tensor.shape == (B,)

    # Constrain
    constrained, diag = constraints(raw)
    assert constrained["e0"][0].item() == 100.0  # Fixed baseline: 100.0 for percentage viability
    assert torch.all(constrained["e1"] >= 0.0)
    assert torch.all(constrained["e1"] <= 1.0)
    assert torch.all(constrained["h1"] >= 0.0)
    assert torch.all(constrained["alpha"] >= 0.0)


def test_bivariate_hill_solver_zero_doses_and_gradients():
    solver = BivariateHillSolver(e0=100.0)

    B = 2
    params = {
        "e0": torch.tensor([100.0, 100.0], requires_grad=False),
        "e1": torch.tensor([0.2, 0.3], requires_grad=True),
        "e2": torch.tensor([0.4, 0.1], requires_grad=True),
        "e3": torch.tensor([0.05, 0.08], requires_grad=True),
        "logc1": torch.tensor([0.0, -1.0], requires_grad=True),
        "logc2": torch.tensor([1.0, 0.5], requires_grad=True),
        "h1": torch.tensor([1.5, 2.0], requires_grad=True),
        "h2": torch.tensor([1.2, 1.0], requires_grad=True),
        "alpha": torch.tensor([1.0, 2.5], requires_grad=True),
    }

    # Dose grids with 0.0 at first position (in uM)
    doses_A = torch.tensor([[0.0, 0.1, 1.0, 10.0], [0.0, 0.1, 1.0, 10.0]])
    doses_B = torch.tensor([[0.0, 0.05, 0.5, 5.0], [0.0, 0.05, 0.5, 5.0]])

    surface = solver(doses_A, doses_B, params["e1"], params["e2"], params["e3"],
                     params["logc1"], params["logc2"], params["h1"], params["h2"], params["alpha"])

    # 1. Output shape (B, D_A, D_B) = (2, 4, 4)
    assert surface.shape == (B, 4, 4)

    # 2. At zero dose for both drugs, viability MUST equal e0 = 100.0
    assert torch.allclose(surface[:, 0, 0], torch.tensor([100.0, 100.0]), atol=1e-4)

    # 3. Check gradients
    loss = surface.sum()
    loss.backward()

    for name in ["e1", "e2", "e3", "logc1", "logc2", "h1", "h2", "alpha"]:
        assert params[name].grad is not None
        assert not torch.isnan(params[name].grad).any()
        assert not torch.isinf(params[name].grad).any()


def test_dose_dependent_bias():
    bias_module = DoseDependentBias(context_dim=1536, enabled=True)
    B = 2
    r_DC = torch.randn(B, 1536)
    doses_A = torch.tensor([[0.0, 0.1, 1.0, 10.0], [0.0, 0.1, 1.0, 10.0]])
    doses_B = torch.tensor([[0.0, 0.05, 0.5, 5.0], [0.0, 0.05, 0.5, 5.0]])

    bias = bias_module(r_DC, doses_A, doses_B)
    assert bias.shape == (B, 4, 4)

    # Bias at (0, 0) MUST be zero
    assert torch.all(bias[:, 0, 0] == 0.0)
    assert not torch.isnan(bias).any()
    assert not torch.isinf(bias).any()
