import torch
try:
    import pytest
except ImportError:
    pytest = None
from blocks.hill_equation import BivariateHillSolver

def test_hill_solver_shapes():
    """Verify solver output matrix grid coordinates shape."""
    solver = BivariateHillSolver(e0=100.0)
    batch_size = 4
    M, N = 6, 8
    
    doses_a = torch.randn(batch_size, M).abs()
    doses_b = torch.randn(batch_size, N).abs()
    
    e1 = torch.tensor([[80.0], [50.0], [90.0], [20.0]])
    e2 = torch.tensor([[70.0], [40.0], [80.0], [30.0]])
    e3 = torch.tensor([[10.0], [5.0], [15.0], [2.0]])
    c1 = torch.tensor([[1.0], [0.5], [2.0], [1.5]])
    c2 = torch.tensor([[2.0], [1.0], [3.0], [2.5]])
    h1 = torch.tensor([[1.2], [0.8], [1.5], [1.0]])
    h2 = torch.tensor([[1.5], [1.0], [2.0], [1.2]])
    alpha = torch.tensor([[1.0], [2.0], [0.5], [1.5]])
    
    out = solver(doses_a, doses_b, e1, e2, e3, c1, c2, h1, h2, alpha)
    assert out.shape == (batch_size, M, N)
    
def test_hill_solver_zero_dose_gradients():
    """Verify that zero concentration dose calculations do not yield NaN gradients."""
    solver = BivariateHillSolver(e0=100.0)
    
    doses_a = torch.zeros(2, 4, requires_grad=False)
    doses_b = torch.zeros(2, 5, requires_grad=False)
    
    e1 = torch.tensor([[80.0], [50.0]], requires_grad=True)
    e2 = torch.tensor([[70.0], [40.0]], requires_grad=True)
    e3 = torch.tensor([[10.0], [5.0]], requires_grad=True)
    c1 = torch.tensor([[1.0], [0.5]], requires_grad=True)
    c2 = torch.tensor([[2.0], [1.0]], requires_grad=True)
    h1 = torch.tensor([[1.2], [0.8]], requires_grad=True)
    h2 = torch.tensor([[1.5], [1.0]], requires_grad=True)
    alpha = torch.tensor([[1.0], [2.0]], requires_grad=True)
    
    out = solver(doses_a, doses_b, e1, e2, e3, c1, c2, h1, h2, alpha)
    
    assert out.shape == (2, 4, 5)
    assert torch.allclose(out, torch.tensor(100.0))
    
    loss = out.sum()
    loss.backward()
    
    for p in [e1, e2, e3, c1, c2, h1, h2, alpha]:
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()
        assert not torch.isinf(p.grad).any()


def test_hill_solver_extreme_doses():
    """Verify numerical stability (no NaN/Inf) with extreme dose concentrations and high exponents."""
    solver = BivariateHillSolver(e0=100.0)
    batch_size = 2
    
    # Grid with 0, 1e-12, and 1e8
    doses_a = torch.tensor([[0.0, 1e-12, 1.0, 1e8], [0.0, 1e-10, 1.0, 1e6]])
    doses_b = torch.tensor([[0.0, 1e-12, 1.0, 1e8], [0.0, 1e-10, 1.0, 1e6]])
    
    e1 = torch.tensor([[80.0], [50.0]], requires_grad=True)
    e2 = torch.tensor([[70.0], [40.0]], requires_grad=True)
    e3 = torch.tensor([[10.0], [5.0]], requires_grad=True)
    c1 = torch.tensor([[1e-4], [1e-2]], requires_grad=True)
    c2 = torch.tensor([[1e-4], [1e-2]], requires_grad=True)
    h1 = torch.tensor([[8.0], [10.0]], requires_grad=True)
    h2 = torch.tensor([[8.0], [10.0]], requires_grad=True)
    alpha = torch.tensor([[50.0], [100.0]], requires_grad=True)
    
    out = solver(doses_a, doses_b, e1, e2, e3, c1, c2, h1, h2, alpha)
    
    assert not torch.isnan(out).any(), "Output contains NaNs under extreme doses!"
    assert not torch.isinf(out).any(), "Output contains Infs under extreme doses!"
    
    loss = out.sum()
    loss.backward()
    
    for name, p in [("e1", e1), ("e2", e2), ("e3", e3), ("c1", c1), ("c2", c2), ("h1", h1), ("h2", h2), ("alpha", alpha)]:
        assert p.grad is not None, f"Gradient for {name} is None!"
        assert not torch.isnan(p.grad).any(), f"Gradient for {name} contains NaN!"
        assert not torch.isinf(p.grad).any(), f"Gradient for {name} contains Inf!"


if __name__ == "__main__":
    test_hill_solver_shapes()
    test_hill_solver_zero_dose_gradients()
    test_hill_solver_extreme_doses()
    print("ALL HILL SOLVER TESTS PASSED SUCCESSFULLY!")


