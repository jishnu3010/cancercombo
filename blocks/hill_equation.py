import torch
import torch.nn as nn

class BivariateHillSolver(nn.Module):
    """Numerically stable, fully differentiable bivariate Hill equation response surface solver."""
    
    def __init__(self, e0: float = 100.0):
        super().__init__()
        self.e0 = e0
        
    def forward(
        self,
        doses_a: torch.Tensor,
        doses_b: torch.Tensor,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        h1: torch.Tensor,
        h2: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """Solves the bivariate Hill equation across the input dose grids.

        Args:
            doses_a: Dose range for Drug A (B, M) or (B, M, N).
            doses_b: Dose range for Drug B (B, N) or (B, M, N).
            e1, e2, e3, c1, c2, h1, h2, alpha: Parameter tensors (B, 1).

        Returns:
            torch.Tensor: Predicted cell viability matrix of shape (B, M, N).
        """
        e1_u = e1.unsqueeze(-1)
        e2_u = e2.unsqueeze(-1)
        e3_u = e3.unsqueeze(-1)
        c1_u = c1.unsqueeze(-1)
        c2_u = c2.unsqueeze(-1)
        h1_u = h1.unsqueeze(-1)
        h2_u = h2.unsqueeze(-1)
        alpha_u = alpha.unsqueeze(-1)
        
        if doses_a.dim() == 2:
            doses_a_grid = doses_a.unsqueeze(2) # (B, M, 1)
        else:
            doses_a_grid = doses_a
            
        if doses_b.dim() == 2:
            doses_b_grid = doses_b.unsqueeze(1) # (B, 1, N)
        else:
            doses_b_grid = doses_b
            
        # Add epsilon to zero values to prevent gradient NaNs at zero dose points
        doses_a_safe = torch.where(doses_a_grid > 0.0, doses_a_grid, torch.ones_like(doses_a_grid) * 1e-15)
        doses_b_safe = torch.where(doses_b_grid > 0.0, doses_b_grid, torch.ones_like(doses_b_grid) * 1e-15)
        
        term_a = (doses_a_safe / c1_u) ** h1_u
        term_b = (doses_b_safe / c2_u) ** h2_u
        
        # Zero out where actual dose was 0
        term_a = torch.where(doses_a_grid > 0.0, term_a, torch.zeros_like(term_a))
        term_b = torch.where(doses_b_grid > 0.0, term_b, torch.zeros_like(term_b))
        
        term_ab = term_a * term_b
        
        numerator = self.e0 + e1_u * term_a + e2_u * term_b + e3_u * alpha_u * term_ab
        denominator = 1.0 + term_a + term_b + alpha_u * term_ab
        
        y_pred = numerator / denominator
        return y_pred
