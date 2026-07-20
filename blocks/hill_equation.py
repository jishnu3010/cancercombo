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
        log_c1: torch.Tensor,
        log_c2: torch.Tensor,
        h1: torch.Tensor,
        h2: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """Solves the bivariate Hill equation across the input dose grids.

        Args:
            doses_a: Dose range for Drug A (B, M) or (B, M, N).
            doses_b: Dose range for Drug B (B, N) or (B, M, N).
            e1, e2, e3: Efficacy parameter tensors (B, 1).
            log_c1, log_c2: Log-concentration parameter tensors (B, 1).
            h1, h2, alpha: Slope and interaction parameter tensors (B, 1).

        Returns:
            torch.Tensor: Predicted cell viability matrix of shape (B, M, N).
        """
        e1_u = e1.unsqueeze(-1)
        e2_u = e2.unsqueeze(-1)
        e3_u = e3.unsqueeze(-1)
        log_c1_u = log_c1.unsqueeze(-1)
        log_c2_u = log_c2.unsqueeze(-1)
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
            
        # Identify active dose masks
        mask_a = doses_a_grid > 0.0
        mask_b = doses_b_grid > 0.0
        mask_ab = mask_a & mask_b

        # Protect dose values before log() to prevent log(0), -inf, or NaN
        doses_a_safe = torch.where(mask_a, doses_a_grid, torch.ones_like(doses_a_grid))
        doses_b_safe = torch.where(mask_b, doses_b_grid, torch.ones_like(doses_b_grid))

        log_x1 = torch.log(doses_a_safe)
        log_x2 = torch.log(doses_b_safe)

        # Log-space concentration parameterization avoids taking log() of C in forward pass
        log_c1 = torch.clamp(log_c1_u, min=-30.0, max=30.0)
        log_c2 = torch.clamp(log_c2_u, min=-30.0, max=30.0)

        # Log-space exponent calculations prior to exponentiation
        exp_term_A = log_c1 * h1_u + log_c2 * h2_u
        exp_term_B = log_x1 * h1_u + log_c2 * h2_u
        exp_term_C = log_c1 * h1_u + log_x2 * h2_u
        exp_term_D = log_x1 * h1_u + log_x2 * h2_u

        # Compute maximum exponent across active terms per batch element for numerical stability
        cand_A = exp_term_A
        cand_B = torch.where(mask_a, exp_term_B, torch.full_like(exp_term_B, -1e9))
        cand_C = torch.where(mask_b, exp_term_C, torch.full_like(exp_term_C, -1e9))
        cand_D = torch.where(mask_ab, exp_term_D, torch.full_like(exp_term_D, -1e9))

        max_exp = torch.maximum(
            torch.maximum(cand_A, cand_B),
            torch.maximum(cand_C, cand_D)
        ).detach()

        dummy_neg = torch.tensor(-100.0, device=doses_a_grid.device, dtype=doses_a_grid.dtype)

        safe_A = torch.clamp(exp_term_A - max_exp, max=50.0)
        safe_B = torch.clamp(torch.where(mask_a, exp_term_B - max_exp, dummy_neg), max=50.0)
        safe_C = torch.clamp(torch.where(mask_b, exp_term_C - max_exp, dummy_neg), max=50.0)
        safe_D = torch.clamp(torch.where(mask_ab, exp_term_D - max_exp, dummy_neg), max=50.0)

        exp_A = torch.exp(safe_A)
        exp_B = torch.where(mask_a, torch.exp(safe_B), torch.zeros_like(exp_term_B))
        exp_C = torch.where(mask_b, torch.exp(safe_C), torch.zeros_like(exp_term_C))
        exp_D = torch.where(mask_ab, torch.exp(safe_D), torch.zeros_like(exp_term_D))

        numerator = self.e0 * exp_A + e1_u * exp_B + e2_u * exp_C + e3_u * alpha_u * exp_D
        denominator = exp_A + exp_B + exp_C + alpha_u * exp_D

        y_pred = numerator / denominator
        return y_pred
