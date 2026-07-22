import torch
import torch.nn as nn

class BivariateHillSolver(nn.Module):
    """Numerically stable, fully differentiable bivariate Hill equation response surface solver."""
    
    def __init__(self, e0: float = 100.0):
        super().__init__()
        self.e0 = e0
        self.register_buffer("dummy_neg", torch.tensor(-100.0), persistent=False)
        
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

        # Broadcast all intermediate log-exponents and masks to the full grid shape (B, M, N)
        # explicitly before any mathematical operations. This avoids implicit autograd broadcasting
        # inside torch.maximum and torch.where, which is prone to CUDA synchronization deadlocks.
        B, M, N = doses_a_grid.size(0), doses_a_grid.size(1), doses_b_grid.size(2)
        
        exp_term_A_exp = exp_term_A.expand(B, M, N)
        exp_term_B_exp = exp_term_B.expand(B, M, N)
        exp_term_C_exp = exp_term_C.expand(B, M, N)
        exp_term_D_exp = exp_term_D.expand(B, M, N)
        
        mask_a_exp = mask_a.expand(B, M, N)
        mask_b_exp = mask_b.expand(B, M, N)
        mask_ab_exp = mask_ab.expand(B, M, N)

        # Compute maximum exponent across active terms per batch element for numerical stability
        cand_A = exp_term_A_exp
        cand_B = torch.where(mask_a_exp, exp_term_B_exp, torch.full_like(exp_term_B_exp, -1e9))
        cand_C = torch.where(mask_b_exp, exp_term_C_exp, torch.full_like(exp_term_C_exp, -1e9))
        cand_D = torch.where(mask_ab_exp, exp_term_D_exp, torch.full_like(exp_term_D_exp, -1e9))

        max_exp = torch.maximum(
            torch.maximum(cand_A, cand_B),
            torch.maximum(cand_C, cand_D)
        ).detach()

        dummy_neg = self.dummy_neg.to(device=doses_a_grid.device, dtype=doses_a_grid.dtype)

        safe_A = torch.clamp(exp_term_A_exp - max_exp, max=50.0)
        safe_B = torch.clamp(torch.where(mask_a_exp, exp_term_B_exp - max_exp, dummy_neg), max=50.0)
        safe_C = torch.clamp(torch.where(mask_b_exp, exp_term_C_exp - max_exp, dummy_neg), max=50.0)
        safe_D = torch.clamp(torch.where(mask_ab_exp, exp_term_D_exp - max_exp, dummy_neg), max=50.0)

        exp_A = torch.exp(safe_A)
        exp_B = torch.where(mask_a_exp, torch.exp(safe_B), torch.zeros_like(exp_term_B_exp))
        exp_C = torch.where(mask_b_exp, torch.exp(safe_C), torch.zeros_like(exp_term_C_exp))
        exp_D = torch.where(mask_ab_exp, torch.exp(safe_D), torch.zeros_like(exp_term_D_exp))

        numerator = self.e0 * exp_A + e1_u * exp_B + e2_u * exp_C + e3_u * alpha_u * exp_D
        denominator = exp_A + exp_B + exp_C + alpha_u * exp_D

        y_pred = numerator / denominator
        return y_pred
