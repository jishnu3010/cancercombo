import torch
import torch.nn as nn

class BivariateHillSolver(nn.Module):
    """Numerically stable Bivariate Hill Solver WITH Proper Masking."""
    
    def __init__(self, e0: float = 100.0):
        super().__init__()
        self.e0 = e0
        
    def forward(self, doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha):
        e1_u, e2_u, e3_u = e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
        log_c1_u, log_c2_u = log_c1.unsqueeze(-1), log_c2.unsqueeze(-1)
        h1_u, h2_u, alpha_u = h1.unsqueeze(-1), h2.unsqueeze(-1), alpha.unsqueeze(-1)
        
        if doses_a.dim() == 2: doses_a = doses_a.unsqueeze(2)
        if doses_b.dim() == 2: doses_b = doses_b.unsqueeze(1)
            
        # PROPER MASKING RESTORED
        mask_a = doses_a > 1e-9
        mask_b = doses_b > 1e-9
        mask_ab = mask_a & mask_b
        
        doses_a_safe = torch.where(mask_a, doses_a, torch.ones_like(doses_a))
        doses_b_safe = torch.where(mask_b, doses_b, torch.ones_like(doses_b))
        
        log_x1 = torch.log(doses_a_safe)
        log_x2 = torch.log(doses_b_safe)

        log_c1 = torch.clamp(log_c1_u, min=-15.0, max=15.0)
        log_c2 = torch.clamp(log_c2_u, min=-15.0, max=15.0)

        exp_A = log_c1 * h1_u + log_c2 * h2_u
        exp_B = log_x1 * h1_u + log_c2 * h2_u
        exp_C = log_c1 * h1_u + log_x2 * h2_u
        exp_D = log_x1 * h1_u + log_x2 * h2_u

        B, M, N = doses_a.size(0), doses_a.size(1), doses_b.size(2)
        exp_A, exp_B = exp_A.expand(B, M, N), exp_B.expand(B, M, N)
        exp_C, exp_D = exp_C.expand(B, M, N), exp_D.expand(B, M, N)
        mask_a_exp = mask_a.expand(B, M, N)
        mask_b_exp = mask_b.expand(B, M, N)
        mask_ab_exp = mask_ab.expand(B, M, N)

        # Prevent exp() overflow/underflow while maintaining gradients
        def safe_exp(x):
            x = torch.clamp(x, min=-50.0, max=50.0)
            return torch.exp(x)
        
        eps = 1e-8
        val_A = safe_exp(exp_A) + eps
        val_B = torch.where(mask_a_exp, safe_exp(exp_B) + eps, torch.zeros_like(exp_B))
        val_C = torch.where(mask_b_exp, safe_exp(exp_C) + eps, torch.zeros_like(exp_C))
        val_D = torch.where(mask_ab_exp, safe_exp(exp_D) + eps, torch.zeros_like(exp_D))

        numerator = self.e0 * val_A + e1_u * val_B + e2_u * val_C + e3_u * alpha_u * val_D
        denominator = val_A + val_B + val_C + alpha_u * val_D

        return numerator / denominator