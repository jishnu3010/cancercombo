"""Vectorized, numerically stable 2D Bivariate Hill Solver matching finalcheck."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


class BivariateHillSolver(nn.Module):
    """Numerically stable 2D Bivariate Hill Solver using SynBa Log-Sum-Exp formulation.

    Matches finalcheck blocks/hill_equation.py exactly:
      - Baseline e0 = 100.0 (percentage viability)
      - Numerical safety: shifts zero concentrations to 1e-6 before torch.log()
      - SynBa Log-Sum-Exp trick across 4 terms (term_A, term_B, term_C, term_D)
      - Exact zero-dose handling via presence masks (mask_a, mask_b, mask_ab)
      - Y(0, 0) == e0 == 100.0
    """

    def __init__(self, e0: float = 100.0, eps: float = 1e-12, **kwargs):
        super().__init__()
        self.e0 = float(e0)
        self.eps = eps

    def synba_likelihood_2d(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        e_1: torch.Tensor,
        e_2: torch.Tensor,
        e_3: torch.Tensor,
        logC_1: torch.Tensor,
        logC_2: torch.Tensor,
        h_1: torch.Tensor,
        h_2: torch.Tensor,
        alpha: torch.Tensor,
        sigma: Optional[Union[float, torch.Tensor]] = None,
        add_noise: bool = False,
    ) -> torch.Tensor:
        """SynBa 2D dose-response likelihood evaluation with Log-Sum-Exp numerical stability.

        Args:
            x_1: Dose grid concentrations for Drug A of shape (B, M) or (B, M, 1) or (M,).
            x_2: Dose grid concentrations for Drug B of shape (B, N) or (B, 1, N) or (N,).
            e_1: Drug A fractional efficacy parameter of shape (B, 1) or (B,) or scalar.
            e_2: Drug B fractional efficacy parameter of shape (B, 1) or (B,) or scalar.
            e_3: Combination fractional efficacy parameter of shape (B, 1) or (B,) or scalar.
            logC_1: Log IC50 for Drug A of shape (B, 1) or (B,) or scalar.
            logC_2: Log IC50 for Drug B of shape (B, 1) or (B,) or scalar.
            h_1: Hill slope coefficient for Drug A of shape (B, 1) or (B,) or scalar.
            h_2: Hill slope coefficient for Drug B of shape (B, 1) or (B,) or scalar.
            alpha: Interaction parameter of shape (B, 1) or (B,) or scalar.
            sigma: Optional noise magnitude parameter.
            add_noise: Whether to add Gaussian noise to the predicted viability matrix.

        Returns:
            torch.Tensor: Predicted viability grid of shape (B, M, N).
        """
        e_0 = self.e0

        # Reshape parameters to 3D tensors (B, 1, 1) for matrix broadcasting
        def _to_3d(p):
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(p, dtype=torch.float32, device=x_1.device if isinstance(x_1, torch.Tensor) else None)
            while p.dim() < 3:
                p = p.unsqueeze(-1)
            return p

        e_1_u = _to_3d(e_1)
        e_2_u = _to_3d(e_2)
        e_3_u = _to_3d(e_3)

        # Scale efficacy parameters to fractional [0, 1] range if passed as percentage [0, 100]
        if (e_1_u > 1.0).any():
            e_1_u = e_1_u / 100.0
        if (e_2_u > 1.0).any():
            e_2_u = e_2_u / 100.0
        if (e_3_u > 1.0).any():
            e_3_u = e_3_u / 100.0

        logC_1_u = _to_3d(logC_1)
        logC_2_u = _to_3d(logC_2)
        h_1_u = _to_3d(h_1)
        h_2_u = _to_3d(h_2)
        alpha_u = _to_3d(alpha)

        B = e_1_u.shape[0]

        # Reshape dose concentration inputs
        if x_1.dim() == 1:
            x_1 = x_1.unsqueeze(0).expand(B, -1).unsqueeze(2)  # (B, M, 1)
        elif x_1.dim() == 2:
            x_1 = x_1.unsqueeze(2)  # (B, M, 1)

        if x_2.dim() == 1:
            x_2 = x_2.unsqueeze(0).expand(B, -1).unsqueeze(1)  # (B, 1, N)
        elif x_2.dim() == 2:
            x_2 = x_2.unsqueeze(1)  # (B, 1, N)

        # Zero-dose presence masks
        mask_a = (x_1 > 0.0)
        mask_b = (x_2 > 0.0)
        mask_ab = mask_a & mask_b

        # Numerical safety: shift zero concentrations to 1e-6 before log() to prevent log(0) NaNs
        x_1_safe = torch.where(mask_a, x_1, torch.ones_like(x_1) * 1e-6)
        x_2_safe = torch.where(mask_b, x_2, torch.ones_like(x_2) * 1e-6)

        log_x1 = torch.log(x_1_safe)
        log_x2 = torch.log(x_2_safe)

        # 4 Log-exponent terms
        term_A = logC_1_u * h_1_u + logC_2_u * h_2_u
        term_B = log_x1 * h_1_u + logC_2_u * h_2_u
        term_C = logC_1_u * h_1_u + log_x2 * h_2_u
        term_D = log_x1 * h_1_u + log_x2 * h_2_u

        # Log-sum-exp trick for numerical stability
        max_exp = torch.maximum(
            term_A,
            torch.maximum(
                term_B,
                torch.maximum(term_C, term_D)
            )
        )

        w_A = torch.exp(term_A - max_exp)
        w_B = torch.where(mask_a, torch.exp(term_B - max_exp), torch.zeros_like(term_B))
        w_C = torch.where(mask_b, torch.exp(term_C - max_exp), torch.zeros_like(term_C))
        w_D = torch.where(mask_ab, torch.exp(term_D - max_exp), torch.zeros_like(term_D))

        exp_A = w_A * e_0
        exp_B = w_B * e_1_u * e_0
        exp_C = w_C * e_2_u * e_0
        exp_D = w_D * e_3_u * e_0 * alpha_u

        numerator = exp_A + exp_B + exp_C + exp_D
        denominator = w_A + w_B + w_C + w_D * alpha_u

        if add_noise and sigma is not None:
            if isinstance(sigma, torch.Tensor):
                while sigma.dim() < 3:
                    sigma = sigma.unsqueeze(-1)
                z = torch.randn_like(numerator) * sigma
            else:
                z = torch.randn_like(numerator) * float(sigma)
        else:
            z = 0.0

        y = numerator / denominator + z
        return y

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Unified forward pass supporting both finalcheck positional arguments and legacy dict arguments.

        Finalcheck signature:
            forward(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha, ...)

        Legacy signature:
            forward(params_dict, doses_A, doses_B)
        """
        if len(args) == 3 and isinstance(args[0], dict):
            params, doses_a, doses_b = args
            return self.synba_likelihood_2d(
                x_1=doses_a,
                x_2=doses_b,
                e_1=params["e1"],
                e_2=params["e2"],
                e_3=params["e3"],
                logC_1=params.get("logc1", params.get("log_c1")),
                logC_2=params.get("logc2", params.get("log_c2")),
                h_1=params["h1"],
                h_2=params["h2"],
                alpha=params["alpha"],
            )

        if len(args) >= 10:
            doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha = args[:10]
            sigma = args[10] if len(args) > 10 else kwargs.get("sigma", None)
            add_noise = args[11] if len(args) > 11 else kwargs.get("add_noise", False)
            return self.synba_likelihood_2d(
                x_1=doses_a,
                x_2=doses_b,
                e_1=e1,
                e_2=e2,
                e_3=e3,
                logC_1=log_c1,
                logC_2=log_c2,
                h_1=h1,
                h_2=h2,
                alpha=alpha,
                sigma=sigma,
                add_noise=add_noise,
            )

        # Keyword argument dispatch
        doses_a = kwargs.get("doses_a", kwargs.get("doses_A", kwargs.get("x_1")))
        doses_b = kwargs.get("doses_b", kwargs.get("doses_B", kwargs.get("x_2")))
        e1 = kwargs.get("e1", kwargs.get("e_1"))
        e2 = kwargs.get("e2", kwargs.get("e_2"))
        e3 = kwargs.get("e3", kwargs.get("e_3"))
        log_c1 = kwargs.get("log_c1", kwargs.get("logc1", kwargs.get("logC_1")))
        log_c2 = kwargs.get("log_c2", kwargs.get("logc2", kwargs.get("logC_2")))
        h1 = kwargs.get("h1", kwargs.get("h_1"))
        h2 = kwargs.get("h2", kwargs.get("h_2"))
        alpha = kwargs.get("alpha")
        sigma = kwargs.get("sigma", None)
        add_noise = kwargs.get("add_noise", False)

        return self.synba_likelihood_2d(
            x_1=doses_a,
            x_2=doses_b,
            e_1=e1,
            e_2=e2,
            e_3=e3,
            logC_1=log_c1,
            logC_2=log_c2,
            h_1=h1,
            h_2=h2,
            alpha=alpha,
            sigma=sigma,
            add_noise=add_noise,
        )
