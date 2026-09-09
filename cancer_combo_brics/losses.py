"""Loss functions for 2D dose-response surface regression and ranking matching finalcheck."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceRegressionLoss(nn.Module):
    """Loss function operating on viability surfaces.

    Supports:
      - 'mse': Mean Squared Error
      - 'huber': Smooth L1 / Huber Loss with configurable delta (default 1.0 for percentage scale)
      - 'mae': Mean Absolute Error
    """

    def __init__(self, loss_type: str = "huber", delta: float = 1.0):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.delta = delta

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute surface loss.

        Args:
            y_pred: Predicted viability surface of shape (B, D_A, D_B).
            y_true: Ground-truth viability surface of shape (B, D_A, D_B).
            mask: Optional boolean or float mask (B, D_A, D_B) indicating valid points.

        Returns:
            Scalar loss tensor.
        """
        assert y_pred.shape == y_true.shape, (
            f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
        )

        if self.loss_type == "mse":
            diff_sq = (y_pred - y_true) ** 2
            if mask is not None:
                mask_f = mask.float()
                loss = (diff_sq * mask_f).sum() / torch.clamp(mask_f.sum(), min=1.0)
            else:
                loss = diff_sq.mean()

        elif self.loss_type == "huber":
            loss_elementwise = F.huber_loss(y_pred, y_true, delta=self.delta, reduction="none")
            if mask is not None:
                mask_f = mask.float()
                loss = (loss_elementwise * mask_f).sum() / torch.clamp(mask_f.sum(), min=1.0)
            else:
                loss = loss_elementwise.mean()

        elif self.loss_type == "mae":
            diff_abs = torch.abs(y_pred - y_true)
            if mask is not None:
                mask_f = mask.float()
                loss = (diff_abs * mask_f).sum() / torch.clamp(mask_f.sum(), min=1.0)
            else:
                loss = diff_abs.mean()

        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return loss


class CancerComboLoss(nn.Module):
    """Composite loss module matching finalcheck.

    Combines Mean Squared Error over dose viability matrices,
    pairwise Margin Ranking Loss (for Spearman rank correlation optimization),
    and optional Auxiliary Parameter Supervision loss.
    """

    def __init__(self, rank_lambda: float = 1.0, aux_lambda: float = 0.05, num_ranking_pairs: int = 256):
        super().__init__()
        self.mse = nn.MSELoss()
        self.margin_ranking = nn.MarginRankingLoss(margin=0.0)
        self.smooth_l1 = nn.SmoothL1Loss()
        self.rank_lambda = rank_lambda
        self.aux_lambda = aux_lambda
        self.num_ranking_pairs = num_ranking_pairs

    def _compute_ranking_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes pairwise margin ranking loss across flattened dose grid points."""
        B = y_pred.size(0)
        pred_flat = y_pred.view(B, -1)
        true_flat = y_true.view(B, -1)
        N = pred_flat.size(1)

        if N < 2:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        # Sample pairs of indices
        idx1 = torch.randint(0, N, (self.num_ranking_pairs,), device=y_pred.device)
        idx2 = torch.randint(0, N, (self.num_ranking_pairs,), device=y_pred.device)

        mask = idx1 != idx2
        idx1 = idx1[mask]
        idx2 = idx2[mask]

        if len(idx1) == 0:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        pred1 = pred_flat[:, idx1]  # (B, P)
        pred2 = pred_flat[:, idx2]  # (B, P)
        true1 = true_flat[:, idx1]  # (B, P)
        true2 = true_flat[:, idx2]  # (B, P)

        target_sign = torch.sign(true1 - true2)  # +1 if true1 > true2, -1 if true1 < true2
        valid_mask = (target_sign != 0).float()

        if not valid_mask.any():
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

        loss_pairs = torch.clamp(-target_sign * (pred1 - pred2), min=0.0)
        num_valid = valid_mask.sum().clamp(min=1.0)
        return (loss_pairs * valid_mask).sum() / num_valid

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        params_pred: Optional[Tuple[torch.Tensor, ...]] = None,
        params_true: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Computes composite loss: Surface MSE + lambda_rank * Ranking Loss + lambda_aux * Param Loss."""
        loss_mse = self.mse(y_pred, y_true)
        loss_rank = self._compute_ranking_loss(y_pred, y_true)
        total_loss = loss_mse + self.rank_lambda * loss_rank

        if params_pred is not None and params_true is not None:
            param_names = ["e1", "e2", "e3", "log_c1", "log_c2", "h1", "h2", "alpha"]
            loss_aux = 0.0
            count = 0
            for pred, name in zip(params_pred, param_names):
                if name in params_true:
                    target = params_true[name]
                    if target.dim() == 1:
                        target = target.unsqueeze(-1)
                    loss_aux += self.smooth_l1(pred, target)
                    count += 1
            if count > 0:
                total_loss += self.aux_lambda * (loss_aux / count)

        return total_loss
