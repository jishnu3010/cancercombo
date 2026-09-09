"""Dose-dependent bias module matching finalcheck DeepSynBa architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
from cancer_combo_brics.pharmacology.parameter_heads import DeepSynBaBlock


class DoseResponsePredictor(nn.Module):
    """DeepSynBa 4-layer MLP Dose-Response Predictor for bias vectors matching finalcheck."""

    def __init__(self, in_channels: int = 1536, emb_size: int = 1024, dropout: float = 0.2):
        super().__init__()
        layers = [
            DeepSynBaBlock(in_channels, emb_size, dropout=dropout),
            DeepSynBaBlock(emb_size, emb_size / 2, dropout=dropout),
            DeepSynBaBlock(emb_size / 2, emb_size / 4, dropout=dropout),
            nn.Linear(int(emb_size / 4), 4),
            nn.ReLU(),
        ]
        self.prediction_head = nn.Sequential(*layers)

        for name, param in self.prediction_head.named_parameters():
            if "weight" in name and len(param.data.shape) > 1:
                nn.init.kaiming_normal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(input)


class DoseDependentBias(nn.Module):
    """Dose-dependent additive bias matrix matching finalcheck DeepSynBa implementation.

    Formula:
      b_A = bias_predictor1(r_DC)  # (B, 4) >= 0 via ReLU
      b_B = bias_predictor2(r_DC)  # (B, 4) >= 0 via ReLU

      Bias[i, j] = b_A[i] * dose_A[i] + b_B[j] * dose_B[j]
    """

    def __init__(
        self,
        context_dim: int = 1536,
        emb_size: int = 1024,
        dropout: float = 0.2,
        enabled: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.emb_size = emb_size
        self.dropout = dropout
        self.enabled = enabled

        # 2 DeepSynBa-style Bias Predictors matching finalcheck
        self.bias_predictor1 = DoseResponsePredictor(context_dim, emb_size, dropout)
        self.bias_predictor2 = DoseResponsePredictor(context_dim, emb_size, dropout)

    def predict_bias(
        self,
        r_DC: torch.Tensor,
        doses_a: torch.Tensor,
        doses_b: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts and broadcasts 2D dose-dependent bias matrix matching official DeepSynBa."""
        b_size = r_DC.shape[0]

        # Standardize 1D doses to 2D
        if doses_a.ndim == 1:
            doses_a = doses_a.unsqueeze(0).expand(b_size, -1)
        if doses_b.ndim == 1:
            doses_b = doses_b.unsqueeze(0).expand(b_size, -1)

        if doses_a.ndim != 2:
            raise ValueError(f"Expected doses_a tensor shape [batch, num_doses] (ndim=2), got shape {tuple(doses_a.shape)}")
        if doses_b.ndim != 2:
            raise ValueError(f"Expected doses_b tensor shape [batch, num_doses] (ndim=2), got shape {tuple(doses_b.shape)}")

        M = doses_a.shape[1]
        N = doses_b.shape[1]

        if not self.enabled:
            return torch.zeros((b_size, M, N), dtype=torch.float32, device=r_DC.device)

        out1 = self.bias_predictor1(r_DC)  # (B, 4)
        out2 = self.bias_predictor2(r_DC)  # (B, 4)

        if M == 4 and N == 4:
            out1_grid = out1.reshape(b_size, 4, 1).repeat(1, 1, 4)
            out2_grid = out2.reshape(b_size, 1, 4).repeat(1, 4, 1)
        else:
            out1_grid = (
                out1[:, :M].unsqueeze(2).repeat(1, 1, N)
                if out1.shape[1] >= M
                else torch.nn.functional.interpolate(
                    out1.unsqueeze(1), size=M, mode="linear", align_corners=False
                ).squeeze(1).unsqueeze(2).repeat(1, 1, N)
            )
            out2_grid = (
                out2[:, :N].unsqueeze(1).repeat(1, M, 1)
                if out2.shape[1] >= N
                else torch.nn.functional.interpolate(
                    out2.unsqueeze(1), size=N, mode="linear", align_corners=False
                ).squeeze(1).unsqueeze(1).repeat(1, M, 1)
            )

        d1_grid = doses_a.reshape(b_size, M, 1).repeat(1, 1, N)
        d2_grid = doses_b.reshape(b_size, 1, N).repeat(1, M, 1)

        bias = torch.mul(out1_grid, d1_grid) + torch.mul(out2_grid, d2_grid)
        return bias

    def forward(
        self,
        r_DC: torch.Tensor,
        doses_A: torch.Tensor,
        doses_B: torch.Tensor,
    ) -> torch.Tensor:
        """Alias forward matching predict_bias."""
        return self.predict_bias(r_DC, doses_A, doses_B)
