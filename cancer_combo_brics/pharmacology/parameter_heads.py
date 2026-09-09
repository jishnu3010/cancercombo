"""Pharmacological parameter heads predicting 8 raw parameters from r_DC matching finalcheck."""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn


class DeepSynBaBlock(nn.Module):
    """DeepSynBa MLP Block with Linear, LayerNorm, ReLU, and Dropout matching finalcheck."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(int(in_channels), int(out_channels)),
            nn.LayerNorm((int(out_channels),), eps=1e-05, elementwise_affine=True),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepSynBaPredictionHead(nn.Module):
    """DeepSynBa 3-layer MLP Prediction Head for biophysical parameters matching finalcheck."""

    def __init__(self, in_channels: int = 1536, emb_size: int = 1024, dropout: float = 0.2):
        super().__init__()
        layers = [
            DeepSynBaBlock(in_channels, emb_size, dropout=dropout),
            DeepSynBaBlock(emb_size, emb_size / 2, dropout=dropout),
            DeepSynBaBlock(emb_size / 2, emb_size / 4, dropout=dropout),
            nn.Linear(int(emb_size / 4), 1),
        ]
        self.prediction_head = nn.Sequential(*layers)

        for name, param in self.prediction_head.named_parameters():
            if "weight" in name and len(param.data.shape) > 1:
                nn.init.kaiming_normal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(input)


class PharmacologicalParameterHeads(nn.Module):
    """DeepSynBa-style parameter prediction heads predicting 8 biophysical parameters from r_DC.

    Parameters predicted in exact finalcheck order:
      1. e1_raw: Drug A fractional efficacy
      2. e2_raw: Drug B fractional efficacy
      3. e3_raw: Combined fractional efficacy
      4. logc1_raw: Drug A log IC50
      5. logc2_raw: Drug B log IC50
      6. h1_raw: Drug A Hill slope coefficient
      7. h2_raw: Drug B Hill slope coefficient
      8. alpha_raw: Drug-drug interaction parameter
    """

    PARAM_NAMES = [
        "e1_raw",
        "e2_raw",
        "e3_raw",
        "logc1_raw",
        "logc2_raw",
        "h1_raw",
        "h2_raw",
        "alpha_raw",
    ]

    def __init__(
        self,
        in_dim: int = 1536,
        emb_size: int = 1024,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.emb_size = emb_size
        self.dropout = dropout

        # 8 individual DeepSynBa Prediction Heads matching finalcheck
        self.head_e1 = DeepSynBaPredictionHead(in_dim, emb_size, dropout)
        self.head_e2 = DeepSynBaPredictionHead(in_dim, emb_size, dropout)
        self.head_e3 = DeepSynBaPredictionHead(in_dim, emb_size, dropout)
        self.head_log_c1 = DeepSynBaPredictionHead(in_dim, emb_size, dropout)
        self.head_log_c2 = DeepSynBaPredictionHead(in_dim, emb_size, dropout)
        self.head_h1 = DeepSynBaPredictionHead(in_dim, emb_size, dropout)
        self.head_h2 = DeepSynBaPredictionHead(in_dim, emb_size, dropout)
        self.head_alpha = DeepSynBaPredictionHead(in_dim, emb_size, dropout)

    def forward_tuple(
        self, r_DC: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Returns 8 raw parameter tensors of shape (B, 1) in exact finalcheck order."""
        raw_e1 = self.head_e1(r_DC)
        raw_e2 = self.head_e2(r_DC)
        raw_e3 = self.head_e3(r_DC)
        raw_log_c1 = self.head_log_c1(r_DC)
        raw_log_c2 = self.head_log_c2(r_DC)
        raw_h1 = self.head_h1(r_DC)
        raw_h2 = self.head_h2(r_DC)
        raw_alpha = self.head_alpha(r_DC)
        return raw_e1, raw_e2, raw_e3, raw_log_c1, raw_log_c2, raw_h1, raw_h2, raw_alpha

    def forward(self, r_DC: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass predicting raw parameters.

        Args:
            r_DC: Drug-Cell representation of shape (B, 1536).

        Returns:
            Dictionary mapping parameter names to tensors of shape (B,).
        """
        assert r_DC.shape[-1] == self.in_dim, (
            f"Expected r_DC dim {self.in_dim}, but got {r_DC.shape[-1]}"
        )

        tup = self.forward_tuple(r_DC)
        return {
            "e1_raw": tup[0].squeeze(-1),
            "e2_raw": tup[1].squeeze(-1),
            "e3_raw": tup[2].squeeze(-1),
            "logc1_raw": tup[3].squeeze(-1),
            "logc2_raw": tup[4].squeeze(-1),
            "h1_raw": tup[5].squeeze(-1),
            "h2_raw": tup[6].squeeze(-1),
            "alpha_raw": tup[7].squeeze(-1),
        }

