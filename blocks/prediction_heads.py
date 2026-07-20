import math
import torch
import torch.nn as nn
from config import ModelConfig
from typing import Tuple

class DeepSynBaPredictionHeads(nn.Module):
    """Eight independent MLP heads predicting constrained pharmacological parameters."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_ff = config.d_ff
        
        # Calculate log-space boundaries for C1 and C2
        c_min_safe = max(config.c_min, 1e-12)
        c_max_safe = max(config.c_max, 1e-12)
        self.log_c_min = math.log(c_min_safe)
        self.log_c_max = math.log(c_max_safe)
        
        # Instantiate independent parameter MLP heads
        self.head_e1 = self._build_head(d_model, d_ff)
        self.head_e2 = self._build_head(d_model, d_ff)
        self.head_e3 = self._build_head(d_model, d_ff)
        self.head_log_c1 = self._build_head(d_model, d_ff)
        self.head_log_c2 = self._build_head(d_model, d_ff)
        self.head_h1 = self._build_head(d_model, d_ff)
        self.head_h2 = self._build_head(d_model, d_ff)
        self.head_alpha = self._build_head(d_model, d_ff)
        
    def _build_head(self, d_model: int, d_ff: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d_ff, 1)
        )
        
    def forward(self, z_combo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts e1, e2, e3, log_c1, log_c2, h1, h2, and alpha parameters under configuration boundaries.

        Args:
            z_combo: Combination feature tensor of shape (B, d_model).

        Returns:
            Tuple[torch.Tensor, ...]: Constrained scalars of shape (B, 1) each.
        """
        raw_e1 = self.head_e1(z_combo)
        raw_e2 = self.head_e2(z_combo)
        raw_e3 = self.head_e3(z_combo)
        raw_log_c1 = self.head_log_c1(z_combo)
        raw_log_c2 = self.head_log_c2(z_combo)
        raw_h1 = self.head_h1(z_combo)
        raw_h2 = self.head_h2(z_combo)
        raw_alpha = self.head_alpha(z_combo)
        
        # Sigmoid scaling maps to physiological ranges
        e1 = self.config.e_min + (self.config.e_max - self.config.e_min) * torch.sigmoid(raw_e1)
        e2 = self.config.e_min + (self.config.e_max - self.config.e_min) * torch.sigmoid(raw_e2)
        e3 = self.config.e_min + (self.config.e_max - self.config.e_min) * torch.sigmoid(raw_e3)
        
        # Log-space scaling for C1 and C2 prevents gradient instability across orders of magnitude
        log_c1 = self.log_c_min + (self.log_c_max - self.log_c_min) * torch.sigmoid(raw_log_c1)
        log_c2 = self.log_c_min + (self.log_c_max - self.log_c_min) * torch.sigmoid(raw_log_c2)
        
        h1 = self.config.h_min + (self.config.h_max - self.config.h_min) * torch.sigmoid(raw_h1)
        h2 = self.config.h_min + (self.config.h_max - self.config.h_min) * torch.sigmoid(raw_h2)
        
        alpha = self.config.alpha_min + (self.config.alpha_max - self.config.alpha_min) * torch.sigmoid(raw_alpha)
        
        return e1, e2, e3, log_c1, log_c2, h1, h2, alpha

