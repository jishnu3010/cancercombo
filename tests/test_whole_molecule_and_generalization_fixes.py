"""Tests for pooling ablations, early stopping, and checkpoint state."""

import os
import pytest
import torch
import torch.nn as nn
import numpy as np

from cancer_combo_brics.config import ModelConfig
from cancer_combo_brics.interaction.pairwise_interaction import ExplicitPairwiseFragmentInteraction
from cancer_combo_brics.model import CancerComboBRICS
from cancer_combo_brics.utils import save_checkpoint, load_checkpoint


def test_mean_vs_mean_max_pooling_ablation():
    """Verify controlled pooling ablation (mean vs mean+max) works cleanly."""
    pw_mean = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512, pooling_mode="mean")
    pw_max = ExplicitPairwiseFragmentInteraction(fragment_dim=512, hidden_dim=512, pooling_mode="mean_max")

    assert pw_mean.pooling_mode == "mean"
    assert pw_max.pooling_mode == "mean_max"

    B = 2
    F_A = torch.randn(B, 3, 512)
    F_B = torch.randn(B, 4, 512)
    mask_A = torch.ones(B, 3)
    mask_B = torch.ones(B, 4)

    r_mean, _ = pw_mean(F_A, F_B, mask_A, mask_B)
    r_max, _ = pw_max(F_A, F_B, mask_A, mask_B)

    assert r_mean.shape == (B, 512)
    assert r_max.shape == (B, 512)
    assert torch.isfinite(r_mean).all()
    assert torch.isfinite(r_max).all()


def test_early_stopping_logic_and_checkpoint_state(tmp_path):
    """Verify early stopping patience, state saving, and restoration."""
    cfg = ModelConfig()
    model = CancerComboBRICS(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ckpt_path = str(tmp_path / "test_ckpt.pt")
    es_state = {
        "patience_counter": 3,
        "best_epoch": 2,
        "best_val_rmse": 28.3309,
        "early_stopping_patience": 10,
    }

    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=5,
        best_metric=28.3309,
        early_stopping_state=es_state,
    )

    loaded = load_checkpoint(ckpt_path, model=model, optimizer=optimizer)
    assert loaded["last_completed_epoch"] == 5
    assert loaded["best_val_rmse"] == 28.3309
    assert loaded["early_stopping_state"]["patience_counter"] == 3
    assert loaded["early_stopping_state"]["best_epoch"] == 2
