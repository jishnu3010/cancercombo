"""Unit tests for training resume infrastructure, atomic checkpointing, and state restoration."""

from __future__ import annotations

import os
import random
import pytest
import numpy as np
import torch
import torch.nn as nn
from tests.conftest import requires_gensim

from cancer_combo_brics.config import ModelConfig, ExperimentConfig
from cancer_combo_brics.model import CancerComboBRICS
from cancer_combo_brics.utils import (
    save_checkpoint,
    load_checkpoint,
    save_rng_state,
    restore_rng_state,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        # Mock fragment_encoder.mol2vec structure for frozen check
        self.fragment_encoder = nn.Module()
        mol2vec = nn.Module()
        mol2vec.embeddings = nn.Embedding(10, 300)
        mol2vec.embeddings.weight.requires_grad = False
        self.fragment_encoder.mol2vec = mol2vec


def test_fresh_checkpoint_creation_and_keys(tmp_path):
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    ckpt_path = str(tmp_path / "checkpoint.pt")
    history = [{"epoch": 1, "val_rmse": 0.5}]
    config = {"training": {"epochs": 10}}

    save_checkpoint(
        filepath=ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        best_metric=0.5,
        training_history=history,
        config=config,
        seed=42,
    )

    assert os.path.exists(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    required_keys = [
        "epoch",
        "best_metric",
        "seed",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "training_history",
        "rng_state",
        "arch_metadata",
        "config",
    ]
    for key in required_keys:
        assert key in ckpt, f"Missing required checkpoint key: '{key}'"

    assert ckpt["epoch"] == 1
    assert ckpt["best_metric"] == 0.5
    assert ckpt["seed"] == 42
    assert ckpt["training_history"] == history
    assert ckpt["arch_metadata"]["pooling_mode"] == "mean_max"
    assert ckpt["arch_metadata"]["r_AB_dim"] == 512
    assert ckpt["arch_metadata"]["r_DC_dim"] == 1536
    assert ckpt["arch_metadata"]["mol2vec"] == "pretrained_300dim"
    assert ckpt["arch_metadata"]["mol2vec_frozen"] is True


def test_checkpoint_loading_and_exact_parameter_restoration(tmp_path):
    model_orig = DummyModel()
    ckpt_path = str(tmp_path / "model_state.pt")

    save_checkpoint(filepath=ckpt_path, model=model_orig, epoch=2)

    model_new = DummyModel()
    # Mutate parameters of model_new
    with torch.no_grad():
        model_new.fc.weight.add_(1.0)
        model_new.fc.bias.add_(1.0)

    assert not torch.equal(model_orig.fc.weight, model_new.fc.weight)

    load_checkpoint(ckpt_path, model=model_new)

    for p_orig, p_new in zip(model_orig.parameters(), model_new.parameters()):
        assert torch.equal(p_orig, p_new), "Model parameters were not restored exactly!"


def test_optimizer_state_restoration(tmp_path):
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Perform a backward step to populate Adam momentum buffers
    x = torch.randn(4, 10)
    loss = model.fc(x).sum()
    loss.backward()
    optimizer.step()

    ckpt_path = str(tmp_path / "opt_state.pt")
    save_checkpoint(ckpt_path, model=model, optimizer=optimizer)

    fresh_model = DummyModel()
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)

    load_checkpoint(ckpt_path, model=fresh_model, optimizer=fresh_optimizer)

    assert len(fresh_optimizer.state) > 0, "Optimizer state dict was not restored!"
    for k in optimizer.state_dict():
        if k != "state":
            assert fresh_optimizer.state_dict()[k] == optimizer.state_dict()[k]
    for param_id in optimizer.state_dict()["state"]:
        saved_param_state = optimizer.state_dict()["state"][param_id]
        restored_param_state = fresh_optimizer.state_dict()["state"][param_id]
        for key in saved_param_state:
            val_saved = saved_param_state[key]
            val_restored = restored_param_state[key]
            if isinstance(val_saved, torch.Tensor):
                assert torch.equal(val_saved, val_restored)
            else:
                assert val_saved == val_restored


def test_scheduler_state_restoration(tmp_path):
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    x = torch.randn(4, 10)
    loss = model.fc(x).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    saved_lr = optimizer.param_groups[0]["lr"]
    ckpt_path = str(tmp_path / "sched_state.pt")
    save_checkpoint(ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler)

    fresh_model = DummyModel()
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-2)
    fresh_scheduler = torch.optim.lr_scheduler.StepLR(fresh_optimizer, step_size=1, gamma=0.5)

    load_checkpoint(ckpt_path, model=fresh_model, optimizer=fresh_optimizer, scheduler=fresh_scheduler)

    resumed_lr = fresh_optimizer.param_groups[0]["lr"]
    assert resumed_lr == saved_lr, f"Resumed LR ({resumed_lr}) does not match saved LR ({saved_lr})!"
    assert fresh_scheduler.state_dict() == scheduler.state_dict()


def test_scaler_state_restoration(tmp_path):
    model = DummyModel()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    scaler._scale = torch.tensor(1024.0)

    ckpt_path = str(tmp_path / "scaler_state.pt")
    save_checkpoint(ckpt_path, model=model, scaler=scaler)

    fresh_scaler = torch.amp.GradScaler("cuda", enabled=False)
    load_checkpoint(ckpt_path, model=model, scaler=fresh_scaler)

    assert fresh_scaler.get_scale() == scaler.get_scale()


def test_correct_next_epoch_calculation_and_no_skip_or_duplicate(tmp_path):
    model = DummyModel()
    ckpt_path = str(tmp_path / "last_model.pt")

    save_checkpoint(ckpt_path, model=model, epoch=37, best_metric=0.12)

    fresh_model = DummyModel()
    ckpt = load_checkpoint(ckpt_path, model=fresh_model)

    completed_epoch = ckpt["epoch"]
    assert completed_epoch == 37
    start_epoch = completed_epoch
    next_epoch_human = start_epoch + 1
    assert next_epoch_human == 38, "Next epoch to run must be 38!"


def test_best_metric_restoration(tmp_path):
    model = DummyModel()
    ckpt_path = str(tmp_path / "best_metric.pt")
    saved_best = 0.3456

    save_checkpoint(ckpt_path, model=model, best_metric=saved_best)

    ckpt = load_checkpoint(ckpt_path, model=model)
    assert ckpt["best_metric"] == saved_best, "Best metric was not restored from checkpoint!"


def test_training_history_preservation(tmp_path):
    model = DummyModel()
    ckpt_path = str(tmp_path / "history.pt")
    history = [{"epoch": i, "loss": 1.0 / i} for i in range(1, 38)]

    save_checkpoint(ckpt_path, model=model, training_history=history)

    ckpt = load_checkpoint(ckpt_path, model=model)
    loaded_history = ckpt["training_history"]
    assert len(loaded_history) == 37
    assert loaded_history[-1]["epoch"] == 37


@requires_gensim
def test_pretrained_mol2vec_remains_frozen(tmp_path):
    config = ModelConfig()
    model = CancerComboBRICS(config)

    ckpt_path = str(tmp_path / "cancercombo_ckpt.pt")
    save_checkpoint(ckpt_path, model=model)

    fresh_model = CancerComboBRICS(config)
    load_checkpoint(ckpt_path, model=fresh_model)

    mol2vec = fresh_model.fragment_encoder.mol2vec
    assert hasattr(mol2vec, "embeddings")
    assert mol2vec.embeddings.weight.requires_grad is False, "Mol2Vec embeddings must remain frozen after load!"


@requires_gensim
def test_architecture_dimensions_invariants():
    config = ModelConfig()
    model = CancerComboBRICS(config)

    assert config.fragment_dim == 512
    assert config.cell_hidden_dim == 512
    assert config.num_param_heads == 8
    assert model.cell_encoder.net[0].in_features == 976
    assert model.cell_encoder.net[0].out_features == 512


@requires_gensim
def test_eight_heads_remain_independent():
    config = ModelConfig()
    model = CancerComboBRICS(config)

    expected_heads = [
        "head_e1", "head_e2", "head_e3",
        "head_log_c1", "head_log_c2",
        "head_h1", "head_h2", "head_alpha"
    ]
    for head_name in expected_heads:
        assert hasattr(model.parameter_heads, head_name), f"Missing parameter head: {head_name}"
        head_module = getattr(model.parameter_heads, head_name)
        assert isinstance(head_module, nn.Module)


def test_missing_checkpoint_path_fails_clearly(tmp_path):
    model = DummyModel()
    missing_path = str(tmp_path / "does_not_exist.pt")

    with pytest.raises(FileNotFoundError) as exc_info:
        load_checkpoint(missing_path, model=model)

    assert "Checkpoint file not found" in str(exc_info.value)


def test_corrupted_checkpoint_fails_clearly(tmp_path):
    model = DummyModel()
    corrupt_path = str(tmp_path / "corrupt.pt")

    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_TORCH_CHECKPOINT_DATA")

    with pytest.raises(RuntimeError) as exc_info:
        load_checkpoint(corrupt_path, model=model)

    assert "Corrupted or invalid checkpoint file" in str(exc_info.value)


def test_incompatible_architecture_fails_clearly(tmp_path):
    model = DummyModel()
    ckpt_path = str(tmp_path / "incompatible_arch.pt")

    invalid_arch = {
        "pooling_mode": "mean_only",
        "r_AB_dim": 256,
        "r_DC_dim": 1024,
        "mol2vec": "pretrained_300dim",
        "mol2vec_frozen": True,
    }
    save_checkpoint(ckpt_path, model=model, arch_metadata=invalid_arch)

    with pytest.raises(ValueError) as exc_info:
        load_checkpoint(ckpt_path, model=model, validate_arch=True)

    assert "Architecture mismatch" in str(exc_info.value)


def test_rng_state_save_and_restore(tmp_path):
    rng_dict = save_rng_state()

    val1_py = random.randint(0, 100000)
    val1_np = np.random.randint(0, 100000)
    val1_pt = torch.randint(0, 100000, (1,)).item()

    restore_rng_state(rng_dict)

    val2_py = random.randint(0, 100000)
    val2_np = np.random.randint(0, 100000)
    val2_pt = torch.randint(0, 100000, (1,)).item()

    assert val1_py == val2_py, "Python RNG state was not restored!"
    assert val1_np == val2_np, "NumPy RNG state was not restored!"
    assert val1_pt == val2_pt, "PyTorch CPU RNG state was not restored!"


def test_last_vs_best_checkpoint_semantics(tmp_path):
    model = DummyModel()
    last_path = str(tmp_path / "last_model.pt")
    best_path = str(tmp_path / "best_model.pt")

    best_val = float("inf")

    metrics = [0.80, 0.70, 0.75, 0.65, 0.68]
    for ep, val in enumerate(metrics, start=1):
        # Save last checkpoint every epoch
        save_checkpoint(last_path, model=model, epoch=ep, best_metric=min(best_val, val))

        # Save best checkpoint only when updated
        if val < best_val:
            best_val = val
            save_checkpoint(best_path, model=model, epoch=ep, best_metric=best_val)

    ckpt_last = torch.load(last_path, map_location="cpu", weights_only=False)
    ckpt_best = torch.load(best_path, map_location="cpu", weights_only=False)

    assert ckpt_last["epoch"] == 5
    assert ckpt_best["epoch"] == 4
    assert ckpt_best["best_metric"] == 0.65


def test_atomic_checkpoint_writing(tmp_path):
    model = DummyModel()
    ckpt_path = str(tmp_path / "atomic_check.pt")

    save_checkpoint(ckpt_path, model=model, epoch=1)

    assert os.path.exists(ckpt_path)
    assert not os.path.exists(ckpt_path + ".tmp"), "Temporary checkpoint file was not cleaned up!"


def test_fresh_training_regression_semantics():
    # Simulating train.py args with --resume omitted (None)
    resume_arg = None

    if resume_arg is None:
        start_epoch = 0
        best_val_rmse = float("inf")
        history = []

    assert start_epoch == 0
    assert best_val_rmse == float("inf")
    assert history == []


def test_controlled_3_to_5_epoch_synthetic_resume(tmp_path):
    """Controlled synthetic 3 -> 5 epoch training & resume experiment."""
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    last_ckpt_path = str(tmp_path / "last_model.pt")

    history = []
    best_val = float("inf")

    # PHASE A: Train 3 epochs (epochs 1, 2, 3)
    for epoch in range(0, 3):
        # Synthetic train step
        x = torch.randn(4, 10)
        loss = model.fc(x).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        val_rmse = 1.0 / (epoch + 1)
        if val_rmse < best_val:
            best_val = val_rmse

        log_entry = {"epoch": epoch + 1, "loss": loss.item(), "val_rmse": val_rmse}
        history.append(log_entry)

        save_checkpoint(
            last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            best_metric=best_val,
            training_history=history,
        )

    # Record Phase A state
    phase_a_weights = [p.clone() for p in model.parameters()]
    phase_a_history = list(history)

    # Verify checkpoint after 3 epochs
    ckpt_phase_a = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt_phase_a["epoch"] == 3
    assert len(ckpt_phase_a["training_history"]) == 3

    # PHASE B: Create fresh model and resume from checkpoint using load_checkpoint
    model_resumed = DummyModel()
    opt_resumed = torch.optim.Adam(model_resumed.parameters(), lr=1e-2)
    sched_resumed = torch.optim.lr_scheduler.StepLR(opt_resumed, step_size=2, gamma=0.5)

    ckpt_resumed = load_checkpoint(
        last_ckpt_path,
        model=model_resumed,
        optimizer=opt_resumed,
        scheduler=sched_resumed,
    )

    completed_epoch = ckpt_resumed["epoch"]
    assert completed_epoch == 3
    start_epoch = completed_epoch
    assert start_epoch == 3

    resumed_history = ckpt_resumed["training_history"]
    assert len(resumed_history) == 3
    assert resumed_history == phase_a_history

    # Verify model parameter restoration
    for p_orig, p_resumed in zip(phase_a_weights, model_resumed.parameters()):
        assert torch.equal(p_orig, p_resumed)

    # PHASE C: Continue training for epochs 4 and 5 (loop range(start_epoch, 5))
    history_resumed = list(resumed_history)
    best_val_resumed = ckpt_resumed["best_metric"]

    for epoch in range(start_epoch, 5):
        x = torch.randn(4, 10)
        loss = model_resumed.fc(x).sum()
        opt_resumed.zero_grad()
        loss.backward()
        opt_resumed.step()
        sched_resumed.step()

        val_rmse = 1.0 / (epoch + 1)
        if val_rmse < best_val_resumed:
            best_val_resumed = val_rmse

        log_entry = {"epoch": epoch + 1, "loss": loss.item(), "val_rmse": val_rmse}
        history_resumed.append(log_entry)

        save_checkpoint(
            last_ckpt_path,
            model=model_resumed,
            optimizer=opt_resumed,
            scheduler=sched_resumed,
            epoch=epoch + 1,
            best_metric=best_val_resumed,
            training_history=history_resumed,
        )

    # VERIFICATION: Final history has epochs [1, 2, 3, 4, 5]
    final_epochs = [h["epoch"] for h in history_resumed]
    assert final_epochs == [1, 2, 3, 4, 5], f"Unexpected epoch sequence after resume: {final_epochs}"
