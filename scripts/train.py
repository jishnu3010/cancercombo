"""Production training script for CancerCombo with AMP and gradient stability monitoring."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cancer_combo_brics.config import ExperimentConfig
from cancer_combo_brics.utils import set_seed, save_checkpoint, count_parameters, get_gpu_memory_mb
from cancer_combo_brics.data.dataset import CancerComboDataset, collate_combo_batch
from cancer_combo_brics.data.preprocessing import CellExpressionPreprocessor
from cancer_combo_brics.chemistry.cache import FunctionalGroupCache
from cancer_combo_brics.model import CancerComboBRICS
from cancer_combo_brics.losses import SurfaceRegressionLoss
from cancer_combo_brics.metrics import compute_surface_metrics, evaluate_predictions_grouped
from cancer_combo_brics.diagnostics import compute_gradient_norms, inspect_surface_diagnostics


def build_optimizer_and_scheduler(
    model: CancerComboBRICS,
    config: ExperimentConfig,
    num_training_steps: int,
):
    """Build optimizer for target architecture modules."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    param_groups = [
        {
            "params": trainable_params,
            "lr": config.optimizer.lr_new,
            "weight_decay": config.optimizer.weight_decay,
            "name": "target_architecture",
        }
    ]

    if config.optimizer.type.lower() == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif config.optimizer.type.lower() == "adam":
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)

    # Learning rate scheduler
    if config.optimizer.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_training_steps),
            eta_min=config.optimizer.min_lr,
        )
    else:
        scheduler = None

    return optimizer, scheduler


def train_one_epoch(
    model: CancerComboBRICS,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: torch.cuda.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    config: ExperimentConfig,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    grad_norms = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.training.epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        cell_expr = batch["cell_expr"].to(device, non_blocking=True)
        frags_A = batch["fragments_A"]
        mask_A = batch["mask_A"].to(device, non_blocking=True)
        frags_B = batch["fragments_B"]
        mask_B = batch["mask_B"].to(device, non_blocking=True)
        doses_A = batch["doses_A"].to(device, non_blocking=True)
        doses_B = batch["doses_B"].to(device, non_blocking=True)
        y_true = batch["viability_matrix"].to(device, non_blocking=True)

        use_amp = config.training.mixed_precision and device.type == "cuda"

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            y_pred, _ = model(
                cell_expr=cell_expr,
                fragments_A=frags_A,
                mask_A=mask_A,
                fragments_B=frags_B,
                mask_B=mask_B,
                doses_A=doses_A,
                doses_B=doses_B,
            )
            loss = criterion(y_pred, y_true)
            if config.training.gradient_accumulation_steps > 1:
                loss = loss / config.training.gradient_accumulation_steps

        # Backward pass with AMP
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % config.training.gradient_accumulation_steps == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)

            g_norms = compute_gradient_norms(model)
            global_norm = g_norms["global_grad_norm"]
            grad_norms.append(global_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        batch_loss = loss.item() * (config.training.gradient_accumulation_steps if config.training.gradient_accumulation_steps > 1 else 1.0)
        total_loss += batch_loss
        pbar.set_postfix({"loss": f"{batch_loss:.4f}", "grad_norm": f"{global_norm:.2f}"})

    mean_loss = total_loss / len(loader)
    mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    return {"train_loss": mean_loss, "train_grad_norm": mean_grad_norm}


@torch.no_grad()
def evaluate_model(
    model: CancerComboBRICS,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any], Optional[Dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    records = []
    last_diagnostics = None

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        cell_expr = batch["cell_expr"].to(device, non_blocking=True)
        frags_A = batch["fragments_A"]
        mask_A = batch["mask_A"].to(device, non_blocking=True)
        frags_B = batch["fragments_B"]
        mask_B = batch["mask_B"].to(device, non_blocking=True)
        doses_A = batch["doses_A"].to(device, non_blocking=True)
        doses_B = batch["doses_B"].to(device, non_blocking=True)
        y_true = batch["viability_matrix"].to(device, non_blocking=True)

        y_pred, diag = model(
            cell_expr=cell_expr,
            fragments_A=frags_A,
            mask_A=mask_A,
            fragments_B=frags_B,
            mask_B=mask_B,
            doses_A=doses_A,
            doses_B=doses_B,
            return_diagnostics=True,
        )
        loss = criterion(y_pred, y_true)
        total_loss += loss.item()

        last_diagnostics = inspect_surface_diagnostics(
            y_true=y_true,
            y_pred=y_pred,
            y_hill=diag["Y_hill"],
            bias=diag["bias"],
        )

        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        scenarios_np = batch["scenarios"].numpy()
        cells = batch["cell_lines"]
        pairs = batch["drug_pairs"]

        for i in range(len(cells)):
            records.append({
                "y_true": y_true_np[i],
                "y_pred": y_pred_np[i],
                "scenario": int(scenarios_np[i]),
                "cell_line": cells[i],
                "drug_pair": pairs[i],
            })

    mean_loss = total_loss / len(loader)
    eval_results = evaluate_predictions_grouped(records)
    return mean_loss, eval_results, last_diagnostics


def main():
    parser = argparse.ArgumentParser(description="Train CancerCombo model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Target device (cuda or cpu)")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if os.path.exists(args.config) else ExperimentConfig()
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size

    set_seed(cfg.training.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n[Training] Using device: {device}")

    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)
    cfg.save_yaml(os.path.join(cfg.logging.checkpoint_dir, "config.yaml"))

    comb_candidates = [
        cfg.data.combination_file,
        cfg.data.combination_file.replace("bricks2/", "") if cfg.data.combination_file and cfg.data.combination_file.startswith("bricks2/") else None,
        os.path.join(cfg.data.root, "scenario3_drug_level.csv"),
        "./data/scenario3_drug_level.csv",
    ]
    comb_file = None
    for cand in comb_candidates:
        if cand and os.path.exists(cand):
            comb_file = cand
            break

    if not comb_file:
        raise FileNotFoundError(f"Combinations file not found. Tried paths: {[c for c in comb_candidates if c]}. Check dataset path.")

    df = pd.read_csv(comb_file) if comb_file.endswith(".csv") else pd.read_parquet(comb_file)

    cell_candidates = [
        cfg.data.cell_expression_file,
        cfg.data.cell_expression_file.replace("bricks2/", "") if cfg.data.cell_expression_file and cfg.data.cell_expression_file.startswith("bricks2/") else None,
        os.path.join(cfg.data.root, "cell_line_gene_expr.csv"),
        "./data/cell_line_gene_expr.csv",
    ]
    cell_file = None
    for cand in cell_candidates:
        if cand and os.path.exists(cand):
            cell_file = cand
            break

    if not cell_file:
        raise FileNotFoundError(f"Cell expressions file not found. Tried paths: {[c for c in cell_candidates if c]}. Check dataset path.")

    if cell_file.endswith(".npz"):
        c_data = np.load(cell_file)
        raw_c_matrix = c_data["expressions"]
        c_names = list(c_data["cell_lines"])
    else:
        c_df = pd.read_csv(cell_file, index_col=0)
        raw_c_matrix = c_df.values
        c_names = list(c_df.index)

    preprocessor = None
    if os.path.exists(cfg.data.cell_preprocessor_file):
        preprocessor = CellExpressionPreprocessor.load(cfg.data.cell_preprocessor_file)
        print(f"Loaded cell preprocessor from: {cfg.data.cell_preprocessor_file}")
    else:
        preprocessor = CellExpressionPreprocessor(expected_dim=raw_c_matrix.shape[1])
        train_cells = df[df["split"] == "train"][cfg.data.cell_id_col].unique() if "split" in df.columns else c_names
        train_indices = [i for i, name in enumerate(c_names) if name in train_cells] or list(range(len(c_names)))
        preprocessor.fit(raw_c_matrix[train_indices])
        preprocessor.save(cfg.data.cell_preprocessor_file)

    norm_c_matrix = preprocessor.transform(raw_c_matrix)
    cell_expr_dict = {name: norm_c_matrix[i] for i, name in enumerate(c_names)}

    if "split" in df.columns:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)
    else:
        train_df = df.sample(frac=0.7, random_state=cfg.training.seed).reset_index(drop=True)
        val_df = df.drop(train_df.index).reset_index(drop=True)
        test_df = val_df

    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    fg_cache = FunctionalGroupCache(db_path=cfg.data.fg_cache_file, radius=cfg.data.fg_radius)

    train_dataset = CancerComboDataset(
        train_df, cell_expr_dict, fg_cache,
        smiles_col_a=cfg.data.smiles_col_a, smiles_col_b=cfg.data.smiles_col_b,
        cell_id_col=cfg.data.cell_id_col, dose_col_a=cfg.data.dose_col_a,
        dose_col_b=cfg.data.dose_col_b, viability_col=cfg.data.viability_col,
        drug_id_col_a=cfg.data.drug_id_col_a, drug_id_col_b=cfg.data.drug_id_col_b,
    )
    val_dataset = CancerComboDataset(
        val_df, cell_expr_dict, fg_cache,
        smiles_col_a=cfg.data.smiles_col_a, smiles_col_b=cfg.data.smiles_col_b,
        cell_id_col=cfg.data.cell_id_col, dose_col_a=cfg.data.dose_col_a,
        dose_col_b=cfg.data.dose_col_b, viability_col=cfg.data.viability_col,
        drug_id_col_a=cfg.data.drug_id_col_a, drug_id_col_b=cfg.data.drug_id_col_b,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_combo_batch(b, max_fragments=cfg.data.max_fragments),
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and (device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_combo_batch(b, max_fragments=cfg.data.max_fragments),
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory and (device.type == "cuda"),
    )

    model = CancerComboBRICS(config=cfg.model).to(device)
    param_counts = count_parameters(model)
    print("\n--- Model Parameter Count ---")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")
    print("-----------------------------\n")

    total_steps = len(train_loader) * cfg.training.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.training.mixed_precision and device.type == "cuda"))
    criterion = SurfaceRegressionLoss(loss_type=cfg.training.loss_type, delta=cfg.training.huber_delta)

    best_val_rmse = float("inf")
    history: List[Dict[str, Any]] = []

    print("\nStarting training loop...")
    for epoch in range(cfg.training.epochs):
        t0 = time.time()

        train_res = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, device, cfg, epoch
        )

        val_loss, val_eval, surface_diag = evaluate_model(model, val_loader, criterion, device)
        val_rmse = val_eval["overall"]["rmse"]
        val_r2 = val_eval["overall"]["r2"]
        val_pearson = val_eval["overall"]["pearson"]
        epoch_time = time.time() - t0

        log_row = {
            "epoch": epoch + 1,
            "train_loss": train_res["train_loss"],
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "val_pearson": val_pearson,
            "train_grad_norm": train_res["train_grad_norm"],
            "gpu_memory_mb": get_gpu_memory_mb(),
            "time_sec": epoch_time,
        }
        if surface_diag:
            log_row["bias_to_hill_ratio"] = surface_diag.get("bias_to_hill_ratio", 0.0)

        history.append(log_row)
        print(
            f"Epoch {epoch+1:02d}/{cfg.training.epochs:02d} | "
            f"Train Loss: {train_res['train_loss']:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val R^2: {val_r2:.4f} | "
            f"Grad Norm: {train_res['train_grad_norm']:.2f} | "
            f"Time: {epoch_time:.1f}s"
        )

        save_checkpoint(
            os.path.join(cfg.logging.checkpoint_dir, "last_model.pt"),
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=epoch + 1, best_metric=best_val_rmse, config=cfg.to_dict(),
            seed=cfg.training.seed,
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            save_checkpoint(
                os.path.join(cfg.logging.checkpoint_dir, "best_model.pt"),
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch + 1, best_metric=best_val_rmse, config=cfg.to_dict(),
                seed=cfg.training.seed,
            )
            print(f"  --> Saved new best checkpoint (Val RMSE: {val_rmse:.4f})")

    log_df = pd.DataFrame(history)
    log_df.to_csv(os.path.join(cfg.logging.log_dir, "training_log.csv"), index=False)
    print(f"\nTraining completed. Training log saved to {cfg.logging.log_dir}/training_log.csv")


if __name__ == "__main__":
    main()
