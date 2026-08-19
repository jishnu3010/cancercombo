"""
DGX Model Training Script for CancerCombo-BRICS-Symmetric.

Designed for high-performance training on NVIDIA DGX A100 / H100 systems.
Features:
    - Config file integration (config.py).
    - GPU acceleration (CUDA) with Automatic Mixed Precision (AMP / FP16).
    - Data loading from data/scenario3_drug1.csv with train/val/test splits.
    - Full training loop with MSE loss, evaluation, and checkpoint saving.
    - Debug printing hook for F_A and F_B fragment embeddings (--debug_print_fragments).
"""

import os
import sys

# Disable Triton JIT compilation to use precompiled cuBLAS CUDA kernels on containers without gcc
os.environ["TRITON_DISABLE"] = "1"

import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from cancer_combo_brics import (
    CancerComboBRICSSymmetric,
    load_cancer_combo_from_csv,
    collate_cancer_combo_batch
)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Helper to move all DataLoader batch tensors to target device (CUDA GPU or CPU)."""
    return {
        "cell_expr": batch["cell_expr"].to(device),
        "fp_A": batch["fp_A"].to(device),
        "mask_A": batch["mask_A"].to(device),
        "fp_B": batch["fp_B"].to(device),
        "mask_B": batch["mask_B"].to(device),
        "dose_grid": (
            batch["dose_grid"][0].to(device),
            batch["dose_grid"][1].to(device)
        ),
        "Y_true": batch["Y_true"].to(device)
    }


def train_epoch(model, loader, optimizer, criterion, scaler, device, debug_print_fragments=False, print_every=100, global_step_offset=0):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, raw_batch in enumerate(loader, start=1 + global_step_offset):
        optimizer.zero_grad()
        batch = move_batch_to_device(raw_batch, device)

        # Automatic Mixed Precision for NVIDIA Tensor Cores (A100 / H100)
        with torch.amp.autocast('cuda', enabled=device.type == "cuda" and config.USE_AMP):
            Y_pred = model(
                cell_expr=batch["cell_expr"],
                drugA_frags=batch["fp_A"],
                drugA_mask=batch["mask_A"],
                drugB_frags=batch["fp_B"],
                drugB_mask=batch["mask_B"],
                dose_grid=batch["dose_grid"],
                debug_print_fragments=debug_print_fragments,
                step=step,
                print_every=print_every
            )
            loss = criterion(Y_pred, batch["Y_true"])

        if scaler is not None and device.type == "cuda" and config.USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1), num_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        with torch.amp.autocast('cuda', enabled=device.type == "cuda" and config.USE_AMP):
            Y_pred = model(
                cell_expr=batch["cell_expr"],
                drugA_frags=batch["fp_A"],
                drugA_mask=batch["mask_A"],
                drugB_frags=batch["fp_B"],
                drugB_mask=batch["mask_B"],
                dose_grid=batch["dose_grid"]
            )
            loss = criterion(Y_pred, batch["Y_true"])

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def compute_evaluation_metrics(y_true_all, y_pred_all):
    """Computes RMSE, MAE, R2, Pearson r, and Spearman rho metrics."""
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    y_t = y_true_all.flatten()
    y_p = y_pred_all.flatten()

    mse = np.mean((y_t - y_p) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_t - y_p)))

    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = float(1.0 - (ss_res / max(ss_tot, 1e-12)))

    p_corr, _ = pearsonr(y_t, y_p)
    s_corr, _ = spearmanr(y_t, y_p)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": float(p_corr) if not np.isnan(p_corr) else 0.0,
        "spearman": float(s_corr) if not np.isnan(s_corr) else 0.0
    }


@torch.no_grad()
def evaluate_full(model, loader, criterion, device):
    """Evaluates loss and returns concatenated predictions and ground truths for metrics."""
    import numpy as np
    model.eval()
    total_loss = 0.0
    num_batches = 0
    preds_list = []
    trues_list = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        with torch.amp.autocast('cuda', enabled=device.type == "cuda" and config.USE_AMP):
            Y_pred = model(
                cell_expr=batch["cell_expr"],
                drugA_frags=batch["fp_A"],
                drugA_mask=batch["mask_A"],
                drugB_frags=batch["fp_B"],
                drugB_mask=batch["mask_B"],
                dose_grid=batch["dose_grid"]
            )
            loss = criterion(Y_pred, batch["Y_true"])

        total_loss += loss.item()
        num_batches += 1

        preds_list.append(Y_pred.cpu().numpy())
        trues_list.append(batch["Y_true"].cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    y_pred_all = np.concatenate(preds_list, axis=0) if preds_list else np.array([])
    y_true_all = np.concatenate(trues_list, axis=0) if trues_list else np.array([])

    metrics = compute_evaluation_metrics(y_true_all, y_pred_all) if len(y_pred_all) > 0 else {}
    metrics["loss"] = avg_loss

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CancerCombo-BRICS-Symmetric E2 Baseline Training")
    parser.add_argument("--data_csv", type=str, default=config.DATA_CSV, help="Path to combination dataset CSV")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--d_dim", type=int, default=config.D_DIM, help="Fragment embedding dimension")
    parser.add_argument("--cell_dim", type=int, default=config.CELL_DIM, help="Cell vector dimension")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS, help="DataLoader num_workers")
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR, help="Directory to save model checkpoints")
    parser.add_argument("--max_samples", type=int, default=config.MAX_SAMPLES, help="Max samples limit (optional)")
    parser.add_argument("--debug_print_fragments", action="store_true", help="Toggle debug printing of raw F_A and F_B fragment embeddings during training")
    parser.add_argument("--print_every", type=int, default=100, help="Print debug fragment stats every N steps (default: 100)")

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = config.DEVICE
    print("=" * 75)
    print("  CancerCombo-BRICS-Symmetric E2 Baseline DGX Training Pipeline")
    print("=" * 75)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")

    # 1. Load TRAIN, VAL, and TEST Datasets
    print(f"\n[1] Loading dataset splits from '{args.data_csv}'...")
    start_time = time.time()
    train_dataset = load_cancer_combo_from_csv(args.data_csv, split=config.TRAIN_SPLIT, max_samples=args.max_samples)
    val_dataset = load_cancer_combo_from_csv(args.data_csv, split=config.VAL_SPLIT, max_samples=args.max_samples)
    test_dataset = load_cancer_combo_from_csv(args.data_csv, split=config.TEST_SPLIT, max_samples=args.max_samples)

    print(f"    - Train dataset loaded: {len(train_dataset)} samples")
    print(f"    - Val dataset loaded:   {len(val_dataset)} samples")
    print(f"    - Test dataset loaded:  {len(test_dataset)} samples ({time.time() - start_time:.2f}s)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_cancer_combo_batch,
        pin_memory=config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_cancer_combo_batch,
        pin_memory=config.PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_cancer_combo_batch,
        pin_memory=config.PIN_MEMORY
    )

    # 2. Instantiate Model
    print("\n[2] Initializing CancerComboBRICSSymmetric Model (E2 Baseline)...")
    model = CancerComboBRICSSymmetric(
        gene_dim=config.GENE_DIM,
        cell_dim=args.cell_dim,
        frag_fp_dim=config.FRAG_FP_DIM,
        d_dim=args.d_dim,
        num_attn_heads=config.NUM_ATTN_HEADS,
        shared_attn_weights=config.SHARED_ATTN_WEIGHTS
    ).to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda" and config.USE_AMP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # 3. Training Loop with Validation Integrity
    print(f"\n[3] Starting Training for {args.epochs} Epochs on {device}...")
    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, n_batches = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            debug_print_fragments=args.debug_print_fragments,
            print_every=args.print_every,
            global_step_offset=global_step
        )
        global_step += n_batches
        scheduler.step()

        # Validation evaluation at end of epoch
        val_metrics = evaluate_full(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]

        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val RMSE: {val_metrics['rmse']:.4f} | Time: {elapsed:.2f}s")

        # Save Checkpoint strictly based on VALIDATION loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = config.BEST_MODEL_PATH
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics
            }, checkpoint_path)

    print("\n" + "=" * 75)
    print(f"Training Complete!")
    print(f"Best Validation Epoch : {best_epoch}")
    print(f"Best Validation Loss  : {best_val_loss:.6f}")
    print(f"Best Checkpoint Path  : {config.BEST_MODEL_PATH}")

    # 4. Final Evaluation on Test Set using Best Checkpoint
    print("\n[4] Evaluating Best Checkpoint on Test Set...")
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_full(model, test_loader, criterion, device)
    print("\n--- FINAL E2 BASELINE TEST METRICS ---")
    print(f"Test MSE Loss : {test_metrics['loss']:.6f}")
    print(f"Test RMSE     : {test_metrics['rmse']:.4f}")
    print(f"Test MAE      : {test_metrics['mae']:.4f}")
    print(f"Test R²       : {test_metrics['r2']:.4f}")
    print(f"Test Pearson  : {test_metrics['pearson']:.4f}")
    print(f"Test Spearman : {test_metrics['spearman']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
