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


def main():
    parser = argparse.ArgumentParser(description="CancerCombo-BRICS-Symmetric DGX Training")
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
    print("  CancerCombo-BRICS-Symmetric DGX Training Pipeline")
    print("=" * 75)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")

    # 1. Load Dataset
    print(f"\n[1] Loading dataset from '{args.data_csv}'...")
    start_time = time.time()
    train_dataset = load_cancer_combo_from_csv(args.data_csv, split=config.TRAIN_SPLIT, max_samples=args.max_samples)
    print(f"    - Train dataset loaded: {len(train_dataset)} samples ({time.time() - start_time:.2f}s)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_cancer_combo_batch,
        pin_memory=config.PIN_MEMORY
    )

    # 2. Instantiate Model
    print("\n[2] Initializing CancerComboBRICSSymmetric Model...")
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

    # 3. Training Loop
    print(f"\n[3] Starting Training for {args.epochs} Epochs on {device}...")
    best_loss = float("inf")
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

        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train MSE Loss: {train_loss:.6f} | Time: {elapsed:.2f}s")

        # Save Checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = config.BEST_MODEL_PATH
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, checkpoint_path)

    print("\n" + "=" * 75)
    print(f"Training Complete! Best MSE Loss: {best_loss:.6f}")
    print(f"Checkpoint saved to: {config.BEST_MODEL_PATH}")
    print("=" * 75)


if __name__ == "__main__":
    main()
