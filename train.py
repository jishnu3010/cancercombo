import os
for _k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "1"

from helpers import enforce_single_thread
enforce_single_thread()

import torch
from torch.utils.data import DataLoader
try:
    import pytorch_lightning as pl  # type: ignore # pyrefly: ignore [missing-import]
    from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore # pyrefly: ignore [missing-import]
except ImportError:
    pl = None
    class ModelCheckpoint:
        def __init__(self, *args, **kwargs): pass

from typing import Optional
from config import load_config
from dataset import DrugComboDataset, load_nci60_gex, load_synergy_dataset, load_precomputed_drug_features
from trainer import CancerComboLightningModule
from cancercombo import CancerCombo
from losses import CancerComboLoss
from helpers import set_seed, generate_mock_data
from logger import setup_logger

import argparse

def run_training(
    config_path: str = "config.yaml",
    epochs: Optional[int] = None,
    max_samples: Optional[int] = None,
    scenario: int = 1,
    engine: str = "auto"
):
    """Initializes dataset generators and executes full model training.

    Args:
        config_path: Path to configuration file.
        epochs: Optional epoch override.
        max_samples: Optional maximum dataset samples limit.
        scenario: The split scenario to use (1, 2, or 3).
        engine: Execution engine: 'auto', 'lightning', or 'native'.
    """
    logger = setup_logger("CancerCombo Train")
    logger.info("Loading configs and setting seed...")
    
    m_config, t_config = load_config(config_path)
    if epochs is not None:
        t_config.epochs = epochs
    set_seed(t_config.seed)
    
    # Configure PyTorch CUDA backends to avoid hangs/deadlocks on GPU container setups
    if torch.cuda.is_available():
        logger.info("Configuring PyTorch CUDA settings...")
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            logger.info("  [SUCCESS] Disabled FlashAttention and MemEfficient Attention SDP backends (preventing CUDA compiler & padding mask hangs).")
        except Exception as e:
            logger.warning(f"  [WARNING] Failed to configure SDPA kernels: {e}")
    
    logger.info("Attempting to load real dataset archives...")
    real_gex = load_nci60_gex("data/features/NCI-60_landmark_gex.csv", target_dim=m_config.cell_in_dim)
    
    # Map scenario number to split file path
    scenario_files = {
        1: "data/splits/scenario1_combination.csv",
        2: "data/splits/scenario2_cell.csv",
        3: "data/splits/scenario3_drug.csv"
    }
    split_path = scenario_files.get(scenario, scenario_files[1])
    
    if not os.path.exists(split_path):
        logger.error(f"Scenario split file not found: {split_path}. Run split_dataset.py first.")
        return
        
    logger.info(f"Loading split scenario from {split_path}...")
    import pandas as pd
    split_df = pd.read_csv(split_path)
    
    if "split" not in split_df.columns:
        logger.error(f"Split file missing 'split' column: {split_path}")
        return
        
    train_df = split_df[split_df["split"] == 1].copy()
    val_df = split_df[split_df["split"] == 2].copy()
    
    if max_samples is not None:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max(1, max_samples // 4))
        
    from dataset import parse_dataframe_to_records
    train_data = parse_dataframe_to_records(train_df, known_gex_dict=real_gex)
    val_data = parse_dataframe_to_records(val_df, known_gex_dict=real_gex)
    cell_features = real_gex
    
    drug_features = load_precomputed_drug_features("data/features/drug_features.pt")
    if not drug_features:
        drug_features = load_precomputed_drug_features("data/features/drug_features.pkl")
    if drug_features:
        logger.info(f"Loaded precomputed drug features for {len(drug_features)} SMILES strings.")
    else:
        logger.info("No precomputed drug feature store found. Falling back to on-the-fly preprocessing.")
    
    train_dataset = DrugComboDataset(train_data, cell_features, drug_feature_store=drug_features)
    val_dataset = DrugComboDataset(val_data, cell_features, drug_feature_store=drug_features)
    
    num_workers = getattr(t_config, "num_workers", 0)
    # Disable pin_memory by default to prevent Docker/Jupyter hub memory-lock (ulimit) deadlocks/slowdowns.
    pin_mem = False
    
    loader_kwargs = {
        "batch_size": t_config.batch_size,
        "pin_memory": pin_mem,
        "num_workers": num_workers
    }
    if num_workers > 0 and os.name != 'nt':
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    use_lightning = (engine == "lightning" or (engine == "auto" and pl is not None and hasattr(pl, "Trainer")))
    
    if use_lightning:
        logger.info("Initializing LightningModule...")
        model = CancerComboLightningModule(m_config, t_config)
        checkpoint_callback = ModelCheckpoint(
            dirpath=t_config.checkpoint_dir,
            filename="cancercombo_best",
            save_top_k=t_config.save_top_k,
            monitor="val_loss",
            mode="min"
        )
        try:
            from pytorch_lightning.callbacks import TQDMProgressBar
            pbar_callback = TQDMProgressBar(refresh_rate=1, leave=True)
            callbacks_list = [checkpoint_callback, pbar_callback]
        except Exception:
            callbacks_list = [checkpoint_callback]
            
        logger.info(f"Starting PyTorch Lightning trainer fit on {accelerator.upper()} for {t_config.epochs} epochs...")
        trainer_kwargs = {
            "max_epochs": t_config.epochs,
            "accelerator": accelerator,
            "devices": 1,
            "gradient_clip_val": 5.0,
            "callbacks": callbacks_list,
            "enable_checkpointing": True,
            "log_every_n_steps": 1,
            "precision": "32-true"
        }
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        logger.info(f"Starting Native PyTorch Training Engine on {accelerator.upper()} for {t_config.epochs} epochs...")
        from tqdm import tqdm
        from metrics import calculate_metrics
        import numpy as np

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = CancerCombo(m_config).to(device)
        loss_fn = CancerComboLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=t_config.lr, weight_decay=t_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        
        best_val_loss = float("inf")
        os.makedirs(t_config.checkpoint_dir, exist_ok=True)
        
        for epoch in range(1, t_config.epochs + 1):
            net.train()
            train_loss_sum = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{t_config.epochs}", leave=True)
            
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                b_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                autocast_ctx = torch.amp.autocast("cuda", enabled=(device.type == "cuda")) if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=(device.type == "cuda"))
                with autocast_ctx:
                    y_pred, params = net(
                        b_gpu["drug_a_ids"], b_gpu["drug_a_mask"], b_gpu["drug_a_morgan"], b_gpu["drug_a_desc"],
                        b_gpu["drug_b_ids"], b_gpu["drug_b_mask"], b_gpu["drug_b_morgan"], b_gpu["drug_b_desc"],
                        b_gpu["cell_line"], b_gpu["doses_a"], b_gpu["doses_b"]
                    )
                    loss = loss_fn(y_pred, b_gpu.get("viability", b_gpu.get("viability_matrix")), params)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss_sum += loss.item()
                pbar.set_postfix({"train_loss_step": f"{loss.item():.4f}"})
                
            train_loss = train_loss_sum / max(len(train_loader), 1)
            
            # Validation step
            net.eval()
            val_loss_sum = 0.0
            val_preds_list, val_trues_list = [], []
            with torch.no_grad():
                for batch in val_loader:
                    b_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    autocast_ctx = torch.amp.autocast("cuda", enabled=(device.type == "cuda")) if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=(device.type == "cuda"))
                    with autocast_ctx:
                        y_pred, params = net(
                            b_gpu["drug_a_ids"], b_gpu["drug_a_mask"], b_gpu["drug_a_morgan"], b_gpu["drug_a_desc"],
                            b_gpu["drug_b_ids"], b_gpu["drug_b_mask"], b_gpu["drug_b_morgan"], b_gpu["drug_b_desc"],
                            b_gpu["cell_line"], b_gpu["doses_a"], b_gpu["doses_b"]
                        )
                        v_loss = loss_fn(y_pred, b_gpu.get("viability", b_gpu.get("viability_matrix")), params)
                    val_loss_sum += v_loss.item()
                    val_preds_list.append(y_pred.detach().cpu().numpy())
                    val_trues_list.append(b_gpu.get("viability", b_gpu.get("viability_matrix")).detach().cpu().numpy())
                    
            val_loss = val_loss_sum / max(len(val_loader), 1)
            scheduler.step(val_loss)
            
            if val_preds_list:
                v_preds = np.concatenate(val_preds_list, axis=0)
                v_trues = np.concatenate(val_trues_list, axis=0)
                val_metrics = calculate_metrics(v_preds, v_trues)
                v_rmse = val_metrics["rmse"]
                v_pearson = val_metrics["pearson"]
                v_spearman = val_metrics["spearman"]
            else:
                v_rmse, v_pearson, v_spearman = 0.0, 0.0, 0.0
                
            logger.info(
                f"Epoch [{epoch}/{t_config.epochs}] Complete | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val RMSE: {v_rmse:.4f} | Val Pearson: {v_pearson:.4f} | Val Spearman: {v_spearman:.4f}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(t_config.checkpoint_dir, "cancercombo_best.ckpt")
                torch.save({"state_dict": net.state_dict(), "config": m_config}, ckpt_path)
                logger.info(f"Saved best model checkpoint to '{ckpt_path}'")
                
    logger.info("Training finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CancerCombo")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--scenario", type=int, default=1, help="Split scenario (1, 2, or 3)")
    parser.add_argument("--engine", type=str, default="auto", choices=["auto", "lightning", "native"], help="Training engine: auto, lightning, or native")
    args = parser.parse_args()
    
    run_training(
        config_path=args.config,
        epochs=args.epochs,
        max_samples=args.max_samples,
        scenario=args.scenario,
        engine=args.engine
    )
