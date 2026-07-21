import os
import torch
from torch.utils.data import DataLoader
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
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

def run_training(config_path: str = "config.yaml", epochs: Optional[int] = None, max_samples: Optional[int] = None, scenario: int = 1):
    """Initializes dataset generators and executes full model training.

    Args:
        config_path: Path to configuration file.
        epochs: Optional epoch override.
        max_samples: Optional maximum dataset samples limit.
        scenario: The split scenario to use (1, 2, or 3).
    """
    logger = setup_logger("CancerCombo Train")
    logger.info("Loading configs and setting seed...")
    
    m_config, t_config = load_config(config_path)
    if epochs is not None:
        t_config.epochs = epochs
    set_seed(t_config.seed)
    
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
        val_df = val_df.head(max_samples // 4)
        
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
    pin_mem = torch.cuda.is_available()
    
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
    
    if pl is not None:
        logger.info("Initializing LightningModule...")
        model = CancerComboLightningModule(m_config, t_config)
        checkpoint_callback = ModelCheckpoint(
            dirpath=t_config.checkpoint_dir,
            filename="cancercombo_best",
            save_top_k=t_config.save_top_k,
            monitor="val_loss",
            mode="min"
        )
        logger.info(f"Starting trainer fit on {accelerator.upper()} for {t_config.epochs} epochs...")
        trainer = pl.Trainer(
            max_epochs=t_config.epochs,
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=5.0,
            callbacks=[checkpoint_callback],
            enable_checkpointing=True,
            log_every_n_steps=1
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        logger.info(f"PyTorch Lightning not found. Starting Native PyTorch training loop on {accelerator.upper()} for {t_config.epochs} epochs...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = CancerCombo(m_config).to(device)
        loss_fn = CancerComboLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=t_config.lr, weight_decay=t_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        best_val_loss = float("inf")
        os.makedirs(t_config.checkpoint_dir, exist_ok=True)
        
        for epoch in range(1, t_config.epochs + 1):
            net.train()
            train_loss_sum = 0.0
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                b_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                y_pred, params = net(
                    b_gpu["drug_a_ids"], b_gpu["drug_a_mask"], b_gpu["drug_a_morgan"], b_gpu["drug_a_desc"],
                    b_gpu["drug_b_ids"], b_gpu["drug_b_mask"], b_gpu["drug_b_morgan"], b_gpu["drug_b_desc"],
                    b_gpu["cell_line"], b_gpu["doses_a"], b_gpu["doses_b"]
                )
                loss = loss_fn(y_pred, b_gpu["viability_matrix"], params)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss_sum += loss.item()
                
            train_loss = train_loss_sum / max(len(train_loader), 1)
            
            # Validation step
            net.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    b_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    y_pred, params = net(
                        b_gpu["drug_a_ids"], b_gpu["drug_a_mask"], b_gpu["drug_a_morgan"], b_gpu["drug_a_desc"],
                        b_gpu["drug_b_ids"], b_gpu["drug_b_mask"], b_gpu["drug_b_morgan"], b_gpu["drug_b_desc"],
                        b_gpu["cell_line"], b_gpu["doses_a"], b_gpu["doses_b"]
                    )
                    v_loss = loss_fn(y_pred, b_gpu["viability_matrix"], params)
                    val_loss_sum += v_loss.item()
                    
            val_loss = val_loss_sum / max(len(val_loader), 1)
            scheduler.step(val_loss)
            
            logger.info(f"Epoch [{epoch}/{t_config.epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
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
    args = parser.parse_args()
    
    run_training(
        config_path=args.config,
        epochs=args.epochs,
        max_samples=args.max_samples,
        scenario=args.scenario
    )
