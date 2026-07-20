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

from config import load_config
from dataset import DrugComboDataset, load_nci60_gex, load_synergy_dataset
from trainer import CancerComboLightningModule
from helpers import set_seed, generate_mock_data
from logger import setup_logger

def run_training(config_path: str = "config.yaml"):
    """Initializes dataset generators and executes full model training.

    Args:
        config_path: Path to configuration file.
    """
    logger = setup_logger("CancerCombo Train")
    logger.info("Loading configs and setting seed...")
    
    m_config, t_config = load_config(config_path)
    set_seed(t_config.seed)
    
    logger.info("Attempting to load real dataset archives...")
    real_gex = load_nci60_gex("data/features/NCI-60_landmark_gex.csv", target_dim=m_config.cell_in_dim)
    train_real = load_synergy_dataset("data/DrugCombination_with_SMILES.zip", split='train')
    val_real = load_synergy_dataset("data/DrugCombination_with_SMILES.zip", split='val')
    
    if train_real and len(train_real) >= 5:
        logger.info(f"Loaded real dataset: {len(train_real)} train samples, {len(val_real)} val samples.")
        train_data = train_real
        val_data = val_real
        cell_features = real_gex
    else:
        logger.info("Real dataset archive not found or incomplete. Generating synthetic datasets for simulation...")
        train_data, cell_features = generate_mock_data(64)
        val_data, _ = generate_mock_data(16)
    
    train_dataset = DrugComboDataset(train_data, cell_features)
    val_dataset = DrugComboDataset(val_data, cell_features)
    
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
    
    logger.info("Initializing LightningModule...")
    model = CancerComboLightningModule(m_config, t_config)
    
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    
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
        devices=devices,
        gradient_clip_val=5.0,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        log_every_n_steps=1
    )
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("Training finished successfully.")

if __name__ == "__main__":
    run_training()
