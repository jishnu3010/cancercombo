import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from config import load_config
from dataset import DrugComboDataset
from trainer import CancerComboLightningModule
from helpers import generate_mock_data, set_seed
from logger import setup_logger
from typing import Dict, Any

class Experimenter:
    """Manages structural and hyperparameter experiments for comparison reports."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.logger = setup_logger("CancerCombo Experimenter")
        
    def run_ablation_study(self) -> Dict[str, Dict[str, float]]:
        """Compares drug-drug cross attention enabled vs disabled configurations.

        Returns:
            Dict[str, Dict[str, float]]: Validation performance results of both configurations.
        """
        self.logger.info("Initializing ablation data...")
        m_config, t_config = load_config(self.config_path)
        set_seed(t_config.seed)
        
        train_data, cell_features = generate_mock_data(32)
        val_data, _ = generate_mock_data(8)
        
        train_dataset = DrugComboDataset(train_data, cell_features)
        val_dataset = DrugComboDataset(val_data, cell_features)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        results = {}
        
        # Test Case 1: Drug-Drug Attention Disabled (Default)
        self.logger.info("--- Ablation Case 1: Drug-Drug Cross-Attention DISABLED ---")
        m_config.enable_drug_drug_attention = False
        model_1 = CancerComboLightningModule(m_config, t_config)
        trainer_1 = pl.Trainer(max_epochs=2, accelerator="cpu", devices=1, enable_checkpointing=False, logger=False)
        trainer_1.fit(model_1, train_dataloaders=train_loader, val_dataloaders=val_loader)
        val_res_1 = trainer_1.validate(model_1, dataloaders=val_loader, verbose=False)[0]
        results["disabled"] = val_res_1
        
        # Test Case 2: Drug-Drug Attention Enabled
        self.logger.info("--- Ablation Case 2: Drug-Drug Cross-Attention ENABLED ---")
        m_config.enable_drug_drug_attention = True
        model_2 = CancerComboLightningModule(m_config, t_config)
        trainer_2 = pl.Trainer(max_epochs=2, accelerator="cpu", devices=1, enable_checkpointing=False, logger=False)
        trainer_2.fit(model_2, train_dataloaders=train_loader, val_dataloaders=val_loader)
        val_res_2 = trainer_2.validate(model_2, dataloaders=val_loader, verbose=False)[0]
        results["enabled"] = val_res_2
        
        self.logger.info("Ablation Study Results Comparison:")
        self.logger.info(f"  Disabled DD Attention - Val Loss: {val_res_1.get('val_loss'):.4f}")
        self.logger.info(f"  Enabled DD Attention  - Val Loss: {val_res_2.get('val_loss'):.4f}")
        
        return results

if __name__ == "__main__":
    exp = Experimenter()
    exp.run_ablation_study()
