import torch
try:
    import pytorch_lightning as pl
except ImportError:
    import torch.nn as _nn
    class _DummyPLModule(_nn.Module):
        def save_hyperparameters(self): pass
        def log(self, *args, **kwargs): pass
    pl = type("pl", (), {"LightningModule": _DummyPLModule})

from config import ModelConfig, TrainingConfig
from cancercombo import CancerCombo
from losses import CancerComboLoss
from metrics import calculate_metrics
import numpy as np
from typing import Dict, Any, Tuple, List

class CancerComboLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper module for training CancerCombo."""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_config = training_config
        
        self.model = CancerCombo(model_config)
        self.loss_fn = CancerComboLoss()
        
        self.validation_step_outputs: List[Dict[str, np.ndarray]] = []
        self.test_step_outputs: List[Dict[str, np.ndarray]] = []
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.model(
            drug_a_ids=batch["drug_a_ids"],
            drug_a_mask=batch["drug_a_mask"],
            drug_a_morgan=batch["drug_a_morgan"],
            drug_a_desc=batch["drug_a_desc"],
            drug_b_ids=batch["drug_b_ids"],
            drug_b_mask=batch["drug_b_mask"],
            drug_b_morgan=batch["drug_b_morgan"],
            drug_b_desc=batch["drug_b_desc"],
            cell_line=batch["cell_line"],
            doses_a=batch["doses_a"],
            doses_b=batch["doses_b"]
        )
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        y_pred, params_pred = self(batch)
        y_true = batch["viability"]
        
        # Extract ground-truth parameter dict if available for auxiliary supervision
        params_true = {p: batch[p] for p in ["e1", "e2", "e3", "log_c1", "log_c2", "h1", "h2", "alpha"] if p in batch}
        
        loss = self.loss_fn(y_pred, y_true, params_pred=params_pred, params_true=params_true if params_true else None)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        y_pred, params_pred = self(batch)
        y_true = batch["viability"]
        
        params_true = {p: batch[p] for p in ["e1", "e2", "e3", "log_c1", "log_c2", "h1", "h2", "alpha"] if p in batch}
        loss = self.loss_fn(y_pred, y_true, params_pred=params_pred, params_true=params_true if params_true else None)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        self.validation_step_outputs.append({
            "pred": y_pred.detach().cpu().numpy(),
            "true": y_true.detach().cpu().numpy()
        })
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        preds = np.concatenate([x["pred"] for x in self.validation_step_outputs], axis=0)
        trues = np.concatenate([x["true"] for x in self.validation_step_outputs], axis=0)
        
        metrics = calculate_metrics(preds, trues)
        self.log("val_rmse", metrics["rmse"], prog_bar=True, sync_dist=True)
        self.log("val_mae", metrics["mae"], prog_bar=False, sync_dist=True)
        self.log("val_r2", metrics["r2"], prog_bar=False, sync_dist=True)
        self.log("val_pearson", metrics["pearson"], prog_bar=True, sync_dist=True)
        self.log("val_spearman", metrics["spearman"], prog_bar=True, sync_dist=True)
        self.log("val_top_k_precision", metrics["top_k_precision"], prog_bar=False, sync_dist=True)
        self.log("val_top_k_recall", metrics["top_k_recall"], prog_bar=False, sync_dist=True)
        self.log("val_top_k_hit_rate", metrics["top_k_hit_rate"], prog_bar=False, sync_dist=True)
        
        self.validation_step_outputs.clear()
        
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        y_pred, _ = self(batch)
        y_true = batch["viability"]
        
        loss = self.loss_fn(y_pred, y_true)
        self.log("test_loss", loss, sync_dist=True)
        
        self.test_step_outputs.append({
            "pred": y_pred.detach().cpu().numpy(),
            "true": y_true.detach().cpu().numpy()
        })
        return loss

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
            
        preds = np.concatenate([x["pred"] for x in self.test_step_outputs], axis=0)
        trues = np.concatenate([x["true"] for x in self.test_step_outputs], axis=0)
        
        metrics = calculate_metrics(preds, trues)
        self.log("test_rmse", metrics["rmse"], sync_dist=True)
        self.log("test_pearson", metrics["pearson"], sync_dist=True)
        self.log("test_spearman", metrics["spearman"], sync_dist=True)
        
        self.test_step_outputs.clear()
        
    def configure_optimizers(self) -> Dict[str, Any]:
        opt_name = getattr(self.training_config, "optimizer_name", "AdamW").lower()
        if opt_name == "adam":
            opt_cls = torch.optim.Adam
        else:
            opt_cls = torch.optim.AdamW

        optimizer = opt_cls(
            self.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay
        )
        factor = getattr(self.training_config, "scheduler_factor", 0.5)
        patience = getattr(self.training_config, "scheduler_patience", 3)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
