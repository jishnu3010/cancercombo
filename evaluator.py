import torch
from torch.utils.data import DataLoader
import numpy as np
from metrics import calculate_metrics
from typing import Dict

class ModelEvaluator:
    """Evaluates the CancerCombo model metrics on validation or test sets."""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        
    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Runs validation loop and calculates standard synergy matrix prediction metrics.

        Args:
            model: PyTorch model.
            dataloader: Test/Validation PyTorch DataLoader.

        Returns:
            Dict[str, float]: Dictionary containing correlation metrics and error scores.
        """
        model.eval()
        model.to(self.device)
        
        preds_list = []
        trues_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                drug_a_ids = batch["drug_a_ids"].to(self.device)
                drug_a_mask = batch["drug_a_mask"].to(self.device)
                drug_a_morgan = batch["drug_a_morgan"].to(self.device)
                drug_a_desc = batch["drug_a_desc"].to(self.device)
                
                drug_b_ids = batch["drug_b_ids"].to(self.device)
                drug_b_mask = batch["drug_b_mask"].to(self.device)
                drug_b_morgan = batch["drug_b_morgan"].to(self.device)
                drug_b_desc = batch["drug_b_desc"].to(self.device)
                
                cell_line = batch["cell_line"].to(self.device)
                doses_a = batch["doses_a"].to(self.device)
                doses_b = batch["doses_b"].to(self.device)
                viability = batch["viability"].to(self.device)
                
                y_pred, _ = model(
                    drug_a_ids=drug_a_ids,
                    drug_a_mask=drug_a_mask,
                    drug_a_morgan=drug_a_morgan,
                    drug_a_desc=drug_a_desc,
                    drug_b_ids=drug_b_ids,
                    drug_b_mask=drug_b_mask,
                    drug_b_morgan=drug_b_morgan,
                    drug_b_desc=drug_b_desc,
                    cell_line=cell_line,
                    doses_a=doses_a,
                    doses_b=doses_b
                )
                
                preds_list.append(y_pred.cpu().numpy())
                trues_list.append(viability.cpu().numpy())
                
        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)
        
        return calculate_metrics(preds, trues)
