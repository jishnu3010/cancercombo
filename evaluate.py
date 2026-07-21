import os
import torch
from torch.utils.data import DataLoader
from config import load_config
from dataset import DrugComboDataset
from cancercombo import CancerCombo
from helpers import generate_mock_data
from logger import setup_logger
from metrics import calculate_metrics
import numpy as np


class ModelEvaluator:
    """Evaluates the CancerCombo model metrics on validation or test sets."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader):
        """Runs validation loop and calculates standard synergy matrix prediction metrics."""
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
                    doses_b=doses_b,
                )

                preds_list.append(y_pred.cpu().numpy())
                trues_list.append(viability.cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)

        return calculate_metrics(preds, trues)

def run_evaluation(checkpoint_path: str = "checkpoints/cancercombo_best.ckpt", config_path: str = "config.yaml"):
    """Load model checkpoint and evaluate performance.

    Args:
        checkpoint_path: Path to checkpoint.
        config_path: Path to configuration YAML.
    """
    logger = setup_logger("CancerCombo Eval")
    logger.info("Setting up configs and mock evaluation dataset...")
    
    m_config, _ = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Generating evaluation dataset...")
    test_data, cell_features = generate_mock_data(32)
    test_dataset = DrugComboDataset(test_data, cell_features)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint path not found: {checkpoint_path}")
        return
        
    model = CancerCombo(m_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Strip PyTorch Lightning 'model.' prefix if present
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    evaluator = ModelEvaluator(device=device)
    logger.info("Evaluating...")
    results = evaluator.evaluate(model, test_loader)
    
    logger.info("Evaluation results:")
    for metric, val in results.items():
        logger.info(f"  {metric.upper()}: {val:.4f}")

if __name__ == "__main__":
    ckpt_path = "checkpoints/cancercombo_best.ckpt"
    if not os.path.exists(ckpt_path):
        if os.path.exists("checkpoints"):
            files = [f for f in os.listdir("checkpoints") if f.endswith(".ckpt")]
            if files:
                ckpt_path = os.path.join("checkpoints", files[0])
    run_evaluation(ckpt_path)
