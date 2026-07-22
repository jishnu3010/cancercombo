import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

import pandas as pd
from torch.utils.data import DataLoader
from config import load_config
from dataset import DrugComboDataset, load_nci60_gex
from cancercombo import CancerCombo
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

def run_evaluation(checkpoint_path: str = "checkpoints/cancercombo_best.ckpt", config_path: str = "config.yaml", scenario: int = 1):
    """Load model checkpoint and evaluate performance.

    Args:
        checkpoint_path: Path to checkpoint.
        config_path: Path to configuration YAML.
        scenario: Split scenario (1, 2, or 3).
    """
    logger = setup_logger("CancerCombo Eval")
    logger.info("Setting up configs and real held-out evaluation dataset...")
    
    m_config, _ = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    scenario_files = {
        1: "data/splits/scenario1_combination.csv",
        2: "data/splits/scenario2_cell.csv",
        3: "data/splits/scenario3_drug.csv"
    }
    split_path = scenario_files.get(scenario, scenario_files[1])
    
    if not os.path.exists(split_path):
        logger.error(
            f"Held-out split file not found: {split_path}. "
            "Run split_dataset.py first and save the scenario split there."
        )
        return

    logger.info(f"Loading held-out test split from {split_path}...")
    split_df = pd.read_csv(split_path)
    if "split" not in split_df.columns:
        logger.error(f"Split file does not contain a 'split' column: {split_path}")
        return

    test_df = split_df[split_df["split"] == 3].copy()
    if test_df.empty:
        logger.error(f"No held-out test rows found in {split_path} (split == 3).")
        return

    cell_features = load_nci60_gex("data/features/NCI-60_landmark_gex.csv", target_dim=m_config.cell_in_dim)
    if not cell_features:
        logger.error("Cell feature file not found or unreadable: data/features/NCI-60_landmark_gex.csv")
        return

    from dataset import parse_dataframe_to_records, load_precomputed_drug_features
    test_records = parse_dataframe_to_records(test_df, known_gex_dict=cell_features)
    drug_features = load_precomputed_drug_features("data/features/drug_features.pt")
    if not drug_features:
        drug_features = load_precomputed_drug_features("data/features/drug_features.pkl")
    test_dataset = DrugComboDataset(test_records, cell_features, drug_feature_store=drug_features)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.error(
            f"Checkpoint path not found: '{checkpoint_path}'. "
            "Please train a model first using 'python main.py --mode train' or specify a valid checkpoint path using '--checkpoint'."
        )
        return
        
    model = CancerCombo(m_config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CancerCombo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cancercombo_best.ckpt")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--scenario", type=int, default=1, help="Split scenario (1, 2, or 3)")
    args = parser.parse_args()
    
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        if os.path.exists("checkpoints"):
            files = [f for f in os.listdir("checkpoints") if f.endswith(".ckpt")]
            if files:
                ckpt_path = os.path.join("checkpoints", files[0])
    run_evaluation(ckpt_path, config_path=args.config, scenario=args.scenario)
