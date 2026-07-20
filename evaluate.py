import os
import torch
from torch.utils.data import DataLoader
from config import load_config
from dataset import DrugComboDataset
from cancercombo import CancerCombo
from evaluator import ModelEvaluator
from helpers import generate_mock_data
from logger import setup_logger

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
