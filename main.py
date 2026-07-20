import argparse
import torch
import sys
from train import run_training
from evaluate import run_evaluation
from predictor import SynergyPredictor
from config import load_config
from helpers import generate_mock_data
from logger import setup_logger
import os
import numpy as np

def run_prediction_cli(checkpoint_path: str):
    logger = setup_logger("CancerCombo Predict CLI")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
        
    m_config, _ = load_config("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = SynergyPredictor(checkpoint_path, m_config, device=device)
    
    # Run CLI test case
    smiles_a = "CC(=O)NC1=CC=C(C=C1)O"
    smiles_b = "CC1=CC(=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    
    cell_line_gene_expr = np.random.randn(20000).astype(np.float32)
    doses_a = np.array([0.0, 0.1, 1.0, 10.0], dtype=np.float32)
    doses_b = np.array([0.0, 0.2, 2.0, 20.0], dtype=np.float32)
    
    logger.info("Executing CLI sample prediction...")
    res = predictor.predict(
        smiles_a=smiles_a,
        smiles_b=smiles_b,
        cell_line_gene_expr=cell_line_gene_expr,
        doses_a=doses_a,
        doses_b=doses_b
    )
    
    logger.info("Prediction successful.")
    logger.info("Estimated parameters:")
    for p, val in res["pharmacological_parameters"].items():
        logger.info(f"  {p}: {val:.6f}")
        
    logger.info("Response surface grid viability:")
    logger.info("\n" + str(res["predicted_viability_matrix"]))

def main():
    parser = argparse.ArgumentParser(description="CancerCombo CLI Management Tool")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "predict"],
        help="CancerCombo execution pipeline mode (default: train)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/cancercombo_best.ckpt",
        help="Path to model checkpoint (.ckpt or .pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        run_training(args.config)
    elif args.mode == "evaluate":
        run_evaluation(args.checkpoint, args.config)
    elif args.mode == "predict":
        run_prediction_cli(args.checkpoint)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
