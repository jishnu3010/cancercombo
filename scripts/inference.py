"""Inference script for predicting complete 2D dose-response surfaces for unseen drug combinations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cancer_combo_brics.config import ExperimentConfig
from cancer_combo_brics.utils import load_checkpoint
from cancer_combo_brics.chemistry.functional_group_fragments import extract_functional_group_fragments
from cancer_combo_brics.chemistry.fragment_utils import pad_fragment_strings
from cancer_combo_brics.data.preprocessing import CellExpressionPreprocessor, load_cell_expression_data
from cancer_combo_brics.model import CancerComboBRICS


def main():
    parser = argparse.ArgumentParser(description="Predict 2D dose-response surface for a drug combination.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--smiles_a", type=str, required=True, help="SMILES string for Drug A")
    parser.add_argument("--smiles_b", type=str, required=True, help="SMILES string for Drug B")
    parser.add_argument("--cell_id", type=str, default=None, help="Cell line identifier")
    parser.add_argument("--cell_expr_path", type=str, default=None, help="Path to cell expression npz/csv")
    parser.add_argument("--doses_a", type=str, default="0.0, 0.01, 0.05, 0.2, 1.0, 3.0, 10.0, 30.0", help="Comma-separated doses for Drug A")
    parser.add_argument("--doses_b", type=str, default="0.0, 0.005, 0.02, 0.1, 0.5, 2.0, 8.0, 25.0", help="Comma-separated doses for Drug B")
    parser.add_argument("--output_json", type=str, default="results/inference_output.json", help="Output path for predictions")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if os.path.exists(args.config) else ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Parse Doses
    d_a = np.array([float(x.strip()) for x in args.doses_a.split(",")], dtype=np.float32)
    d_b = np.array([float(x.strip()) for x in args.doses_b.split(",")], dtype=np.float32)

    # 2. Extract Functional Group Fragments
    frags_a = extract_functional_group_fragments(args.smiles_a)
    frags_b = extract_functional_group_fragments(args.smiles_b)
    print(f"\nDrug A functional group fragments ({len(frags_a)}): {frags_a}")
    print(f"Drug B functional group fragments ({len(frags_b)}): {frags_b}")

    padded_a, mask_a = pad_fragment_strings([frags_a], max_fragments=cfg.data.max_fragments)
    padded_b, mask_b = pad_fragment_strings([frags_b], max_fragments=cfg.data.max_fragments)

    # 3. Load Cell Line Expression
    cell_file = args.cell_expr_path or cfg.data.cell_expression_file or "./data/cell_line_gene_expr.csv"
    if os.path.exists(cell_file):
        raw_c_matrix, c_names = load_cell_expression_data(cell_file, known_cell_names=[args.cell_id] if args.cell_id else None)
        idx = c_names.index(args.cell_id) if args.cell_id in c_names else 0
        raw_expr = raw_c_matrix[idx]
    else:
        raw_expr = np.random.randn(cfg.model.cell_dim).astype(np.float32)

    # Normalize cell expression
    if os.path.exists(cfg.data.cell_preprocessor_file):
        try:
            prep = CellExpressionPreprocessor.load(cfg.data.cell_preprocessor_file)
            if prep.expected_dim == len(raw_expr):
                norm_expr = prep.transform(raw_expr)
            else:
                norm_expr = (raw_expr - np.mean(raw_expr)) / (np.std(raw_expr) + 1e-6)
        except Exception:
            norm_expr = (raw_expr - np.mean(raw_expr)) / (np.std(raw_expr) + 1e-6)
    else:
        norm_expr = (raw_expr - np.mean(raw_expr)) / (np.std(raw_expr) + 1e-6)


    # 4. Load Model
    model = CancerComboBRICS(config=cfg.model).to(device)
    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model=model, map_location=str(device))
        print(f"Loaded weights from {args.checkpoint}")
    else:
        print(f"[NOTE] Checkpoint not found at {args.checkpoint}, using randomly initialized model.")

    model.eval()
    with torch.no_grad():
        cell_t = torch.from_numpy(norm_expr).unsqueeze(0).to(device)
        mask_a_t = mask_a.to(device)
        mask_b_t = mask_b.to(device)
        d_a_t = torch.from_numpy(d_a).unsqueeze(0).to(device)
        d_b_t = torch.from_numpy(d_b).unsqueeze(0).to(device)

        y_pred, diag = model(
            cell_expr=cell_t,
            fragments_A=padded_a,
            mask_A=mask_a_t,
            fragments_B=padded_b,
            mask_B=mask_b_t,
            doses_A=d_a_t,
            doses_B=d_b_t,
            return_diagnostics=True,
        )

    # Extract results
    pred_surface = y_pred[0].cpu().numpy()
    hill_surface = diag["Y_hill"][0].cpu().numpy()
    bias_surface = diag["bias"][0].cpu().numpy()
    params = {k: float(v[0].item()) for k, v in diag["constrained_params"].items()}

    print("\n================ PREDICTED PHARMACOLOGICAL PARAMETERS ================")
    print(f"  e0 (baseline):  {params.get('e0', 100.0):.4f} (fixed, 100.0 = 100% viability)")
    print(f"  e1 (efficacy A): {params.get('e1', 0.0):.4f}")
    print(f"  e2 (efficacy B): {params.get('e2', 0.0):.4f}")
    print(f"  e3 (combined):   {params.get('e3', 0.0):.4f}")
    print(f"  logC1 (potency A): {params.get('logc1', 0.0):.4f} -> C1 = {np.exp(params.get('logc1', 0.0)):.4e}")
    print(f"  logC2 (potency B): {params.get('logc2', 0.0):.4f} -> C2 = {np.exp(params.get('logc2', 0.0)):.4e}")
    print(f"  h1 (Hill slope A): {params.get('h1', 0.0):.4f}")
    print(f"  h2 (Hill slope B): {params.get('h2', 0.0):.4f}")
    print(f"  alpha (synergy):   {params.get('alpha', 0.0):.4f}")

    print("\n================ PREDICTED 2D SURFACE ================")
    print(f"Dose A axis: {d_a}")
    print(f"Dose B axis: {d_b}")
    print(f"Predicted Surface (shape: {pred_surface.shape}, min: {pred_surface.min():.3f}, max: {pred_surface.max():.3f}):")
    print(np.array2string(pred_surface, precision=3, suppress_small=True))
    print("============================================================\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    out_data = {
        "smiles_a": args.smiles_a,
        "smiles_b": args.smiles_b,
        "doses_a": d_a.tolist(),
        "doses_b": d_b.tolist(),
        "pharmacological_parameters": params,
        "predicted_surface": pred_surface.tolist(),
        "hill_surface": hill_surface.tolist(),
        "bias_surface": bias_surface.tolist(),
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
