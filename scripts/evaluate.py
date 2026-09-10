"""Evaluation script for CancerCombo model with unseen-drug scenario breakdowns."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cancer_combo_brics.config import ExperimentConfig
from cancer_combo_brics.utils import load_checkpoint
from cancer_combo_brics.data.dataset import CancerComboDataset, ComboBatchCollator
from cancer_combo_brics.data.preprocessing import CellExpressionPreprocessor, load_cell_expression_data
from cancer_combo_brics.chemistry.cache import FunctionalGroupCache
from cancer_combo_brics.model import CancerComboBRICS
from cancer_combo_brics.losses import SurfaceRegressionLoss
from cancer_combo_brics.metrics import evaluate_predictions_grouped


def main():
    parser = argparse.ArgumentParser(description="Evaluate CancerCombo model checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="results/evaluation_metrics.json", help="Path to save output JSON")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if os.path.exists(args.config) else ExperimentConfig()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n[Evaluation] Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    model = CancerComboBRICS(config=cfg.model).to(device)
    ckpt = load_checkpoint(args.checkpoint, model=model, map_location=str(device))
    print(f"Loaded checkpoint from {args.checkpoint} (Epoch: {ckpt.get('epoch', 'N/A')})")

    comb_candidates = [
        cfg.data.combination_file,
        cfg.data.combination_file.replace("bricks2/", "") if cfg.data.combination_file and cfg.data.combination_file.startswith("bricks2/") else None,
        os.path.join(cfg.data.root, "scenario3_drug_level.csv"),
        "./data/scenario3_drug_level.csv",
    ]
    comb_file = None
    for cand in comb_candidates:
        if cand and os.path.exists(cand):
            comb_file = cand
            break

    if not comb_file:
        raise FileNotFoundError(f"Combinations file not found. Tried paths: {[c for c in comb_candidates if c]}. Check dataset path.")

    df = pd.read_csv(comb_file) if comb_file.endswith(".csv") else pd.read_parquet(comb_file)

    if "split" in df.columns:
        s_col = df["split"].astype(str)
        if set(s_col.unique()).issubset({"1", "2", "3"}):
            test_df = df[s_col == "3"].reset_index(drop=True)
        else:
            test_df = df[s_col.str.lower().isin(["test", "3"])].reset_index(drop=True)
    else:
        test_df = df
    print(f"Evaluating on {len(test_df)} test combination samples...")

    cell_candidates = [
        cfg.data.cell_expression_file,
        cfg.data.cell_expression_file.replace("bricks2/", "") if cfg.data.cell_expression_file and cfg.data.cell_expression_file.startswith("bricks2/") else None,
        os.path.join(cfg.data.root, "cell_line_gene_expr.csv"),
        "./data/cell_line_gene_expr.csv",
    ]
    cell_file = None
    for cand in cell_candidates:
        if cand and os.path.exists(cand):
            cell_file = cand
            break

    if not cell_file:
        raise FileNotFoundError(f"Cell expressions file not found. Tried paths: {[c for c in cell_candidates if c]}. Check dataset path.")

    known_cells = test_df[cfg.data.cell_id_col].dropna().unique().tolist() if cfg.data.cell_id_col in test_df.columns else None
    raw_c_matrix, c_names = load_cell_expression_data(cell_file, known_cell_names=known_cells)

    if os.path.exists(cfg.data.cell_preprocessor_file):
        try:
            preprocessor = CellExpressionPreprocessor.load(cfg.data.cell_preprocessor_file)
            if preprocessor.expected_dim != raw_c_matrix.shape[1]:
                print(f"[WARNING] Saved preprocessor expected_dim ({preprocessor.expected_dim}) != cell feature dim ({raw_c_matrix.shape[1]}). Re-fitting preprocessor.")
                preprocessor = CellExpressionPreprocessor(expected_dim=raw_c_matrix.shape[1]).fit(raw_c_matrix)
        except Exception as e:
            print(f"[WARNING] Failed to load preprocessor ({e}). Re-fitting preprocessor.")
            preprocessor = CellExpressionPreprocessor(expected_dim=raw_c_matrix.shape[1]).fit(raw_c_matrix)
    else:
        preprocessor = CellExpressionPreprocessor(expected_dim=raw_c_matrix.shape[1]).fit(raw_c_matrix)

    norm_c_matrix = preprocessor.transform(raw_c_matrix)
    cell_expr_dict = {name: norm_c_matrix[i] for i, name in enumerate(c_names)}


    fg_cache = FunctionalGroupCache(db_path=cfg.data.fg_cache_file, radius=cfg.data.fg_radius)
    test_dataset = CancerComboDataset(
        test_df, cell_expr_dict, fg_cache,
        smiles_col_a=cfg.data.smiles_col_a, smiles_col_b=cfg.data.smiles_col_b,
        cell_id_col=cfg.data.cell_id_col, dose_col_a=cfg.data.dose_col_a,
        dose_col_b=cfg.data.dose_col_b, viability_col=cfg.data.viability_col,
        drug_id_col_a=cfg.data.drug_id_col_a, drug_id_col_b=cfg.data.drug_id_col_b,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=ComboBatchCollator(max_fragments=cfg.data.max_fragments),
    )


    model.eval()
    records = []
    with torch.no_grad():
        for batch in test_loader:
            cell_expr = batch["cell_expr"].to(device)
            frags_A = batch["fragments_A"]
            mask_A = batch["mask_A"].to(device)
            frags_B = batch["fragments_B"]
            mask_B = batch["mask_B"].to(device)
            doses_A = batch["doses_A"].to(device)
            doses_B = batch["doses_B"].to(device)
            y_true = batch["viability_matrix"].to(device)

            y_pred, _ = model(
                cell_expr=cell_expr,
                fragments_A=frags_A,
                mask_A=mask_A,
                fragments_B=frags_B,
                mask_B=mask_B,
                doses_A=doses_A,
                doses_B=doses_B,
            )

            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            scenarios_np = batch["scenarios"].numpy()
            cells = batch["cell_lines"]
            pairs = batch["drug_pairs"]

            for i in range(len(cells)):
                records.append({
                    "y_true": y_true_np[i],
                    "y_pred": y_pred_np[i],
                    "scenario": int(scenarios_np[i]),
                    "cell_line": cells[i],
                    "drug_pair": pairs[i],
                })

    metrics = evaluate_predictions_grouped(records)

    print("\n================ EVALUATION METRICS REPORT ================")
    print(f"Total Test Samples: {metrics.get('total_samples', 0)}")
    overall = metrics.get("overall", {})
    print(f"Overall RMSE:       {overall.get('rmse', 0.0):.4f}")
    print(f"Overall MAE:        {overall.get('mae', 0.0):.4f}")
    print(f"Overall R^2:        {overall.get('r2', 0.0):.4f}")
    print(f"Overall Pearson:    {overall.get('pearson', 0.0):.4f}")
    print(f"Overall Spearman:   {overall.get('spearman', 0.0):.4f}")

    print("\n--- PERFORMANCE BY UNSEEN-DRUG SCENARIO ---")
    scenarios = metrics.get("scenarios", {})
    for scen_name, s_m in scenarios.items():
        desc = {
            "scenario_1": "Scenario 1 (Both drugs seen)",
            "scenario_2": "Scenario 2 (One drug seen + one unseen)",
            "scenario_3": "Scenario 3 (Both drugs UNSEEN - Primary Goal)",
        }.get(scen_name, scen_name)
        print(f"[{desc}] (N={s_m.get('sample_count', 0)}):")
        print(f"   RMSE: {s_m.get('rmse', 0.0):.4f} | R^2: {s_m.get('r2', 0.0):.4f} | Pearson: {s_m.get('pearson', 0.0):.4f} | MAE: {s_m.get('mae', 0.0):.4f}")
    print("===========================================================\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
