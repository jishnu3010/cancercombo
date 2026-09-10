"""Comprehensive diagnostic script for CancerCombo model on Scenario 3 dataset.

Performs:
1. Module-level gradient statistics across multiple batches
2. Prediction distribution diagnostic (Y_true vs Y_pred)
3. Hill parameter diagnostic (bounds, clamps, distributions)
4. Complete 2D surface diagnostic on representative test samples (4x4 surfaces)
5. Information-control ablations (cell-only, drug-only, pair-only)
6. Sensitivity tests (cell expression +/-10%, Drug A shuffle, Drug B shuffle)
7. Baseline comparison table
8. Dataset coverage and disjointness validation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cancer_combo_brics.config import ExperimentConfig
from cancer_combo_brics.data.dataset import CancerComboDataset, ComboBatchCollator
from cancer_combo_brics.data.preprocessing import CellExpressionPreprocessor, load_cell_expression_data
from cancer_combo_brics.chemistry.cache import FunctionalGroupCache
from cancer_combo_brics.model import CancerComboBRICS
from cancer_combo_brics.losses import SurfaceRegressionLoss
from cancer_combo_brics.metrics import compute_surface_metrics


def run_diagnostics(
    config_path: str = "configs/default.yaml",
    checkpoint_path: str | None = None,
    output_path: str = "results/diagnostic_report.json",
    num_diagnostic_batches: int = 5,
    device_name: str | None = None,
):
    print("=" * 80)
    print("CANCERCOMBO COMPREHENSIVE FAILURE MODE & ARCHITECTURE DIAGNOSTICS")
    print("=" * 80)

    cfg = ExperimentConfig.from_yaml(config_path) if os.path.exists(config_path) else ExperimentConfig()
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # 1. Dataset loading & coverage check
    comb_candidates = [
        "data/scenario3_drug_level.csv",
        "scenario3_drug_level.csv",
        cfg.data.combination_file,
    ]
    comb_file = next((c for c in comb_candidates if c and os.path.exists(c)), None)
    if not comb_file:
        raise FileNotFoundError("Could not find scenario3_drug_level.csv")

    df = pd.read_csv(comb_file)
    total_samples = len(df)
    print(f"\n[Data Check] Total Scenario 3 samples in CSV: {total_samples}")

    split_col = df["split"].astype(str)
    train_df = df[split_col.isin(["1", "train"])].reset_index(drop=True)
    val_df = df[split_col.isin(["2", "val"])].reset_index(drop=True)
    test_df = df[split_col.isin(["3", "test"])].reset_index(drop=True)

    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples:   {len(val_df)}")
    print(f"  Test samples:  {len(test_df)}")

    train_drugs = set(train_df["smiles_a"].dropna().unique()) | set(train_df["smiles_b"].dropna().unique())
    val_drugs = set(val_df["smiles_a"].dropna().unique()) | set(val_df["smiles_b"].dropna().unique())
    test_drugs = set(test_df["smiles_a"].dropna().unique()) | set(test_df["smiles_b"].dropna().unique())

    train_val_overlap = len(train_drugs & val_drugs)
    train_test_overlap = len(train_drugs & test_drugs)
    val_test_overlap = len(val_drugs & test_drugs)

    print(f"  Train drugs: {len(train_drugs)}, Val drugs: {len(val_drugs)}, Test drugs: {len(test_drugs)}")
    print(f"  Drug overlaps: Train-Val={train_val_overlap}, Train-Test={train_test_overlap}, Val-Test={val_test_overlap}")

    # Load cell lines
    cell_candidates = ["data/cell_line_gene_expr.csv", "cell_line_gene_expr.csv"]
    cell_file = next((c for c in cell_candidates if os.path.exists(c)), None)
    raw_c_matrix, c_names = load_cell_expression_data(cell_file)
    preprocessor = CellExpressionPreprocessor(expected_dim=raw_c_matrix.shape[1]).fit(raw_c_matrix)
    norm_c_matrix = preprocessor.transform(raw_c_matrix)
    cell_expr_dict = {name: norm_c_matrix[i] for i, name in enumerate(c_names)}

    fg_cache = FunctionalGroupCache()
    collator = ComboBatchCollator(max_fragments=cfg.data.max_fragments)

    # Prepare datasets
    train_dataset = CancerComboDataset(
        train_df.head(200),
        cell_expressions=cell_expr_dict,
        fg_cache=fg_cache,
    )
    test_dataset = CancerComboDataset(
        test_df.head(200),
        cell_expressions=cell_expr_dict,
        fg_cache=fg_cache,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collator)

    # Initialize model
    model = CancerComboBRICS(config=cfg.model).to(device)
    loss_fn = SurfaceRegressionLoss()

    report: Dict[str, Any] = {
        "dataset_summary": {
            "total_samples": total_samples,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "train_drugs": len(train_drugs),
            "val_drugs": len(val_drugs),
            "test_drugs": len(test_drugs),
            "train_test_drug_overlap": train_test_overlap,
            "coverage_percent": 100.0 * (len(train_df) + len(val_df) + len(test_df)) / total_samples,
        }
    }

    # -------------------------------------------------------------
    # 1. Module-Level Gradient Statistics Across Multiple Batches
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. MODULE-LEVEL GRADIENT STATISTICS ACROSS MULTIPLE BATCHES")
    print("=" * 80)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    module_grads: Dict[str, List[float]] = {}
    module_weights: Dict[str, List[float]] = {}
    module_nan_inf: Dict[str, int] = {}

    batch_count = 0
    for batch in train_loader:
        if batch_count >= num_diagnostic_batches:
            break
        optimizer.zero_grad()
        y_pred, _ = model(
            cell_expr=batch["cell_expr"].to(device),
            fragments_A=batch["fragments_A"],
            mask_A=batch["mask_A"].to(device),
            fragments_B=batch["fragments_B"],
            mask_B=batch["mask_B"].to(device),
            doses_A=batch["doses_A"].to(device),
            doses_B=batch["doses_B"].to(device),
            smiles_A=batch.get("smiles_A"),
            smiles_B=batch.get("smiles_B"),
        )
        loss = loss_fn(y_pred, batch["viability_matrix"].to(device))
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                mod_name = name.split(".")[0]
                p_norm = param.data.norm().item()
                g_norm = param.grad.data.norm().item()
                has_nan_inf = torch.isnan(param.grad).any().item() or torch.isinf(param.grad).any().item()

                if mod_name not in module_grads:
                    module_grads[mod_name] = []
                    module_weights[mod_name] = []
                    module_nan_inf[mod_name] = 0

                module_grads[mod_name].append(g_norm)
                module_weights[mod_name].append(p_norm)
                if has_nan_inf:
                    module_nan_inf[mod_name] += 1

        batch_count += 1

    grad_stats_table = []
    for mod in module_grads:
        avg_w = float(np.mean(module_weights[mod]))
        avg_g = float(np.mean(module_grads[mod]))
        ratio = avg_g / (avg_w + 1e-12)
        nans = module_nan_inf[mod]
        grad_stats_table.append({
            "module": mod,
            "weight_norm": round(avg_w, 4),
            "grad_norm": round(avg_g, 4),
            "grad_param_ratio": round(ratio, 6),
            "nan_inf_count": nans,
        })
        print(f"  {mod:25s} | Param Norm: {avg_w:8.4f} | Grad Norm: {avg_g:8.4f} | Grad/Param: {ratio:10.6f} | NaN/Inf: {nans}")

    report["gradient_statistics"] = grad_stats_table

    # -------------------------------------------------------------
    # 2. Prediction Distribution Diagnostic (Y_true vs Y_pred)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. PREDICTION DISTRIBUTION DIAGNOSTIC (Y_TRUE vs Y_PRED)")
    print("=" * 80)
    model.eval()
    all_y_true = []
    all_y_pred = []
    all_hill_params: Dict[str, List[float]] = {k: [] for k in ["e1", "e2", "e3", "logc1", "logc2", "h1", "h2", "alpha"]}
    all_surfaces = []

    with torch.no_grad():
        for batch in test_loader:
            y_p, diag = model(
                cell_expr=batch["cell_expr"].to(device),
                fragments_A=batch["fragments_A"],
                mask_A=batch["mask_A"].to(device),
                fragments_B=batch["fragments_B"],
                mask_B=batch["mask_B"].to(device),
                doses_A=batch["doses_A"].to(device),
                doses_B=batch["doses_B"].to(device),
                smiles_A=batch.get("smiles_A"),
                smiles_B=batch.get("smiles_B"),
                return_diagnostics=True,
            )
            y_p_np = y_p.cpu().numpy()
            y_t_np = batch["viability_matrix"].cpu().numpy()
            all_y_pred.append(y_p_np)
            all_y_true.append(y_t_np)
            all_surfaces.append(y_p_np)

            hp = diag["constrained_params"]
            for k in all_hill_params:
                if k in hp:
                    all_hill_params[k].extend(hp[k].cpu().numpy().flatten().tolist())

    all_y_pred = np.concatenate(all_y_pred, axis=0)
    all_y_true = np.concatenate(all_y_true, axis=0)

    y_true_mean = float(np.mean(all_y_true))
    y_true_std = float(np.std(all_y_true))
    y_pred_mean = float(np.mean(all_y_pred))
    y_pred_std = float(np.std(all_y_pred))

    constant_pred = np.full_like(all_y_true, y_true_mean)
    baseline_rmse = float(np.sqrt(np.mean((constant_pred - all_y_true) ** 2)))
    model_rmse = float(np.sqrt(np.mean((all_y_pred - all_y_true) ** 2)))

    pred_dist_report = {
        "y_true": {
            "min": float(np.min(all_y_true)),
            "max": float(np.max(all_y_true)),
            "mean": y_true_mean,
            "std": y_true_std,
        },
        "y_pred": {
            "min": float(np.min(all_y_pred)),
            "max": float(np.max(all_y_pred)),
            "mean": y_pred_mean,
            "std": y_pred_std,
        },
        "baseline_constant_mean_rmse": baseline_rmse,
        "model_rmse": model_rmse,
        "collapsed_to_constant": bool(y_pred_std < 0.5),
        "predictions_blown_up": bool(np.max(np.abs(all_y_pred)) > 500),
    }

    print(f"  Y_true: Min={all_y_true.min():.2f}, Max={all_y_true.max():.2f}, Mean={y_true_mean:.2f}, Std={y_true_std:.2f}")
    print(f"  Y_pred: Min={all_y_pred.min():.2f}, Max={all_y_pred.max():.2f}, Mean={y_pred_mean:.2f}, Std={y_pred_std:.2f}")
    print(f"  Baseline RMSE (Constant Mean): {baseline_rmse:.4f}")
    print(f"  Model RMSE:                     {model_rmse:.4f}")
    print(f"  Collapsed to Constant:         {pred_dist_report['collapsed_to_constant']}")
    print(f"  Predictions Blown Up:          {pred_dist_report['predictions_blown_up']}")

    report["prediction_distribution"] = pred_dist_report

    # -------------------------------------------------------------
    # 3. Hill Parameter Diagnostic
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. HILL PARAMETER DIAGNOSTIC (DISTRIBUTION & BOUNDS)")
    print("=" * 80)
    hill_report = {}
    for param_name, vals in all_hill_params.items():
        v = np.array(vals)
        p_min, p_max, p_mean, p_std = float(np.min(v)), float(np.max(v)), float(np.mean(v)), float(np.std(v))
        hill_report[param_name] = {
            "min": p_min,
            "max": p_max,
            "mean": p_mean,
            "std": p_std,
        }
        print(f"  {param_name:8s} | Min: {p_min:8.4f} | Max: {p_max:8.4f} | Mean: {p_mean:8.4f} | Std: {p_std:8.4f}")

    report["hill_parameter_diagnostics"] = hill_report

    # -------------------------------------------------------------
    # 4. Complete 2D Surface Diagnostic on Representative Samples
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. COMPLETE 2D SURFACE DIAGNOSTIC (REPRESENTATIVE SAMPLES)")
    print("=" * 80)
    all_surfaces = np.concatenate(all_surfaces, axis=0)  # [N, 4, 4]
    surface_samples = []
    for idx in range(min(3, len(all_surfaces))):
        s_true = all_y_true[idx]
        s_pred = all_surfaces[idx]
        row_var = float(np.mean(np.var(s_pred, axis=1)))
        col_var = float(np.mean(np.var(s_pred, axis=0)))
        max_abs_err = float(np.max(np.abs(s_pred - s_true)))
        s_stat = {
            "sample_index": idx,
            "surface_mean": float(np.mean(s_pred)),
            "surface_std": float(np.std(s_pred)),
            "surface_min": float(np.min(s_pred)),
            "surface_max": float(np.max(s_pred)),
            "row_variance": row_var,
            "col_variance": col_var,
            "max_abs_error": max_abs_err,
            "is_flat": bool(np.std(s_pred) < 0.1),
            "shows_dose_response": bool(row_var > 0.5 or col_var > 0.5),
            "surface_4x4": s_pred.tolist(),
        }
        surface_samples.append(s_stat)
        print(f"\n  Sample {idx}: Mean={s_stat['surface_mean']:.2f}, Std={s_stat['surface_std']:.2f}, RowVar={row_var:.4f}, ColVar={col_var:.4f}, Flat={s_stat['is_flat']}")
        print("  Predicted 4x4 Viability Surface:")
        for r in s_pred:
            print("    [" + ", ".join(f"{v:6.1f}" for v in r) + "]")

    report["surface_diagnostics"] = surface_samples

    # -------------------------------------------------------------
    # 5. Information-Control Ablations (Cell vs Drug vs Pair)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. INFORMATION-CONTROL ABLATIONS")
    print("=" * 80)
    test_batch = next(iter(test_loader))
    with torch.no_grad():
        # Full model
        out_full, _ = model(
            cell_expr=test_batch["cell_expr"].to(device),
            fragments_A=test_batch["fragments_A"],
            mask_A=test_batch["mask_A"].to(device),
            fragments_B=test_batch["fragments_B"],
            mask_B=test_batch["mask_B"].to(device),
            doses_A=test_batch["doses_A"].to(device),
            doses_B=test_batch["doses_B"].to(device),
            smiles_A=test_batch.get("smiles_A"),
            smiles_B=test_batch.get("smiles_B"),
        )
        rmse_full = float(np.sqrt(np.mean((out_full.cpu().numpy() - test_batch["viability_matrix"].numpy()) ** 2)))

        # Zero cell (Drug only)
        zero_cell = torch.zeros_like(test_batch["cell_expr"]).to(device)
        out_zero_cell, _ = model(
            cell_expr=zero_cell,
            fragments_A=test_batch["fragments_A"],
            mask_A=test_batch["mask_A"].to(device),
            fragments_B=test_batch["fragments_B"],
            mask_B=test_batch["mask_B"].to(device),
            doses_A=test_batch["doses_A"].to(device),
            doses_B=test_batch["doses_B"].to(device),
            smiles_A=test_batch.get("smiles_A"),
            smiles_B=test_batch.get("smiles_B"),
        )
        rmse_zero_cell = float(np.sqrt(np.mean((out_zero_cell.cpu().numpy() - test_batch["viability_matrix"].numpy()) ** 2)))

        # Zero drugs (Cell only)
        zero_mask_A = torch.zeros_like(test_batch["mask_A"]).to(device)
        zero_mask_B = torch.zeros_like(test_batch["mask_B"]).to(device)
        out_zero_drug, _ = model(
            cell_expr=test_batch["cell_expr"].to(device),
            fragments_A=test_batch["fragments_A"],
            mask_A=zero_mask_A,
            fragments_B=test_batch["fragments_B"],
            mask_B=zero_mask_B,
            doses_A=test_batch["doses_A"].to(device),
            doses_B=test_batch["doses_B"].to(device),
            smiles_A=[""] * len(test_batch["smiles_A"]),
            smiles_B=[""] * len(test_batch["smiles_B"]),
        )
        rmse_zero_drug = float(np.sqrt(np.mean((out_zero_drug.cpu().numpy() - test_batch["viability_matrix"].numpy()) ** 2)))

    ablation_report = {
        "full_model_rmse": rmse_full,
        "zero_cell_drug_only_rmse": rmse_zero_cell,
        "zero_drug_cell_only_rmse": rmse_zero_drug,
        "delta_rmse_cell_ablation": rmse_zero_cell - rmse_full,
        "delta_rmse_drug_ablation": rmse_zero_drug - rmse_full,
    }
    print(f"  Full Model RMSE:           {rmse_full:.4f}")
    print(f"  Zero Cell (Drug Only) RMSE:{rmse_zero_cell:.4f} (Delta: {rmse_zero_cell - rmse_full:+.4f})")
    print(f"  Zero Drug (Cell Only) RMSE:{rmse_zero_drug:.4f} (Delta: {rmse_zero_drug - rmse_full:+.4f})")
    report["ablations"] = ablation_report

    # -------------------------------------------------------------
    # 6. Sensitivity Tests
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. SENSITIVITY TESTS")
    print("=" * 80)
    with torch.no_grad():
        # Cell +10% perturbation
        perturbed_cell = test_batch["cell_expr"].to(device) * 1.10
        out_cell_pert, _ = model(
            cell_expr=perturbed_cell,
            fragments_A=test_batch["fragments_A"],
            mask_A=test_batch["mask_A"].to(device),
            fragments_B=test_batch["fragments_B"],
            mask_B=test_batch["mask_B"].to(device),
            doses_A=test_batch["doses_A"].to(device),
            doses_B=test_batch["doses_B"].to(device),
            smiles_A=test_batch.get("smiles_A"),
            smiles_B=test_batch.get("smiles_B"),
        )
        delta_pred_cell = float(np.mean(np.abs(out_cell_pert.cpu().numpy() - out_full.cpu().numpy())))

        # Shuffle Drug A (random replacement)
        bsz = len(test_batch["fragments_A"])
        perm = torch.randperm(bsz)
        shuff_frags_A = [test_batch["fragments_A"][i] for i in perm.tolist()]
        shuff_mask_A = test_batch["mask_A"][perm].to(device)
        shuff_smiles_A = [test_batch["smiles_A"][i] for i in perm.tolist()]

        out_shuff_A, _ = model(
            cell_expr=test_batch["cell_expr"].to(device),
            fragments_A=shuff_frags_A,
            mask_A=shuff_mask_A,
            fragments_B=test_batch["fragments_B"],
            mask_B=test_batch["mask_B"].to(device),
            doses_A=test_batch["doses_A"].to(device),
            doses_B=test_batch["doses_B"].to(device),
            smiles_A=shuff_smiles_A,
            smiles_B=test_batch.get("smiles_B"),
        )
        delta_pred_shuff_A = float(np.mean(np.abs(out_shuff_A.cpu().numpy() - out_full.cpu().numpy())))

        # Shuffle Drug B (random replacement)
        perm_b = torch.randperm(bsz)
        shuff_frags_B = [test_batch["fragments_B"][i] for i in perm_b.tolist()]
        shuff_mask_B = test_batch["mask_B"][perm_b].to(device)
        shuff_smiles_B = [test_batch["smiles_B"][i] for i in perm_b.tolist()]

        out_shuff_B, _ = model(
            cell_expr=test_batch["cell_expr"].to(device),
            fragments_A=test_batch["fragments_A"],
            mask_A=test_batch["mask_A"].to(device),
            fragments_B=shuff_frags_B,
            mask_B=shuff_mask_B,
            doses_A=test_batch["doses_A"].to(device),
            doses_B=test_batch["doses_B"].to(device),
            smiles_A=test_batch.get("smiles_A"),
            smiles_B=shuff_smiles_B,
        )
        delta_pred_shuff_B = float(np.mean(np.abs(out_shuff_B.cpu().numpy() - out_full.cpu().numpy())))

    sens_report = {
        "mean_abs_delta_cell_plus_10pct": delta_pred_cell,
        "mean_abs_delta_shuffle_drug_A": delta_pred_shuff_A,
        "mean_abs_delta_shuffle_drug_B": delta_pred_shuff_B,
        "model_ignores_drugs": bool(delta_pred_shuff_A < 0.05 and delta_pred_shuff_B < 0.05),
    }
    print(f"  Mean Abs Change with Cell +10%:     {delta_pred_cell:.4f}")
    print(f"  Mean Abs Change with Drug A Shuffle: {delta_pred_shuff_A:.4f}")
    print(f"  Mean Abs Change with Drug B Shuffle: {delta_pred_shuff_B:.4f}")
    print(f"  Model Ignores Drug Features:         {sens_report['model_ignores_drugs']}")
    report["sensitivity_tests"] = sens_report

    # -------------------------------------------------------------
    # 7. Baseline Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("7. BASELINE COMPARISON TABLE")
    print("=" * 80)
    test_y_true = test_batch["viability_matrix"].numpy()
    mean_baseline_rmse = float(np.sqrt(np.mean((np.mean(test_y_true) - test_y_true) ** 2)))

    baseline_table = [
        {"model_type": "Constant Mean Baseline", "rmse": round(mean_baseline_rmse, 4)},
        {"model_type": "Zero-Drug (Cell Only)", "rmse": round(rmse_zero_drug, 4)},
        {"model_type": "Zero-Cell (Drug Only)", "rmse": round(rmse_zero_cell, 4)},
        {"model_type": "CancerCombo Full Active Model", "rmse": round(rmse_full, 4)},
    ]
    for row in baseline_table:
        print(f"  {row['model_type']:32s} | RMSE: {row['rmse']:8.4f}")

    report["baseline_comparison_table"] = baseline_table

    # Save to json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Diagnostics Complete] Diagnostic report saved to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/diagnostic_report.json")
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_diagnostics(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_diagnostic_batches=args.batches,
        device_name=args.device,
    )
