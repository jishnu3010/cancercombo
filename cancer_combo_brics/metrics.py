"""Comprehensive evaluation metrics for dose-response surfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_surface_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute core regression metrics on flattened viability surfaces.

    Args:
        y_true: Ground truth array of shape (N,) or (B, D_A, D_B).
        y_pred: Predicted array of shape (N,) or (B, D_A, D_B).
        prefix: Optional prefix for metric dictionary keys.

    Returns:
        Dict with MSE, RMSE, MAE, R2, Pearson, Spearman.
    """
    yt = np.asarray(y_true).flatten()
    yp = np.asarray(y_pred).flatten()

    # Filter out NaNs if present
    valid_mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(valid_mask):
        return {
            f"{prefix}mse": float("nan"),
            f"{prefix}rmse": float("nan"),
            f"{prefix}mae": float("nan"),
            f"{prefix}r2": float("nan"),
            f"{prefix}pearson": float("nan"),
            f"{prefix}spearman": float("nan"),
        }

    yt = yt[valid_mask]
    yp = yp[valid_mask]

    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(yt, yp))

    # R^2 score
    try:
        r2 = float(r2_score(yt, yp))
    except Exception:
        r2 = float("nan")

    # Pearson correlation
    try:
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8 and len(yt) > 1:
            p_corr, _ = pearsonr(yt, yp)
            pearson = float(p_corr)
        else:
            pearson = 0.0
    except Exception:
        pearson = float("nan")

    # Spearman correlation
    try:
        if len(yt) > 1:
            s_corr, _ = spearmanr(yt, yp)
            spearman = float(s_corr)
        else:
            spearman = 0.0
    except Exception:
        spearman = float("nan")

    return {
        f"{prefix}mse": mse,
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}r2": r2,
        f"{prefix}pearson": pearson,
        f"{prefix}spearman": spearman,
    }


def evaluate_predictions_grouped(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluates metrics grouped by overall, scenario (1, 2, 3), cell line, and drug pair.

    Args:
        records: List of sample dictionaries containing:
                 'y_true' (np.ndarray or list),
                 'y_pred' (np.ndarray or list),
                 'scenario' (int: 1, 2, or 3),
                 'cell_line' (str),
                 'drug_pair' (str)

    Returns:
        Structured evaluation dictionary.
    """
    if not records:
        return {}

    all_yt = []
    all_yp = []

    # Buckets
    scenario_buckets: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {
        1: ([], []),
        2: ([], []),
        3: ([], []),
    }
    cell_buckets: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    pair_buckets: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}

    for r in records:
        yt = np.asarray(r["y_true"]).flatten()
        yp = np.asarray(r["y_pred"]).flatten()

        all_yt.append(yt)
        all_yp.append(yp)

        scen = r.get("scenario", 1)
        if scen in scenario_buckets:
            scenario_buckets[scen][0].append(yt)
            scenario_buckets[scen][1].append(yp)

        cell = r.get("cell_line", "unknown")
        if cell not in cell_buckets:
            cell_buckets[cell] = ([], [])
        cell_buckets[cell][0].append(yt)
        cell_buckets[cell][1].append(yp)

        pair = r.get("drug_pair", "unknown")
        if pair not in pair_buckets:
            pair_buckets[pair] = ([], [])
        pair_buckets[pair][0].append(yt)
        pair_buckets[pair][1].append(yp)

    # 1. Overall
    overall_yt = np.concatenate(all_yt)
    overall_yp = np.concatenate(all_yp)
    results: Dict[str, Any] = {
        "overall": compute_surface_metrics(overall_yt, overall_yp),
        "total_samples": len(records),
    }

    # 2. By Scenario
    results["scenarios"] = {}
    for scen, (yt_list, yp_list) in scenario_buckets.items():
        if yt_list:
            scen_yt = np.concatenate(yt_list)
            scen_yp = np.concatenate(yp_list)
            scen_metrics = compute_surface_metrics(scen_yt, scen_yp)
            scen_metrics["sample_count"] = len(yt_list)
            results["scenarios"][f"scenario_{scen}"] = scen_metrics

    # 3. By Cell Line (summarized)
    cell_rmses = []
    for cell, (yt_list, yp_list) in cell_buckets.items():
        if yt_list:
            c_yt = np.concatenate(yt_list)
            c_yp = np.concatenate(yp_list)
            c_m = compute_surface_metrics(c_yt, c_yp)
            if np.isfinite(c_m["rmse"]):
                cell_rmses.append(c_m["rmse"])

    results["cell_lines"] = {
        "num_unique_cells": len(cell_buckets),
        "mean_cell_rmse": float(np.mean(cell_rmses)) if cell_rmses else float("nan"),
    }

    # 4. By Drug Pair (summarized)
    pair_rmses = []
    for pair, (yt_list, yp_list) in pair_buckets.items():
        if yt_list:
            p_yt = np.concatenate(yt_list)
            p_yp = np.concatenate(yp_list)
            p_m = compute_surface_metrics(p_yt, p_yp)
            if np.isfinite(p_m["rmse"]):
                pair_rmses.append(p_m["rmse"])

    results["drug_pairs"] = {
        "num_unique_pairs": len(pair_buckets),
        "mean_pair_rmse": float(np.mean(pair_rmses)) if pair_rmses else float("nan"),
    }

    return results
