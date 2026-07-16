import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Dict

def calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Calculate statistical validation metrics over flattening viability arrays.

    Args:
        y_pred: Flattened predicted viability array.
        y_true: Flattened experimental viability array.

    Returns:
        Dict[str, float]: Dictionary containing Pearson, Spearman, MSE, and RMSE scores.
    """
    pred_flat = y_pred.flatten()
    true_flat = y_true.flatten()
    
    mse_val = float(np.mean((pred_flat - true_flat) ** 2))
    rmse_val = float(np.sqrt(mse_val))
    
    try:
        pearson_val, _ = pearsonr(pred_flat, true_flat)
        if np.isnan(pearson_val):
            pearson_val = 0.0
    except Exception:
        pearson_val = 0.0
        
    try:
        spearman_val, _ = spearmanr(pred_flat, true_flat)
        if np.isnan(spearman_val):
            spearman_val = 0.0
    except Exception:
        spearman_val = 0.0
        
    return {
        "mse": mse_val,
        "rmse": rmse_val,
        "pearson": pearson_val,
        "spearman": spearman_val
    }
