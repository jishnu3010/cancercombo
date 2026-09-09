"""Cell-line expression preprocessing with train-only statistics calculation."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Union
import numpy as np


class CellExpressionPreprocessor:
    """Preprocesses 976-dimensional landmark gene expression profiles.

    Rule: Imputation and standardization statistics are computed EXCLUSIVELY on
    the training split and reused for validation, test, and inference.
    """

    def __init__(
        self,
        expected_dim: int = 976,
        eps: float = 1e-6,
    ):
        self.expected_dim = expected_dim
        self.eps = eps
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.impute_values: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, train_expressions: np.ndarray) -> "CellExpressionPreprocessor":
        """Compute mean, standard deviation, and imputation values from training data only.

        Args:
            train_expressions: 2D array of shape (N_train, 976).
        """
        assert train_expressions.ndim == 2, "Expected 2D array (samples, genes)"
        assert train_expressions.shape[1] == self.expected_dim, (
            f"Expected {self.expected_dim} genes, but got {train_expressions.shape[1]}"
        )

        # Imputation values (median per gene)
        self.impute_values = np.nanmedian(train_expressions, axis=0)
        # If any entire column is NaN, fill with 0
        self.impute_values = np.nan_to_num(self.impute_values, nan=0.0)

        # Impute missing values for computing mean and std
        imputed = np.where(np.isnan(train_expressions), self.impute_values, train_expressions)

        self.means = np.mean(imputed, axis=0)
        self.stds = np.std(imputed, axis=0)
        # Prevent division by zero
        self.stds = np.where(self.stds < self.eps, 1.0, self.stds)

        self.is_fitted = True
        return self

    def transform(self, expressions: np.ndarray) -> np.ndarray:
        """Apply train-derived imputation and standardization.

        Args:
            expressions: 2D array of shape (N, 976) or 1D array of shape (976,).

        Returns:
            Normalized 2D array of shape (N, 976).
        """
        if not self.is_fitted:
            raise RuntimeError("CellExpressionPreprocessor must be fitted on training data first.")

        single_sample = expressions.ndim == 1
        x = np.atleast_2d(expressions).copy()
        assert x.shape[1] == self.expected_dim, (
            f"Expected {self.expected_dim} features, but got {x.shape[1]}"
        )

        # 1. Impute NaNs using train imputation values
        nan_mask = np.isnan(x)
        if np.any(nan_mask):
            x = np.where(nan_mask, self.impute_values, x)

        # 2. Standardize using train mean and std
        x_norm = (x - self.means) / self.stds

        if single_sample:
            return x_norm[0]
        return x_norm

    def fit_transform(self, train_expressions: np.ndarray) -> np.ndarray:
        """Fit on train and transform."""
        self.fit(train_expressions)
        return self.transform(train_expressions)

    def save(self, filepath: str) -> None:
        """Save fitted statistics to disk (.npz format)."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            means=self.means,
            stds=self.stds,
            impute_values=self.impute_values,
            expected_dim=self.expected_dim,
        )

    @classmethod
    def load(cls, filepath: str) -> "CellExpressionPreprocessor":
        """Load fitted statistics from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor stats not found at: {filepath}")

        data = np.load(filepath)
        preprocessor = cls(expected_dim=int(data["expected_dim"]))
        preprocessor.means = data["means"]
        preprocessor.stds = data["stds"]
        preprocessor.impute_values = data["impute_values"]
        preprocessor.is_fitted = True
        return preprocessor


def load_cell_expression_data(
    cell_file: str,
    known_cell_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Load cell line expression matrix and cell line names from CSV or NPZ.

    Automatically handles CSV orientation (whether cell lines are rows or columns).

    Args:
        cell_file: Path to .npz or .csv file.
        known_cell_names: Optional list of cell line names from combination table.

    Returns:
        Tuple of (raw_c_matrix of shape (N_cells, N_genes), list of cell_line_names of length N_cells).
    """
    if cell_file.endswith(".npz"):
        c_data = np.load(cell_file)
        raw_c_matrix = c_data["expressions"].astype(np.float32)
        c_names = list(c_data["cell_lines"])
        return raw_c_matrix, [str(x) for x in c_names]

    import pandas as pd

    c_df = pd.read_csv(cell_file, index_col=0)

    should_transpose = False
    if c_df.index.name and "gene" in str(c_df.index.name).lower():
        should_transpose = True
    elif known_cell_names:
        known_set = set(known_cell_names)
        cols_match = len(known_set.intersection(set(c_df.columns)))
        idx_match = len(known_set.intersection(set(c_df.index)))
        if cols_match > idx_match:
            should_transpose = True
    elif c_df.shape[0] > c_df.shape[1] and c_df.shape[1] < 200:
        should_transpose = True

    if should_transpose:
        c_df = c_df.T

    raw_c_matrix = c_df.values.astype(np.float32)
    c_names = [str(x) for x in c_df.index]
    return raw_c_matrix, c_names

