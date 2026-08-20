"""
Real 976-Gene Cell Line Expression Loader Module for CancerCombo.

Enforces Leakage-Safe Gene Expression Normalization:
    1. Loading real 976-dimensional landmark gene expression vectors from configurable CSV.
    2. Strict feature dimension verification (gene_dim == 976).
    3. Fixed deterministic gene feature ordering.
    4. Training-set fit z-score normalization:
       - fit_normalization() is called EXCLUSIVELY on training-set cell lines.
       - transform() reuses training mean/std unchanged for validation, test, and inference.
       - Refitting on validation or test sets is strictly prohibited.
    5. Loud error handling: raises explicit ValueError if any required cell line is missing.
       NO synthetic vector fallbacks or random hashes!
"""

import os
import csv
from typing import Dict, List, Optional, Tuple, Set, Iterable
import torch
import numpy as np


class CellExpressionLoader:
    """
    Cell Line Expression Loader and Leakage-Safe Normalizer.

    Args:
        csv_path: Path to cell line gene expression CSV file (e.g. data/cell_line_gene_expr.csv).
        gene_dim: Expected gene expression dimension (default: 976).
        normalize: Whether to apply z-score normalization (default: True).
    """

    def __init__(
        self,
        csv_path: str = os.path.join("data", "cell_line_gene_expr.csv"),
        gene_dim: int = 976,
        normalize: bool = True
    ):
        self.csv_path = csv_path
        self.gene_dim = gene_dim
        self.normalize = normalize

        self.raw_cell_expr_dict: Dict[str, torch.Tensor] = {}
        self.gene_names: List[str] = []
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.fitted: bool = False
        self.fitted_cell_ids: Set[str] = set()

        if os.path.exists(csv_path):
            self._load_from_csv()
        else:
            print(f"Warning: Cell expression file '{csv_path}' not found upon initialization.")

    def _load_from_csv(self):
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            if not header or len(header) <= 1:
                raise ValueError(f"Cell expression CSV '{self.csv_path}' is empty or invalid.")

            self.gene_names = [col.strip() for col in header[1:]]
            if len(self.gene_names) != self.gene_dim:
                raise ValueError(
                    f"Feature dimension mismatch in '{self.csv_path}': "
                    f"expected {self.gene_dim} genes, but found {len(self.gene_names)}."
                )

            for line_idx, row in enumerate(reader, start=2):
                if not row or not row[0].strip():
                    continue
                cell_id = str(row[0]).strip()

                try:
                    expr_vals = [float(x) for x in row[1:]]
                except ValueError as e:
                    raise ValueError(
                        f"Failed parsing expression values for cell line '{cell_id}' on line {line_idx} in '{self.csv_path}': {e}"
                    )

                if len(expr_vals) != self.gene_dim:
                    raise ValueError(
                        f"Cell line '{cell_id}' has {len(expr_vals)} features, expected {self.gene_dim}."
                    )

                expr_tensor = torch.tensor(expr_vals, dtype=torch.float32)
                if torch.isnan(expr_tensor).any() or torch.isinf(expr_tensor).any():
                    raise ValueError(f"Cell line '{cell_id}' contains NaN or Inf gene expression values.")

                self.raw_cell_expr_dict[cell_id] = expr_tensor

    def fit_normalization(self, train_cell_ids: Optional[Set[str]] = None, force_refit: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fits z-score normalization parameters on TRAINING SET cell lines ONLY.

        Args:
            train_cell_ids: Set of cell line IDs belonging strictly to the training split.
            force_refit: Whether to force refitting if already fitted (default: False).

        Returns:
            Tuple of (mean_tensor, std_tensor) calculated from training cell lines.
        """
        if not self.normalize or not self.raw_cell_expr_dict:
            return torch.zeros(self.gene_dim), torch.ones(self.gene_dim)

        if self.fitted and not force_refit:
            # Already fitted on training set; preserve existing training statistics unchanged
            return self.mean, self.std

        if train_cell_ids:
            fit_tensors = [self.raw_cell_expr_dict[cid] for cid in train_cell_ids if cid in self.raw_cell_expr_dict]
            self.fitted_cell_ids = set(train_cell_ids)
        else:
            fit_tensors = list(self.raw_cell_expr_dict.values())
            self.fitted_cell_ids = set(self.raw_cell_expr_dict.keys())

        if not fit_tensors:
            raise ValueError("Cannot fit normalization: no valid cell line expression vectors found for training set.")

        stacked = torch.stack(fit_tensors, dim=0)  # (N_train_cells, 976)
        self.mean = torch.mean(stacked, dim=0)
        if stacked.size(0) > 1:
            self.std = torch.std(stacked, dim=0, unbiased=False)
        else:
            self.std = torch.ones_like(self.mean)
        # Avoid division by zero: clamp std to min epsilon 1e-6
        self.std = torch.clamp(self.std, min=1e-6)

        self.fitted = True
        return self.mean, self.std

    def transform(self, expr_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a gene expression tensor using the fitted TRAINING SET mean and std.

        Raises RuntimeError if fit_normalization() has not been called on training set first.
        """
        if not self.normalize:
            return expr_tensor

        if not self.fitted or self.mean is None or self.std is None:
            raise RuntimeError(
                "[LEAKAGE PROTECTION ERROR] CellExpressionLoader must be fitted on training set cell lines before transforming data!"
            )

        norm_tensor = (expr_tensor - self.mean) / self.std

        if torch.isnan(norm_tensor).any() or torch.isinf(norm_tensor).any():
            raise ValueError("Normalized gene expression vector contains NaN or Inf values.")

        return norm_tensor

    def get_cell_expression(self, cell_id: str) -> torch.Tensor:
        """
        Retrieves normalized 976-dimensional gene expression vector for target cell line.
        Uses training-set fitted mean and std. Fails loudly if the required cell line is missing.
        """
        clean_id = str(cell_id).strip()
        if clean_id not in self.raw_cell_expr_dict:
            available_cells = sorted(list(self.raw_cell_expr_dict.keys()))[:5]
            raise ValueError(
                f"\n[CRITICAL ERROR] Missing Required Biological Data!\n"
                f"  - Cell Line ID Requested: '{clean_id}'\n"
                f"  - Expression Source CSV : '{self.csv_path}'\n"
                f"  - Expected Gene Features: {self.gene_dim} (LINCS L1000 Landmark Genes)\n"
                f"  - Cell Lines Loaded     : {len(self.raw_cell_expr_dict)} (e.g. {available_cells}...)\n"
                f"  - Policy                : NO synthetic random vector generation allowed. Training halted.\n"
            )
        return self.transform(self.raw_cell_expr_dict[clean_id])

    def get_all_normalized_expressions(self) -> Dict[str, torch.Tensor]:
        """Returns dictionary of cell_id -> normalized expression tensor for all loaded cell lines."""
        return {cid: self.transform(raw_tensor) for cid, raw_tensor in self.raw_cell_expr_dict.items()}

    def print_summary(self):
        print("=" * 75)
        print("  Cell Line Expression Dataset Summary")
        print("=" * 75)
        print(f"Source CSV Path    : {self.csv_path}")
        print(f"Loaded Cell Lines  : {len(self.raw_cell_expr_dict)}")
        print(f"Gene Features (dim): {self.gene_dim}")
        print(f"Normalization Fitted: {self.fitted} (Fitted on {len(self.fitted_cell_ids)} training cells)")
        if self.fitted and self.mean is not None:
            print(f"  - Training Mean Range: [{self.mean.min():.4f}, {self.mean.max():.4f}]")
            print(f"  - Training Std Range : [{self.std.min():.4f}, {self.std.max():.4f}]")
        print("=" * 75)
