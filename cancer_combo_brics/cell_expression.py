"""
Real 976-Gene Cell Line Expression Loader Module for CancerCombo.

Enforces:
    1. Loading real 976-dimensional landmark gene expression vectors from configurable CSV.
    2. Strict feature dimension verification (gene_dim == 976).
    3. Fixed deterministic gene feature ordering.
    4. Training-set fit z-score normalization (leakage-safe).
    5. Loud error handling: raises explicit ValueError if any required cell line is missing.
       NO silent synthetic vector fallbacks or random hashes!
"""

import os
import csv
from typing import Dict, List, Optional, Tuple, Set
import torch
import numpy as np


class CellExpressionLoader:
    """
    Cell Line Expression Loader and Normalizer.

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

        self.cell_expr_dict: Dict[str, torch.Tensor] = {}
        self.gene_names: List[str] = []
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

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

                self.cell_expr_dict[cell_id] = expr_tensor

    def fit_normalization(self, train_cell_ids: Optional[Set[str]] = None):
        """Fits z-score normalization parameters on training set cell lines only to prevent data leakage."""
        if not self.normalize or not self.cell_expr_dict:
            return

        if train_cell_ids:
            fit_tensors = [self.cell_expr_dict[cid] for cid in train_cell_ids if cid in self.cell_expr_dict]
        else:
            fit_tensors = list(self.cell_expr_dict.values())

        if not fit_tensors:
            return

        stacked = torch.stack(fit_tensors, dim=0)  # (N_cells, 976)
        self.mean = torch.mean(stacked, dim=0)
        self.std = torch.std(stacked, dim=0)
        # Avoid division by zero
        self.std = torch.where(self.std < 1e-6, torch.ones_like(self.std), self.std)

        # Apply normalization to dictionary
        for cid in list(self.cell_expr_dict.keys()):
            self.cell_expr_dict[cid] = (self.cell_expr_dict[cid] - self.mean) / self.std

    def get_cell_expression(self, cell_id: str) -> torch.Tensor:
        """
        Retrieves normalized 976-dimensional gene expression vector for target cell line.
        Fails loudly if the required cell line is missing.
        """
        clean_id = str(cell_id).strip()
        if clean_id not in self.cell_expr_dict:
            available_cells = sorted(list(self.cell_expr_dict.keys()))[:5]
            raise ValueError(
                f"\n[CRITICAL ERROR] Missing Required Biological Data!\n"
                f"  - Cell Line ID Requested: '{clean_id}'\n"
                f"  - Expression Source CSV : '{self.csv_path}'\n"
                f"  - Expected Gene Features: {self.gene_dim} (LINCS L1000 Landmark Genes)\n"
                f"  - Cell Lines Loaded     : {len(self.cell_expr_dict)} (e.g. {available_cells}...)\n"
                f"  - Policy                : NO synthetic random vector generation allowed. Training halted.\n"
            )
        return self.cell_expr_dict[clean_id]

    def print_summary(self):
        print("=" * 75)
        print("  Cell Line Expression Dataset Summary")
        print("=" * 75)
        print(f"Source CSV Path    : {self.csv_path}")
        print(f"Loaded Cell Lines  : {len(self.cell_expr_dict)}")
        print(f"Gene Features (dim): {self.gene_dim}")
        print(f"Normalized         : {self.normalize} (Mean/Std fit on training set)")
        print("=" * 75)
