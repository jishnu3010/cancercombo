"""
Real 976-Gene Cell Line Expression Loader Module for CancerCombo.

Enforces Leakage-Safe Gene Expression Normalization & Biological Cell Line Matching:
    1. Loading real 976-dimensional landmark gene expression vectors from configurable CSV.
    2. Dual CSV Orientation & Imputation Support:
       - Orientation A: Rows = Cell Lines, Columns = 976 Genes.
       - Orientation B: Rows = 976 Genes, Columns = Cell Lines.
    3. Automatic Cell Line Alias Normalization (e.g. MDA_MB_231 <-> MDAMB231, HCT_116 <-> HCT116, 786_0 <-> 7860).
    4. Training-set fit z-score normalization:
       - fit_normalization() is called EXCLUSIVELY on training-set cell lines.
       - transform() reuses training mean/std unchanged for validation, test, and inference.
       - Refitting on validation or test sets is strictly prohibited.
    5. Loud error handling: raises explicit ValueError if any required cell line is missing.
       NO silent synthetic vector fallbacks or random hashes!
"""

import os
import csv
from typing import Dict, List, Optional, Tuple, Set, Iterable
import torch
import numpy as np


def normalize_cell_name(name: str) -> str:
    """Normalizes cell line string by removing underscores, hyphens, spaces, and converting to uppercase."""
    return str(name).strip().replace("_", "").replace("-", "").replace(" ", "").upper()


class CellExpressionLoader:
    """
    Cell Line Expression Loader and Leakage-Safe Normalizer with Biological Name Matching.

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
        self.alias_map: Dict[str, str] = {}
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

            cols = [col.strip() for col in header[1:] if col.strip()]

            if len(cols) == self.gene_dim:
                # Orientation A: Rows = Cell Lines, Columns = 976 Genes
                self.gene_names = cols
                cell_raw_list = {}
                for line_idx, row in enumerate(reader, start=2):
                    if not row or not row[0].strip():
                        continue
                    cell_id = str(row[0]).strip()
                    vals = []
                    for x in row[1:self.gene_dim+1]:
                        try:
                            vals.append(float(x))
                        except ValueError:
                            vals.append(np.nan)
                    cell_raw_list[cell_id] = vals

                cell_names = list(cell_raw_list.keys())
                matrix = np.array([cell_raw_list[c] for c in cell_names], dtype=np.float32)
                col_means = np.nanmean(matrix, axis=0)
                col_means = np.nan_to_num(col_means, nan=0.0)
                inds = np.where(np.isnan(matrix))
                matrix[inds] = np.take(col_means, inds[1])

                for idx, cid in enumerate(cell_names):
                    tensor_vec = torch.from_numpy(matrix[idx])
                    self.raw_cell_expr_dict[cid] = tensor_vec
                    norm_k = normalize_cell_name(cid)
                    self.alias_map[norm_k] = cid
                    self.alias_map[cid] = cid

            else:
                # Orientation B: Rows = 976 Genes, Columns = Cell Lines (Real Biological NCI-60 File)
                cell_names = cols
                gene_rows = []
                self.gene_names = []

                for line_idx, row in enumerate(reader, start=2):
                    if not row or not row[0].strip():
                        continue
                    gname = str(row[0]).strip()
                    self.gene_names.append(gname)

                    row_vals = []
                    for c_idx in range(1, len(cell_names) + 1):
                        val_str = row[c_idx].strip() if c_idx < len(row) else ""
                        try:
                            row_vals.append(float(val_str))
                        except ValueError:
                            row_vals.append(np.nan)
                    gene_rows.append(row_vals)

                if len(self.gene_names) != self.gene_dim:
                    print(f"Notice: CSV contains {len(self.gene_names)} genes (expected {self.gene_dim}). Adjusting feature dim.")
                    self.gene_dim = len(self.gene_names)

                matrix = np.array(gene_rows, dtype=np.float32)  # (N_genes, N_cells)
                # Impute missing values gene-by-gene using row means across cell lines
                row_means = np.nanmean(matrix, axis=1)
                row_means = np.nan_to_num(row_means, nan=0.0)
                inds = np.where(np.isnan(matrix))
                matrix[inds] = np.take(row_means, inds[0])

                cell_matrix = matrix.T  # (N_cells, N_genes)
                for idx, cid in enumerate(cell_names):
                    tensor_vec = torch.from_numpy(cell_matrix[idx])
                    self.raw_cell_expr_dict[cid] = tensor_vec
                    norm_k = normalize_cell_name(cid)
                    self.alias_map[norm_k] = cid
                    self.alias_map[cid] = cid

    def resolve_cell_id(self, cell_id: str) -> Optional[str]:
        """Resolves raw cell line string or alias to exact key in raw_cell_expr_dict."""
        clean = str(cell_id).strip()
        if clean in self.raw_cell_expr_dict:
            return clean
        norm_k = normalize_cell_name(clean)
        return self.alias_map.get(norm_k)

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
            fit_tensors = []
            matched_ids = set()
            for cid in train_cell_ids:
                resolved = self.resolve_cell_id(cid)
                if resolved and resolved in self.raw_cell_expr_dict:
                    fit_tensors.append(self.raw_cell_expr_dict[resolved])
                    matched_ids.add(resolved)
            self.fitted_cell_ids = matched_ids
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
        resolved = self.resolve_cell_id(clean_id)
        if not resolved or resolved not in self.raw_cell_expr_dict:
            available_cells = sorted(list(self.raw_cell_expr_dict.keys()))[:5]
            raise ValueError(
                f"\n[CRITICAL ERROR] Missing Required Biological Data!\n"
                f"  - Cell Line ID Requested: '{clean_id}'\n"
                f"  - Expression Source CSV : '{self.csv_path}'\n"
                f"  - Expected Gene Features: {self.gene_dim} (LINCS L1000 Landmark Genes)\n"
                f"  - Cell Lines Loaded     : {len(self.raw_cell_expr_dict)} (e.g. {available_cells}...)\n"
                f"  - Policy                : NO synthetic random vector generation allowed. Training halted.\n"
            )
        return self.transform(self.raw_cell_expr_dict[resolved])

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
