"""
Dataset and Data Loading Module for CancerCombo-BRICS-Symmetric.

Enforces:
    1. Loading real 976-dimensional landmark gene expression profiles via CellExpressionLoader.
    2. Strict loud error handling: NO silent synthetic random vector fallback or hash(cell_id).
    3. Fast precomputed BRICS fragment feature lookup via BRICSCache.
    4. Target range auditing and explicit clipping to [0, 1].
    5. Arbitrary M x N dose grid support.
"""

import os
import csv
import json
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from .brics_utils import collate_brics_fragments
from .cell_expression import CellExpressionLoader
from .brics_cache import BRICSCache


class CancerComboDataset(Dataset):
    """
    CancerCombo Dataset class storing drug combination samples.
    """

    def __init__(
        self,
        drug_pairs: List[Tuple[str, str, str]],
        dose_grids: List[Tuple[torch.Tensor, torch.Tensor]],
        viability_surfaces: List[torch.Tensor],
        cell_expr_dict: Dict[str, torch.Tensor],
    ):
        assert len(drug_pairs) == len(dose_grids) == len(viability_surfaces), (
            "Length mismatch among drug_pairs, dose_grids, and viability_surfaces."
        )
        self.drug_pairs = drug_pairs
        self.dose_grids = dose_grids
        self.viability_surfaces = viability_surfaces
        self.cell_expr_dict = cell_expr_dict

    def __len__(self) -> int:
        return len(self.drug_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        cell_id, smiles_A, smiles_B = self.drug_pairs[idx]
        doses_A, doses_B = self.dose_grids[idx]
        Y_true = self.viability_surfaces[idx]

        if cell_id not in self.cell_expr_dict:
            raise ValueError(
                f"[CRITICAL DATA ERROR] Cell line '{cell_id}' missing from expression dictionary during dataset indexing!"
            )
        cell_expr = self.cell_expr_dict[cell_id]

        return {
            "cell_id": cell_id,
            "cell_expr": cell_expr,
            "smiles_A": smiles_A,
            "smiles_B": smiles_B,
            "doses_A": doses_A,
            "doses_B": doses_B,
            "Y_true": Y_true
        }


def collate_cancer_combo_batch(
    batch: List[Dict[str, Union[torch.Tensor, str]]],
    frag_fp_dim: int = 2048,
    device: Union[torch.device, str] = "cpu"
) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Custom collate function for CancerComboDataset.
    """
    cell_expr_list = [item["cell_expr"] for item in batch]
    smiles_A_list = [item["smiles_A"] for item in batch]
    smiles_B_list = [item["smiles_B"] for item in batch]
    doses_A_list = [item["doses_A"] for item in batch]
    doses_B_list = [item["doses_B"] for item in batch]
    Y_true_list = [item["Y_true"] for item in batch]

    # Stack cell line expressions: (B, 976)
    cell_expr = torch.stack(cell_expr_list, dim=0).to(device)

    # Collate BRICS fragments for Drug A and Drug B into padded tensors and boolean masks
    fp_A, mask_A, frags_A_list = collate_brics_fragments(smiles_A_list, n_bits=frag_fp_dim, device=device)
    fp_B, mask_B, frags_B_list = collate_brics_fragments(smiles_B_list, n_bits=frag_fp_dim, device=device)

    doses_A = torch.stack(doses_A_list, dim=0).to(device)
    doses_B = torch.stack(doses_B_list, dim=0).to(device)
    Y_true = torch.stack(Y_true_list, dim=0).to(device)

    return {
        "cell_expr": cell_expr,
        "fp_A": fp_A,
        "mask_A": mask_A,
        "fp_B": fp_B,
        "mask_B": mask_B,
        "dose_grid": (doses_A, doses_B),
        "Y_true": Y_true,
        "smiles_A": smiles_A_list,
        "smiles_B": smiles_B_list,
        "frags_A": frags_A_list,
        "frags_B": frags_B_list
    }


def load_cancer_combo_from_csv(
    csv_path: str,
    cell_expr_csv: Optional[str] = os.path.join("data", "cell_line_gene_expr.csv"),
    split: Optional[Union[int, List[int]]] = None,
    max_samples: Optional[int] = None,
    gene_dim: int = 976,
    brics_cache: Optional[BRICSCache] = None,
    expr_loader: Optional[CellExpressionLoader] = None,
    fit_normalization: bool = False
) -> CancerComboDataset:
    """
    Loads dataset directly from combination dataset CSV files.

    Enforces Leakage-Safe Normalization Protocol:
        - If expr_loader is not provided, creates a new CellExpressionLoader.
        - If fit_normalization is True (or expr_loader is not yet fitted and split == 1),
          fits normalization statistics (mean/std) strictly on training set cell lines.
        - If expr_loader is already fitted, reuses training mean/std unchanged without refitting.
        - Fails loudly if any required cell line is missing from expr_loader.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Combination dataset CSV not found at '{csv_path}'.")

    # 1. Initialize or Reuse Cell Expression Loader
    if expr_loader is None:
        expr_loader = CellExpressionLoader(csv_path=cell_expr_csv, gene_dim=gene_dim)

    drug_pairs: List[Tuple[str, str, str]] = []
    dose_grids: List[Tuple[torch.Tensor, torch.Tensor]] = []
    viability_surfaces: List[torch.Tensor] = []
    required_cell_ids: Set[str] = set()

    # Target statistics tracking
    raw_min, raw_max = float("inf"), float("-inf")
    total_elements = 0
    outside_01_count = 0

    # 2. Parse Combination CSV File
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_samples and len(drug_pairs) >= max_samples:
                break

            # Filter by split if specified
            if split is not None and row.get("split") is not None:
                raw_split = str(row["split"]).strip()
                if raw_split:
                    try:
                        split_val = int(float(raw_split))
                        if isinstance(split, (list, tuple, set)):
                            if split_val not in split:
                                continue
                        else:
                            if split_val != split:
                                continue
                    except (ValueError, TypeError):
                        pass

            cell_id = str(row.get("cell_line_name") or row.get("cell_line_id") or row.get("cell") or "").strip()
            smiles_A = str(row.get("smiles_a") or row.get("smiles_A") or "").strip()
            smiles_B = str(row.get("smiles_b") or row.get("smiles_B") or "").strip()

            if not cell_id or not smiles_A or not smiles_B or smiles_A.lower() in ("nan", "none", "null") or smiles_B.lower() in ("nan", "none", "null"):
                continue

            # Parse doses
            doses_a_str = str(row.get("doses_a") or row.get("doses_A") or "").strip()
            doses_b_str = str(row.get("doses_b") or row.get("doses_B") or "").strip()

            if doses_a_str.startswith("["):
                try:
                    doses_a_list = json.loads(doses_a_str)
                except (json.JSONDecodeError, ValueError):
                    import ast
                    doses_a_list = ast.literal_eval(doses_a_str)
            else:
                doses_a_list = [float(x) for x in doses_a_str.replace(",", " ").split() if x.strip()]

            if doses_b_str.startswith("["):
                try:
                    doses_b_list = json.loads(doses_b_str)
                except (json.JSONDecodeError, ValueError):
                    import ast
                    doses_b_list = ast.literal_eval(doses_b_str)
            else:
                doses_b_list = [float(x) for x in doses_b_str.replace(",", " ").split() if x.strip()]

            doses_A = torch.tensor(doses_a_list, dtype=torch.float32)
            doses_B = torch.tensor(doses_b_list, dtype=torch.float32)

            # Parse viability matrix
            viab_str = str(row.get("viability_matrix") or row.get("Y_true") or row.get("viability") or "").strip()
            if viab_str.startswith("["):
                try:
                    matrix_raw = json.loads(viab_str)
                except (json.JSONDecodeError, ValueError):
                    import ast
                    matrix_raw = ast.literal_eval(viab_str)
                matrix_arr = np.array(matrix_raw, dtype=np.float32)
                if matrix_arr.max() > 2.0:
                    matrix_arr = matrix_arr / 100.0
                Y_true = torch.from_numpy(matrix_arr)
            else:
                matrix_rows = []
                for r in viab_str.split(";"):
                    if r.strip():
                        cols = [float(x) for x in r.replace(",", " ").split() if x.strip()]
                        matrix_rows.append(cols)
                Y_true = torch.tensor(matrix_rows, dtype=torch.float32)
                if Y_true.max() > 2.0:
                    Y_true = Y_true / 100.0

            # Target Statistics Audit
            raw_min = min(raw_min, float(Y_true.min()))
            raw_max = max(raw_max, float(Y_true.max()))
            total_elements += Y_true.numel()
            outside_01_count += int(((Y_true < 0.0) | (Y_true > 1.0)).sum())

            # Explicitly clip target viability to valid physical range [0.0, 1.0]
            Y_true_clipped = torch.clamp(Y_true, 0.0, 1.0)

            drug_pairs.append((cell_id, smiles_A, smiles_B))
            dose_grids.append((doses_A, doses_B))
            viability_surfaces.append(Y_true_clipped)
            required_cell_ids.add(cell_id)

    # 3. Fit Normalization ONLY if explicitly requested or if expr_loader is not yet fitted
    if fit_normalization or not expr_loader.fitted:
        expr_loader.fit_normalization(train_cell_ids=required_cell_ids)

    # 4. Construct Cell Expression Dictionary for dataset using training-fitted mean & std
    cell_expr_dict: Dict[str, torch.Tensor] = {}
    for cid in required_cell_ids:
        cell_expr_dict[cid] = expr_loader.get_cell_expression(cid)

    # Print Target Range Audit Summary
    print(f"Loaded {len(drug_pairs)} samples from '{csv_path}' (split={split}).")
    if total_elements > 0:
        pct_outside = (outside_01_count / total_elements) * 100.0
        print(f"  Target Range Audit: Raw Min={raw_min:.4f}, Raw Max={raw_max:.4f} | {pct_outside:.2f}% values outside [0, 1] (clipped to [0, 1]).")

    dataset = CancerComboDataset(
        drug_pairs=drug_pairs,
        dose_grids=dose_grids,
        viability_surfaces=viability_surfaces,
        cell_expr_dict=cell_expr_dict
    )
    dataset.expr_loader = expr_loader
    return dataset


def load_cancer_combo_splits(
    data_csv: str,
    cell_expr_csv: Optional[str] = os.path.join("data", "cell_line_gene_expr.csv"),
    train_split: int = 1,
    val_split: int = 2,
    test_split: int = 3,
    max_samples: Optional[int] = None,
    gene_dim: int = 976
) -> Tuple[CancerComboDataset, CancerComboDataset, CancerComboDataset, CellExpressionLoader]:
    """
    Loads Train, Val, and Test dataset splits with LEAKAGE-SAFE cell expression normalization:
        1. Fit normalization mean and std ONCE using training-split cell lines ONLY.
        2. Reuse the exact same fitted training normalizer for validation and test splits without refitting.
    """
    expr_loader = CellExpressionLoader(csv_path=cell_expr_csv, gene_dim=gene_dim)

    # 1. Load Train Split (Fits expr_loader ONCE on training cell lines)
    train_dataset = load_cancer_combo_from_csv(
        csv_path=data_csv,
        cell_expr_csv=cell_expr_csv,
        split=train_split,
        max_samples=max_samples,
        gene_dim=gene_dim,
        expr_loader=expr_loader,
        fit_normalization=True
    )

    # 2. Load Val Split (Reuses fitted training expr_loader WITHOUT refitting)
    val_dataset = load_cancer_combo_from_csv(
        csv_path=data_csv,
        cell_expr_csv=cell_expr_csv,
        split=val_split,
        max_samples=max_samples,
        gene_dim=gene_dim,
        expr_loader=expr_loader,
        fit_normalization=False
    )

    # 3. Load Test Split (Reuses fitted training expr_loader WITHOUT refitting)
    test_dataset = load_cancer_combo_from_csv(
        csv_path=data_csv,
        cell_expr_csv=cell_expr_csv,
        split=test_split,
        max_samples=max_samples,
        gene_dim=gene_dim,
        expr_loader=expr_loader,
        fit_normalization=False
    )

    return train_dataset, val_dataset, test_dataset, expr_loader

