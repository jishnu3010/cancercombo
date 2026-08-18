"""
Dataset and DataLoader Utilities for CancerCombo-BRICS-Symmetric.

Provides CancerComboDataset, load_cancer_combo_from_csv parser for datasets like
data/scenario3_drug1.csv (NCI-ALMANAC), and custom batch collation for PyTorch DataLoader.
"""

import os
import csv
import json
from typing import List, Dict, Union, Tuple, Optional
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .brics_utils import collate_brics_fragments


class CancerComboDataset(Dataset):
    """
    PyTorch Dataset for Cancer Drug Combination Dose-Response Viability Surfaces.

    Expected Input Format:
        - drug_pairs: List of tuples (cell_line_id, smiles_A, smiles_B).
        - dose_grids: List of tuples (doses_A, doses_B) where doses_A is (M,) and doses_B is (N,).
        - viability_surfaces: List of tensors of shape (M, N) representing percentage viability in [0, 1].
        - cell_expr_dict: Dict mapping cell_line_id -> (976,) gene expression tensor.
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
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for CancerComboDataset.

    Collates a list of dataset samples into padded tensors with BRICS fragment padding masks.
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
    fp_A, mask_A, _ = collate_brics_fragments(smiles_A_list, n_bits=frag_fp_dim, device=device)
    fp_B, mask_B, _ = collate_brics_fragments(smiles_B_list, n_bits=frag_fp_dim, device=device)

    # Stack dose concentrations and ground truth viability surface matrices
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
        "Y_true": Y_true
    }


def load_cancer_combo_from_csv(
    csv_path: str,
    cell_expr_csv: Optional[str] = None,
    split: Optional[int] = None,
    max_samples: Optional[int] = None,
    gene_dim: int = 976
) -> CancerComboDataset:
    """
    Loads dataset directly from combination dataset CSV files (such as data/scenario3_drug1.csv).

    CSV Column Mapping:
        - smiles_a / smiles_A: SMILES string of Drug A
        - smiles_b / smiles_B: SMILES string of Drug B
        - cell_line_name / cell_line_id: Cell line identifier (e.g. 7860, A549, HCT116)
        - doses_a / doses_A: JSON array string or delimited float list of Drug A concentrations
        - doses_b / doses_B: JSON array string or delimited float list of Drug B concentrations
        - viability_matrix / Y_true: JSON 2D array or delimited string of percentage viability values
        - split (optional): Integer split index for train/val/test filtering.

    Args:
        csv_path: Path to combination CSV file (e.g., data/scenario3_drug1.csv).
        cell_expr_csv: Optional path to cell line expression CSV. If None, gene expression
                       vectors are generated deterministically per unique cell line.
        split: Optional split index to filter rows (e.g., split=3).
        max_samples: Optional max sample count limit.
        gene_dim: Landmark gene count (default: 976).

    Returns:
        dataset: Configured CancerComboDataset ready for PyTorch DataLoader.
    """
    # 1. Load Cell Line Expression Dictionary if file provided
    cell_expr_dict: Dict[str, torch.Tensor] = {}
    if cell_expr_csv and os.path.exists(cell_expr_csv):
        with open(cell_expr_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row:
                    continue
                cell_id = str(row[0]).strip()
                expr_vals = [float(x) for x in row[1:1 + gene_dim]]
                if len(expr_vals) < gene_dim:
                    expr_vals.extend([0.0] * (gene_dim - len(expr_vals)))
                cell_expr_dict[cell_id] = torch.tensor(expr_vals[:gene_dim], dtype=torch.float32)

    drug_pairs: List[Tuple[str, str, str]] = []
    dose_grids: List[Tuple[torch.Tensor, torch.Tensor]] = []
    viability_surfaces: List[torch.Tensor] = []

    # 2. Parse Combination CSV File
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_samples and len(drug_pairs) >= max_samples:
                break

            # Filter by split if specified (safely handles None or missing values)
            if split is not None and row.get("split") is not None:
                raw_split = str(row["split"]).strip()
                if raw_split:
                    try:
                        if int(float(raw_split)) != split:
                            continue
                    except (ValueError, TypeError):
                        pass

            cell_id = str(row.get("cell_line_name") or row.get("cell_line_id") or row.get("cell") or "cell").strip()
            smiles_A = str(row.get("smiles_a") or row.get("smiles_A") or "").strip()
            smiles_B = str(row.get("smiles_b") or row.get("smiles_B") or "").strip()

            # Skip missing, empty, or invalid 'nan' SMILES strings
            if not smiles_A or not smiles_B or smiles_A.lower() in ("nan", "none", "null") or smiles_B.lower() in ("nan", "none", "null"):
                continue

            # Ensure cell line landmark gene expression exists in dict
            if cell_id not in cell_expr_dict:
                # Deterministic seed from cell_id string hash
                seed = abs(hash(cell_id)) % (2**31 - 1)
                rng = np.random.RandomState(seed)
                cell_expr_dict[cell_id] = torch.from_numpy(
                    rng.normal(loc=0.0, scale=1.0, size=gene_dim).astype(np.float32)
                )

            # Parse doses_a and doses_b
            doses_a_str = str(row.get("doses_a") or row.get("doses_A") or "").strip()
            doses_b_str = str(row.get("doses_b") or row.get("doses_B") or "").strip()

            if doses_a_str.startswith("["):
                doses_a_list = json.loads(doses_a_str)
            else:
                doses_a_list = [float(x) for x in doses_a_str.replace(",", " ").split() if x.strip()]

            if doses_b_str.startswith("["):
                doses_b_list = json.loads(doses_b_str)
            else:
                doses_b_list = [float(x) for x in doses_b_str.replace(",", " ").split() if x.strip()]

            doses_A = torch.tensor(doses_a_list, dtype=torch.float32)
            doses_B = torch.tensor(doses_b_list, dtype=torch.float32)

            # Parse viability matrix from JSON 2D array or delimited string
            viab_str = str(row.get("viability_matrix") or row.get("Y_true") or row.get("viability") or "").strip()
            if viab_str.startswith("["):
                matrix_raw = json.loads(viab_str)
                matrix_arr = np.array(matrix_raw, dtype=np.float32)
                # Normalize percentage (0 to 100) to fraction (0.0 to 1.0)
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

            drug_pairs.append((cell_id, smiles_A, smiles_B))
            dose_grids.append((doses_A, doses_B))
            viability_surfaces.append(Y_true)

    return CancerComboDataset(
        drug_pairs=drug_pairs,
        dose_grids=dose_grids,
        viability_surfaces=viability_surfaces,
        cell_expr_dict=cell_expr_dict
    )
