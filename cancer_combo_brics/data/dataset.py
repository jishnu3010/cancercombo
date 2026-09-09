"""Cancer combination dataset and collation function with explicit invalid/missing SMILES tracking."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cancer_combo_brics.chemistry.cache import FunctionalGroupCache
from cancer_combo_brics.chemistry.fragment_utils import pad_fragment_strings


def is_valid_smiles(val: Any) -> bool:
    """Check whether a SMILES input is a non-empty, non-NaN valid string."""
    if pd.isna(val) or val is None:
        return False
    s = str(val).strip()
    if not s or s.lower() == "nan" or s.lower() == "none" or s.lower() == "null":
        return False
    return True


def parse_dose_array(val: Any) -> np.ndarray:
    """Safely parse dose values from string, list, or array into 1D float32 numpy array."""
    if isinstance(val, np.ndarray):
        return val.astype(np.float32).flatten()
    if isinstance(val, (list, tuple)):
        return np.array(val, dtype=np.float32).flatten()
    if isinstance(val, str):
        cleaned = val.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                return np.array(json.loads(cleaned), dtype=np.float32).flatten()
            except Exception:
                pass
        parts = cleaned.replace("[", "").replace("]", "").replace(",", " ").split()
        return np.array([float(p) for p in parts], dtype=np.float32)
    raise ValueError(f"Unable to parse dose array from type {type(val)}: {val}")


def parse_viability_matrix(val: Any, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Safely parse 2D viability surface from string, list, or array."""
    if isinstance(val, np.ndarray):
        arr = val.astype(np.float32)
    elif isinstance(val, (list, tuple)):
        arr = np.array(val, dtype=np.float32)
    elif isinstance(val, str):
        cleaned = val.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                arr = np.array(json.loads(cleaned), dtype=np.float32)
            except Exception:
                parts = cleaned.replace("[", "").replace("]", "").replace(",", " ").split()
                arr = np.array([float(p) for p in parts], dtype=np.float32)
        else:
            parts = cleaned.replace(",", " ").split()
            arr = np.array([float(p) for p in parts], dtype=np.float32)
    else:
        raise ValueError(f"Unable to parse viability matrix from type {type(val)}")

    if shape is not None and arr.shape != shape:
        arr = arr.reshape(shape)

    return arr


class CancerComboDataset(Dataset):
    """Dataset for drug combination synergy / 2D dose-response surface prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        cell_expressions: Dict[str, np.ndarray],
        fg_cache: Optional[FunctionalGroupCache] = None,
        smiles_col_a: str = "smiles_a",
        smiles_col_b: str = "smiles_b",
        cell_id_col: str = "cell_line",
        dose_col_a: str = "doses_a",
        dose_col_b: str = "doses_b",
        viability_col: str = "viability_matrix",
        drug_id_col_a: str = "drug_a",
        drug_id_col_b: str = "drug_b",
        target_scale: str = "normalized_viability",
        max_fragments: int = 32,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.cell_expressions = cell_expressions
        self.fg_cache = fg_cache or FunctionalGroupCache()

        # Schema resilience with fallbacks
        self.smiles_col_a = smiles_col_a if smiles_col_a in self.df.columns else ("Drug1_SMILES" if "Drug1_SMILES" in self.df.columns else smiles_col_a)
        self.smiles_col_b = smiles_col_b if smiles_col_b in self.df.columns else ("Drug2_SMILES" if "Drug2_SMILES" in self.df.columns else smiles_col_b)
        self.cell_id_col = cell_id_col if cell_id_col in self.df.columns else ("cell_line_name" if "cell_line_name" in self.df.columns else ("Sample" if "Sample" in self.df.columns else cell_id_col))
        self.dose_col_a = dose_col_a if dose_col_a in self.df.columns else ("dose_a" if "dose_a" in self.df.columns else dose_col_a)
        self.dose_col_b = dose_col_b if dose_col_b in self.df.columns else ("dose_b" if "dose_b" in self.df.columns else dose_col_b)
        self.viability_col = viability_col if viability_col in self.df.columns else ("viability" if "viability" in self.df.columns else viability_col)
        self.drug_id_col_a = drug_id_col_a
        self.drug_id_col_b = drug_id_col_b
        self.target_scale = target_scale
        self.max_fragments = max_fragments

        # Finalcheck parity: Ground truth viability is percentage scale (100.0 = 100% viability)
        # Never divide by 100.0. Scale factor is strictly 1.0.
        self.scale_factor = 1.0

        # Preload functional group extractions for VALID unique SMILES only
        valid_a = [str(s).strip() for s in self.df[self.smiles_col_a] if is_valid_smiles(s)]
        valid_b = [str(s).strip() for s in self.df[self.smiles_col_b] if is_valid_smiles(s)]
        all_valid_smiles = list(set(valid_a + valid_b))
        self.fg_cache.preload_dataset_smiles(all_valid_smiles)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # 1. Check SMILES validity
        raw_a = row[self.smiles_col_a]
        raw_b = row[self.smiles_col_b]

        valid_a = is_valid_smiles(raw_a)
        valid_b = is_valid_smiles(raw_b)
        is_valid_sample = valid_a and valid_b

        if is_valid_sample:
            smiles_a = str(raw_a).strip()
            smiles_b = str(raw_b).strip()
            frags_a = self.fg_cache.get_or_decompose(smiles_a)
            frags_b = self.fg_cache.get_or_decompose(smiles_b)
        else:
            smiles_a = str(raw_a) if pd.notna(raw_a) else ""
            smiles_b = str(raw_b) if pd.notna(raw_b) else ""
            frags_a = []
            frags_b = []

        # 2. Cell expression
        cell_id = str(row[self.cell_id_col])
        if cell_id not in self.cell_expressions:
            norm_key = "".join(c for c in cell_id.upper() if c.isalnum())
            if norm_key in self.cell_expressions:
                cell_expr = self.cell_expressions[norm_key]
            else:
                raise KeyError(f"Cell line '{cell_id}' not found in cell expression dictionary.")
        else:
            cell_expr = self.cell_expressions[cell_id]

        # 3. Doses converted from Molar to MicroMolar exactly once (* 1e6)
        doses_a = (parse_dose_array(row[self.dose_col_a]) * 1e6).astype(np.float32)
        doses_b = (parse_dose_array(row[self.dose_col_b]) * 1e6).astype(np.float32)

        # 4. Viability matrix preserved in percentage scale (100.0 = 100% viability)
        expected_shape = (len(doses_a), len(doses_b))
        raw_viab = parse_viability_matrix(row[self.viability_col], shape=expected_shape)
        viab_percentage = (raw_viab * self.scale_factor).astype(np.float32)

        # Metadata
        drug_a_id = str(row[self.drug_id_col_a]) if self.drug_id_col_a in row else smiles_a
        drug_b_id = str(row[self.drug_id_col_b]) if self.drug_id_col_b in row else smiles_b
        scenario = int(row["scenario"]) if "scenario" in row else 1

        return {
            "cell_expr": cell_expr.astype(np.float32),
            "frags_a": frags_a,
            "frags_b": frags_b,
            "doses_a": doses_a,
            "doses_b": doses_b,
            "viability_matrix": viab_percentage,
            "cell_line": cell_id,
            "drug_a": drug_a_id,
            "drug_b": drug_b_id,
            "scenario": scenario,
            "is_valid_sample": is_valid_sample,
        }


def collate_combo_batch(batch: List[Dict[str, Any]], max_fragments: int = 32) -> Dict[str, Any]:
    """Collates a batch of drug combination samples with fragment padding and tensor conversion."""
    B = len(batch)

    # 1. Stack cell expressions -> (B, 976)
    cell_exprs = torch.from_numpy(np.stack([b["cell_expr"] for b in batch], axis=0))

    # 2. Pad fragments for Drug A and Drug B
    frags_a_raw = [b["frags_a"] for b in batch]
    frags_b_raw = [b["frags_b"] for b in batch]

    padded_a, mask_a = pad_fragment_strings(frags_a_raw, max_fragments=max_fragments)
    padded_b, mask_b = pad_fragment_strings(frags_b_raw, max_fragments=max_fragments)

    # 3. Doses
    doses_a = torch.from_numpy(np.stack([b["doses_a"] for b in batch], axis=0))
    doses_b = torch.from_numpy(np.stack([b["doses_b"] for b in batch], axis=0))

    # 4. Viability matrices
    viab_matrices = torch.from_numpy(np.stack([b["viability_matrix"] for b in batch], axis=0))

    # Metadata & Validity masks
    is_valid_samples = torch.tensor([b["is_valid_sample"] for b in batch], dtype=torch.bool)
    scenarios = torch.tensor([b["scenario"] for b in batch], dtype=torch.long)
    cell_lines = [b["cell_line"] for b in batch]
    drug_pairs = [f"{b['drug_a']} + {b['drug_b']}" for b in batch]

    return {
        "cell_expr": cell_exprs,
        "fragments_A": padded_a,
        "mask_A": mask_a,
        "fragments_B": padded_b,
        "mask_B": mask_b,
        "doses_A": doses_a,
        "doses_B": doses_b,
        "viability_matrix": viab_matrices,
        "is_valid_sample": is_valid_samples,
        "scenarios": scenarios,
        "cell_lines": cell_lines,
        "drug_pairs": drug_pairs,
    }


class ComboBatchCollator:
    """Picklable collator class for PyTorch DataLoader worker processes."""

    def __init__(self, max_fragments: int = 32):
        self.max_fragments = max_fragments

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_combo_batch(batch, max_fragments=self.max_fragments)
