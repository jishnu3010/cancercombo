"""Data discovery, schema inspection, and validation utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import rdkit.Chem as Chem

from cancer_combo_brics.data.splitting import check_drug_overlap, categorize_scenario
from cancer_combo_brics.data.dataset import parse_dose_array, parse_viability_matrix
from cancer_combo_brics.data.preprocessing import load_cell_expression_data



def discover_data_files(root_dir: str = "./data") -> Dict[str, Optional[str]]:
    """Automatically discover combination files and cell expression files inside a directory."""
    if not os.path.exists(root_dir):
        return {"combination_file": None, "cell_expression_file": None}

    candidates = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
    comb_candidates = []
    expr_candidates = []

    for path in candidates:
        name_lower = os.path.basename(path).lower()
        if not (name_lower.endswith(".csv") or name_lower.endswith(".tsv") or
                name_lower.endswith(".parquet") or name_lower.endswith(".npz")):
            continue

        if any(k in name_lower for k in ["cell", "expression", "rnaseq", "landmark", "gene"]):
            expr_candidates.append(path)
        elif any(k in name_lower for k in ["combo", "combination", "synergy", "viability", "drug_pair"]):
            comb_candidates.append(path)

    # If not named specifically, pick largest non-expression table as combinations
    if not comb_candidates:
        other_tables = [p for p in candidates if (p.endswith(".csv") or p.endswith(".parquet") or p.endswith(".tsv")) and p not in expr_candidates]
        if other_tables:
            comb_candidates = other_tables

    return {
        "combination_file": comb_candidates[0] if comb_candidates else None,
        "cell_expression_file": expr_candidates[0] if expr_candidates else None,
    }


def validate_smiles_list(smiles_list: List[str]) -> Tuple[int, int, List[str]]:
    """Validates SMILES strings using RDKit."""
    valid_count = 0
    invalid_examples: List[str] = []

    for s in smiles_list:
        if not s or not isinstance(s, str):
            invalid_examples.append(str(s))
            continue
        mol = Chem.MolFromSmiles(s.strip())
        if mol is not None:
            valid_count += 1
        else:
            invalid_examples.append(s)

    invalid_count = len(invalid_examples)
    return valid_count, invalid_count, invalid_examples[:5]


def validate_dataset(
    comb_path: str,
    cell_path: Optional[str] = None,
    smiles_col_a: str = "smiles_a",
    smiles_col_b: str = "smiles_b",
    drug_id_col_a: str = "drug_a",
    drug_id_col_b: str = "drug_b",
    cell_id_col: str = "cell_line",
    dose_col_a: str = "doses_a",
    dose_col_b: str = "doses_b",
    viability_col: str = "viability_matrix",
    split_col: Optional[str] = "split",
) -> Dict[str, Any]:
    """Inspect and comprehensively validate combination dataset and cell expression profiles."""
    if not os.path.exists(comb_path):
        raise FileNotFoundError(f"Combinations file not found: {comb_path}")

    print(f"\n================ DATASET VALIDATION REPORT ================")
    print(f"Combinations file: {comb_path}")

    # 1. Load combinations
    if comb_path.endswith(".parquet"):
        df = pd.read_parquet(comb_path)
    elif comb_path.endswith(".tsv"):
        df = pd.read_csv(comb_path, sep="\t")
    else:
        df = pd.read_csv(comb_path)

    print(f"Total rows loaded: {len(df):,}")
    print(f"Columns present:   {list(df.columns)}")

    # 2. Check required columns
    required_cols = [smiles_col_a, smiles_col_b, cell_id_col, dose_col_a, dose_col_b, viability_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {comb_path}: {missing_cols}")

    # 3. Drugs and Cell Lines statistics
    da_col = drug_id_col_a if drug_id_col_a in df.columns else smiles_col_a
    db_col = drug_id_col_b if drug_id_col_b in df.columns else smiles_col_b

    unique_drugs = set(df[da_col].unique()) | set(df[db_col].unique())
    unique_cells = set(df[cell_id_col].unique())
    unique_pairs = set(tuple(sorted([r[da_col], r[db_col]])) for _, r in df.iterrows())

    print(f"Unique drugs:      {len(unique_drugs):,}")
    print(f"Unique cell lines: {len(unique_cells):,}")
    print(f"Unique drug pairs: {len(unique_pairs):,}")

    # 4. Validate SMILES
    unique_smiles = set(df[smiles_col_a].dropna().unique()) | set(df[smiles_col_b].dropna().unique())
    valid_smi, invalid_smi, invalid_examples = validate_smiles_list(list(unique_smiles))
    print(f"Unique SMILES:     {len(unique_smiles):,} (Valid: {valid_smi:,}, Invalid: {invalid_smi:,})")
    if invalid_smi > 0:
        print(f"[WARNING] Invalid SMILES examples: {invalid_examples}")

    # 5. Dose and Viability Dimensions & Values
    sample_doses_a = parse_dose_array(df.iloc[0][dose_col_a])
    sample_doses_b = parse_dose_array(df.iloc[0][dose_col_b])
    dim_a = len(sample_doses_a)
    dim_b = len(sample_doses_b)

    sample_matrix = parse_viability_matrix(df.iloc[0][viability_col])
    print(f"Dose dimensions:   Drug A = {dim_a}, Drug B = {dim_b}")
    print(f"Surface dimensions: {sample_matrix.shape} (Expected: ({dim_a}, {dim_b}))")

    # Sample viability values to detect units
    all_sample_viab = []
    for i in range(min(50, len(df))):
        m = parse_viability_matrix(df.iloc[i][viability_col])
        all_sample_viab.extend(m.flatten().tolist())

    v_arr = np.array(all_sample_viab)
    v_min = float(np.nanmin(v_arr))
    v_max = float(np.nanmax(v_arr))
    v_mean = float(np.nanmean(v_arr))
    v_std = float(np.nanstd(v_arr))

    print(f"Target distribution (sample 50 surfaces):")
    print(f"  Min: {v_min:.2f} | Max: {v_max:.2f} | Mean: {v_mean:.2f} | Std: {v_std:.2f}")

    if v_max > 3.0:
        print(f"  Scale detected: PERCENTAGE VIABILITY")
        print(f"  Training convention: 100.0 = 100% viability. No target /100 normalization is applied.")
    else:
        print(f"  Scale detected: ALREADY NORMALIZED VIABILITY (1.0 = 100%)")

    # 6. Cell expression validation
    cell_report = {}
    if cell_path and os.path.exists(cell_path):
        print(f"\nCell expression file: {cell_path}")
        expr_matrix, cell_names = load_cell_expression_data(cell_path, known_cell_names=list(unique_cells))


        print(f"Cell expression matrix shape: {expr_matrix.shape}")
        feature_dim = expr_matrix.shape[1]
        print(f"Landmark gene feature count: {feature_dim} (Expected: 976)")
        missing_in_expr = unique_cells - set(cell_names)
        print(f"Cell lines covered: {len(cell_names):,} (Missing for combinations: {len(missing_in_expr)})")
        if missing_in_expr:
            print(f"[WARNING] Cell lines missing expression: {list(missing_in_expr)[:5]}")

        cell_report = {
            "cell_feature_dim": feature_dim,
            "cell_count": len(cell_names),
            "missing_cells_count": len(missing_in_expr),
        }
    else:
        print(f"\n[INFO] Cell expression file not specified or not yet present at: {cell_path}")

    # 7. Split overlap check if split column exists
    split_report = {}
    if split_col and split_col in df.columns:
        print(f"\nSplit column '{split_col}' detected. Validating existing splits...")
        splits = df[split_col].unique()
        print(f"Split counts: {df[split_col].value_counts().to_dict()}")

        train_drugs = set(df[df[split_col] == "train"][da_col]) | set(df[df[split_col] == "train"][db_col])
        val_drugs = set(df[df[split_col] == "val"][da_col]) | set(df[df[split_col] == "val"][db_col]) if "val" in splits else set()
        test_drugs = set(df[df[split_col] == "test"][da_col]) | set(df[df[split_col] == "test"][db_col]) if "test" in splits else set()

        overlap = check_drug_overlap(train_drugs, val_drugs, test_drugs)
        split_report = {"overlap": overlap, "split_counts": df[split_col].value_counts().to_dict()}

    print(f"===========================================================\n")

    return {
        "num_rows": len(df),
        "num_drugs": len(unique_drugs),
        "num_cells": len(unique_cells),
        "num_pairs": len(unique_pairs),
        "dose_dims": (dim_a, dim_b),
        "target_stats": {"min": v_min, "max": v_max, "mean": v_mean, "std": v_std},
        "smiles_valid_count": valid_smi,
        "smiles_invalid_count": invalid_smi,
        "cell_report": cell_report,
        "split_report": split_report,
    }
