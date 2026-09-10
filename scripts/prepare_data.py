"""Data preparation, validation, splitting, and synthetic dataset generation script."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cancer_combo_brics.config import ExperimentConfig
from cancer_combo_brics.data.validation import discover_data_files, validate_dataset
from cancer_combo_brics.data.splitting import create_drug_disjoint_splits, validate_existing_splits
from cancer_combo_brics.data.preprocessing import CellExpressionPreprocessor
from cancer_combo_brics.chemistry.cache import FunctionalGroupCache


def generate_synthetic_data(data_dir: str = "./data", num_samples: int = 100) -> Tuple[str, str]:
    """Generate a realistic synthetic cancer combination dataset for testing and dry runs."""
    os.makedirs(data_dir, exist_ok=True)
    comb_path = os.path.join(data_dir, "synthetic_combinations.csv")
    cell_path = os.path.join(data_dir, "synthetic_cell_expressions.npz")

    print(f"Generating synthetic dataset with {num_samples} samples in {data_dir}...")

    # Canonical SMILES for synthetic drugs
    drug_catalog = [
        ("D01", "CC(=O)Oc1ccccc1C(=O)O"),                # Aspirin
        ("D02", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"),   # Testosterone
        ("D03", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),          # Caffeine
        ("D04", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),            # Ibuprofen
        ("D05", "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1"),         # Albuterol
        ("D06", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c3ccc(Cl)cc3"), # Indomethacin
        ("D07", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),            # Theophylline
        ("D08", "CC(=O)Nc1ccc(O)cc1"),                    # Paracetamol
        ("D09", "c1ccc(cc1)c2ccccc2"),                    # Biphenyl
        ("D10", "COc1ccccc1O"),                           # Guaiacol
    ]

    cell_lines = [f"CELL_{i:02d}" for i in range(1, 9)]

    # 1. Generate 976-D landmark gene expressions
    np.random.seed(42)
    expr_matrix = np.random.randn(len(cell_lines), 976).astype(np.float32)
    np.savez_compressed(cell_path, expressions=expr_matrix, cell_lines=np.array(cell_lines))
    print(f"Saved synthetic cell expressions: {cell_path} (shape: {expr_matrix.shape})")

    # 2. 4-point dose grids in Molar (M) matching real data (e.g., 6e-8 M = 0.06 uM)
    doses_a_M = np.array([0.0, 6e-08, 6e-07, 6e-06], dtype=np.float32)
    doses_b_M = np.array([0.0, 5e-08, 5e-07, 5e-06], dtype=np.float32)

    # In uM for Hill equation computation
    d_a_uM = doses_a_M * 1e6
    d_b_uM = doses_b_M * 1e6

    rows = []
    for i in range(num_samples):
        # Pick two distinct drugs
        d1_idx, d2_idx = np.random.choice(len(drug_catalog), size=2, replace=False)
        d1_id, d1_smi = drug_catalog[d1_idx]
        d2_id, d2_smi = drug_catalog[d2_idx]
        cell_id = np.random.choice(cell_lines)

        # Synthetic ground truth bivariate Hill surface with e0 = 100.0 (percentage viability):
        uA = np.where(d_a_uM > 0, (d_a_uM / 0.5) ** 1.2, 0.0)[:, None]
        uB = np.where(d_b_uM > 0, (d_b_uM / 0.5) ** 1.5, 0.0)[None, :]
        uAB = 2.0 * uA * uB
        viab = (100.0 + 30.0 * uA + 40.0 * uB + 10.0 * uAB) / (1.0 + uA + uB + uAB)
        # Add small realistic noise
        viab = np.clip(viab + np.random.normal(0, 1.0, size=viab.shape), 0.0, 120.0)

        viab_percentage = viab.tolist()

        rows.append({
            "drug_a": d1_id,
            "drug_b": d2_id,
            "smiles_a": d1_smi,
            "smiles_b": d2_smi,
            "cell_line": cell_id,
            "cell_line_name": cell_id,
            "doses_a": json.dumps(doses_a_M.tolist()),
            "doses_b": json.dumps(doses_b_M.tolist()),
            "viability_matrix": json.dumps(viab_percentage),
        })

    df = pd.DataFrame(rows)
    df.to_csv(comb_path, index=False)
    print(f"Saved synthetic combinations: {comb_path} ({len(df)} rows)")
    return comb_path, cell_path


def main():
    parser = argparse.ArgumentParser(description="Prepare, validate, and split CancerCombo-BRICS data.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test data in ./data")
    parser.add_argument("--samples", type=int, default=100, help="Number of synthetic samples to generate")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if os.path.exists(args.config) else ExperimentConfig()

    comb_file = cfg.data.combination_file
    cell_file = cfg.data.cell_expression_file

    if args.synthetic:
        comb_file, cell_file = generate_synthetic_data(data_dir=cfg.data.root, num_samples=args.samples)
        cfg.data.combination_file = comb_file
        cfg.data.cell_expression_file = cell_file

    # Auto-discover if files not explicitly defined
    if not comb_file or not os.path.exists(comb_file):
        print(f"Discovering data files in {cfg.data.root}...")
        discovered = discover_data_files(cfg.data.root)
        comb_file = discovered.get("combination_file")
        cell_file = discovered.get("cell_expression_file")

    if not comb_file or not os.path.exists(comb_file):
        print(f"\n[NOTICE] No combinations file found in '{cfg.data.root}'.")
        print("To generate synthetic test data immediately, run: python scripts/prepare_data.py --synthetic")
        sys.exit(0)

    # Validate dataset
    val_report = validate_dataset(
        comb_path=comb_file,
        cell_path=cell_file,
        smiles_col_a=cfg.data.smiles_col_a,
        smiles_col_b=cfg.data.smiles_col_b,
        drug_id_col_a=cfg.data.drug_id_col_a,
        drug_id_col_b=cfg.data.drug_id_col_b,
        cell_id_col=cfg.data.cell_id_col,
        dose_col_a=cfg.data.dose_col_a,
        dose_col_b=cfg.data.dose_col_b,
        viability_col=cfg.data.viability_col,
        split_col=cfg.data.split_col,
    )

    # Load dataframe
    df = pd.read_csv(comb_file) if comb_file.endswith(".csv") else pd.read_parquet(comb_file)

    # Perform or validate drug-disjoint split
    if cfg.data.split_col and cfg.data.split_col in df.columns:
        print(f"Validating existing split column '{cfg.data.split_col}'...")
        validate_existing_splits(
            df,
            split_col=cfg.data.split_col,
            drug_col_a=cfg.data.drug_id_col_a if cfg.data.drug_id_col_a in df.columns else cfg.data.smiles_col_a,
            drug_col_b=cfg.data.drug_id_col_b if cfg.data.drug_id_col_b in df.columns else cfg.data.smiles_col_b,
        )
    else:
        print("Creating strict drug-disjoint splits...")
        df = create_drug_disjoint_splits(
            df,
            drug_col_a=cfg.data.drug_id_col_a if cfg.data.drug_id_col_a in df.columns else cfg.data.smiles_col_a,
            drug_col_b=cfg.data.drug_id_col_b if cfg.data.drug_id_col_b in df.columns else cfg.data.smiles_col_b,
            test_ratio=0.15,
            val_ratio=0.15,
            seed=cfg.training.seed,
        )
        # Save split back to disk
        df.to_csv(comb_file, index=False)
        print(f"Updated {comb_file} with 'split' and 'scenario' columns.")

    # Train cell expression preprocessor
    if cell_file and os.path.exists(cell_file):
        print("\nComputing train-only cell expression preprocessing statistics...")
        if cell_file.endswith(".npz"):
            c_data = np.load(cell_file)
            c_exprs = c_data["expressions"]
            c_names = list(c_data["cell_lines"])
        else:
            c_df = pd.read_csv(cell_file, index_col=0)
            c_exprs = c_df.values
            c_names = list(c_df.index)

        # Identify cells present in train split
        train_cells = df[df["split"] == "train"][cfg.data.cell_id_col].unique()
        train_indices = [i for i, name in enumerate(c_names) if name in train_cells]
        if not train_indices:
            train_indices = list(range(len(c_names)))

        train_expr_subset = c_exprs[train_indices]
        preprocessor = CellExpressionPreprocessor(expected_dim=c_exprs.shape[1])
        preprocessor.fit(train_expr_subset)
        preprocessor.save(cfg.data.cell_preprocessor_file)
        print(f"Saved train cell preprocessor to {cfg.data.cell_preprocessor_file}")

    # Pre-populate functional-group fragment cache
    print("\nPre-computing functional-group fragmentations for unique SMILES...")
    cache = FunctionalGroupCache(db_path=cfg.data.fg_cache_file)
    all_smi = list(df[cfg.data.smiles_col_a].dropna().unique()) + list(df[cfg.data.smiles_col_b].dropna().unique())
    newly_cached = cache.preload_dataset_smiles(all_smi)
    print(f"FG Cache populated at {cfg.data.fg_cache_file} ({newly_cached} newly decomposed molecules).")

    print("\n[SUCCESS] Data preparation completed.")


if __name__ == "__main__":
    main()
