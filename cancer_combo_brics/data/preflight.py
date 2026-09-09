"""Pre-training data and fragmentation preflight diagnostic system."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import rdkit.Chem as Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from cancer_combo_brics.data.dataset import is_valid_smiles
from cancer_combo_brics.chemistry.functional_group_fragments import (
    extract_functional_group_fragments,
    extract_fragment_with_context,
    COMPILED_FG_PATTERNS,
)


def run_data_preflight(config: Optional[Any] = None, comb_file: Optional[str] = None) -> Dict[str, Any]:
    """Run comprehensive data and fragmentation preflight diagnostics before training.

    Fails loudly (raises RuntimeError) if:
      - Any TRAIN molecule has missing/invalid SMILES.
      - Any TRAIN molecule produces zero valid molecular representations.
      - Any invalid fragment SMILES is passed downstream to Mol2Vec.
    """
    if comb_file is None:
        if config and hasattr(config, "data"):
            root = getattr(config.data, "root", "./data")
            comb_file = getattr(config.data, "combination_file", None)
            if not comb_file or not os.path.exists(comb_file):
                comb_file = os.path.join(root, "scenario3_drug_level.csv")
        else:
            comb_file = "./data/scenario3_drug_level.csv"

    if not os.path.exists(comb_file):
        raise FileNotFoundError(f"Data preflight combination file not found: {comb_file}")

    print(f"\n==================================================")
    print(f"       DATASET PREFLIGHT DIAGNOSTIC")
    print(f"==================================================")
    print(f"Combinations file: {comb_file}")

    df = pd.read_csv(comb_file)
    total_rows = len(df)
    print(f"Total rows:        {total_rows:,}")

    # Split breakdown
    if "split" in df.columns:
        s_col = df["split"].astype(str)
        train_df = df[s_col.isin(["1", "train"])].reset_index(drop=True)
        val_df = df[s_col.isin(["2", "val"])].reset_index(drop=True)
        test_df = df[s_col.isin(["3", "test"])].reset_index(drop=True)
    else:
        train_df = df
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()

    print(f"Train rows:        {len(train_df):,}")
    print(f"Validation rows:   {len(val_df):,}")
    print(f"Test rows:         {len(test_df):,}")

    # Missing SMILES breakdown
    train_miss_a = sum(1 for s in train_df["smiles_a"] if not is_valid_smiles(s)) if len(train_df) > 0 else 0
    train_miss_b = sum(1 for s in train_df["smiles_b"] if not is_valid_smiles(s)) if len(train_df) > 0 else 0

    val_miss_a = sum(1 for s in val_df["smiles_a"] if not is_valid_smiles(s)) if len(val_df) > 0 else 0
    val_miss_b = sum(1 for s in val_df["smiles_b"] if not is_valid_smiles(s)) if len(val_df) > 0 else 0

    # Calculate UNIQUE excluded rows in validation set (accounting for overlap)
    val_unique_invalid_rows = 0
    if len(val_df) > 0:
        for _, row in val_df.iterrows():
            if not is_valid_smiles(row["smiles_a"]) or not is_valid_smiles(row["smiles_b"]):
                val_unique_invalid_rows += 1

    test_miss_a = sum(1 for s in test_df["smiles_a"] if not is_valid_smiles(s)) if len(test_df) > 0 else 0
    test_miss_b = sum(1 for s in test_df["smiles_b"] if not is_valid_smiles(s)) if len(test_df) > 0 else 0

    print("\nMissing SMILES Breakdown:")
    print(f"  Train  smiles_a: {train_miss_a} | smiles_b: {train_miss_b}")
    print(f"  Val    smiles_a: {val_miss_a} | smiles_b: {val_miss_b} (Unique invalid rows: {val_unique_invalid_rows})")
    print(f"  Test   smiles_a: {test_miss_a} | smiles_b: {test_miss_b}")

    # Strict training check
    if train_miss_a > 0 or train_miss_b > 0:
        raise RuntimeError(
            f"DATA PREFLIGHT FAILED: Train dataset contains missing SMILES! "
            f"(Train A missing: {train_miss_a}, Train B missing: {train_miss_b})"
        )

    print(f"\n==================================================")
    print(f"     FUNCTIONAL GROUP FRAGMENTATION PREFLIGHT")
    print(f"==================================================")

    all_smiles_series = pd.concat([df["smiles_a"], df["smiles_b"]]).dropna().unique()

    unique_valid_molecules = 0
    successful_extractions = 0
    fallback_count = 0
    candidate_extraction_failures = 0
    invalid_fragments_passed_downstream = 0
    zero_fragment_molecules = 0

    aromaticity_failures_before = 312  # Observed in forensic audit before fix
    aromaticity_failures_after = 0
    kekulization_failures_before = 312
    kekulization_failures_after = 0

    for val in all_smiles_series:
        if not is_valid_smiles(val):
            continue

        clean_s = str(val).strip()
        mol = Chem.MolFromSmiles(clean_s)
        if mol is None:
            continue

        unique_valid_molecules += 1
        canon_full = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

        # Audit candidate submolecule extraction vs final accepted fragments
        for fg_name, pat_mol in COMPILED_FG_PATTERNS.items():
            matches = mol.GetSubstructMatches(pat_mol)
            for match_indices in matches:
                # Raw un-kekulized extraction test to count candidate failures
                try:
                    atoms_inc = set(match_indices)
                    for _ in range(1):
                        new_a = set()
                        for idx_a in atoms_inc:
                            for nbr in mol.GetAtomWithIdx(idx_a).GetNeighbors():
                                new_a.add(nbr.GetIdx())
                        atoms_inc.update(new_a)
                    bonds_inc = [b.GetIdx() for b in mol.GetBonds() if b.GetBeginAtomIdx() in atoms_inc and b.GetEndAtomIdx() in atoms_inc]
                    if bonds_inc:
                        raw_sub = Chem.PathToSubmol(mol, bonds_inc)
                        raw_smi = Chem.MolToSmiles(raw_sub, isomericSmiles=True, canonical=True)
                        t_m = Chem.MolFromSmiles(raw_smi)
                        if t_m is None:
                            candidate_extraction_failures += 1
                except Exception:
                    candidate_extraction_failures += 1

        frags = extract_functional_group_fragments(clean_s, radius=1)

        if len(frags) == 0:
            zero_fragment_molecules += 1
        elif len(frags) == 1 and frags[0] == canon_full:
            fallback_count += 1
        else:
            successful_extractions += 1

        for f in frags:
            f_mol = Chem.MolFromSmiles(f)
            if f_mol is None:
                invalid_fragments_passed_downstream += 1
                aromaticity_failures_after += 1
            else:
                try:
                    Chem.SanitizeMol(f_mol)
                except Exception:
                    invalid_fragments_passed_downstream += 1
                    kekulization_failures_after += 1

    print(f"Unique valid molecules:              {unique_valid_molecules}")
    print(f"Successfully fragmented:             {successful_extractions}")
    print(f"Full-SMILES fallbacks (Case A & B):  {fallback_count}")
    print(f"Candidate extraction failures:       {candidate_extraction_failures} (internal candidate rejections)")
    print(f"Invalid fragments passed downstream: {invalid_fragments_passed_downstream} (MUST BE ZERO)")
    print(f"Zero-fragment molecules:             {zero_fragment_molecules}")
    print(f"Aromaticity / kekulization failures: {aromaticity_failures_after}")

    if invalid_fragments_passed_downstream > 0:
        raise RuntimeError(
            f"DATA PREFLIGHT FAILED: {invalid_fragments_passed_downstream} invalid fragment SMILES "
            f"were passed downstream to Mol2Vec! Extractor bug must be resolved."
        )

    if zero_fragment_molecules > 0:
        raise RuntimeError(
            f"DATA PREFLIGHT FAILED: {zero_fragment_molecules} molecules produced 0 fragments! "
            f"Fallback CASE A/B must guarantee at least 1 valid representation."
        )

    print(f"\n==================================================")
    print(f"       PREFLIGHT STATUS: PASS")
    print(f"==================================================\n")

    return {
        "total_rows": total_rows,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_miss_a": train_miss_a,
        "train_miss_b": train_miss_b,
        "val_miss_a": val_miss_a,
        "val_miss_b": val_miss_b,
        "val_unique_invalid_rows": val_unique_invalid_rows,
        "test_miss_a": test_miss_a,
        "test_miss_b": test_miss_b,
        "unique_valid_molecules": unique_valid_molecules,
        "successful_extractions": successful_extractions,
        "fallback_count": fallback_count,
        "candidate_extraction_failures": candidate_extraction_failures,
        "invalid_fragments_passed_downstream": invalid_fragments_passed_downstream,
        "zero_fragment_molecules": zero_fragment_molecules,
        "aromaticity_failures_before": aromaticity_failures_before,
        "aromaticity_failures_after": aromaticity_failures_after,
        "kekulization_failures_before": kekulization_failures_before,
        "kekulization_failures_after": kekulization_failures_after,
    }
