"""Drug-disjoint dataset splitting and unseen-drug scenario categorization."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def check_drug_overlap(
    train_drugs: Set[str],
    val_drugs: Set[str],
    test_drugs: Set[str],
) -> Dict[str, int]:
    """Verify strict drug-disjointness between splits."""
    train_val_overlap = len(train_drugs.intersection(val_drugs))
    train_test_overlap = len(train_drugs.intersection(test_drugs))
    val_test_overlap = len(val_drugs.intersection(test_drugs))

    overlap_summary = {
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "is_strictly_disjoint": int(train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0),
    }

    print("\n--- DRUG OVERLAP VERIFICATION ---")
    print(f"Unique drugs in train: {len(train_drugs)}")
    print(f"Unique drugs in val:   {len(val_drugs)}")
    print(f"Unique drugs in test:  {len(test_drugs)}")
    print(f"Train & Val overlap:   {train_val_overlap}")
    print(f"Train & Test overlap:  {train_test_overlap}")
    print(f"Val & Test overlap:    {val_test_overlap}")
    if overlap_summary["is_strictly_disjoint"]:
        print("[PASS] Strict drug-disjoint condition satisfied (Drugs(train) & Drugs(test) = empty).")
    else:
        print("[WARNING] Overlap detected across splits!")
    print("---------------------------------\n")

    return overlap_summary


def categorize_scenario(
    drug_a: str,
    drug_b: str,
    train_drugs: Set[str],
) -> int:
    """Classify a drug combination into generalization scenarios.

    Scenario 1: Both drugs seen in training.
    Scenario 2: One drug seen + one drug unseen.
    Scenario 3: Both drugs unseen in training.
    """
    a_seen = drug_a in train_drugs
    b_seen = drug_b in train_drugs

    if a_seen and b_seen:
        return 1
    elif a_seen or b_seen:
        return 2
    else:
        return 3


def create_drug_disjoint_splits(
    df: pd.DataFrame,
    drug_col_a: str = "drug_a",
    drug_col_b: str = "drug_b",
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a strict drug-level disjoint split on combinations DataFrame.

    Partitions unique drugs into disjoint sets: TrainDrugs, ValDrugs, TestDrugs.
    Combinations where both drugs are in TestDrugs -> test set (Scenario 3 pure).
    Combinations where both drugs are in ValDrugs -> val set.
    Combinations where both drugs are in TrainDrugs -> train set (Scenario 1).
    Other pairs are categorized and assigned to appropriate evaluation splits.

    Args:
        df: Input combinations DataFrame.
        drug_col_a: Column name for Drug A identifier or SMILES.
        drug_col_b: Column name for Drug B identifier or SMILES.
        test_ratio: Desired fraction of drugs for test set.
        val_ratio: Desired fraction of drugs for validation set.
        seed: Random seed.

    Returns:
        DataFrame with added 'split' and 'scenario' columns.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Extract all unique drugs across drug_a and drug_b
    unique_drugs = sorted(list(set(df[drug_col_a].dropna().unique()) | set(df[drug_col_b].dropna().unique())))
    num_drugs = len(unique_drugs)

    # Shuffle drugs
    shuffled = rng.permutation(unique_drugs)
    num_test = max(2, int(num_drugs * test_ratio)) if num_drugs >= 6 else max(1, int(num_drugs * test_ratio))
    num_val = max(2, int(num_drugs * val_ratio)) if num_drugs >= 6 else max(1, int(num_drugs * val_ratio))

    test_drugs = set(shuffled[:num_test])
    val_drugs = set(shuffled[num_test : num_test + num_val])
    train_drugs = set(shuffled[num_test + num_val:])

    # Assign scenarios relative to train_drugs
    df["scenario"] = [
        categorize_scenario(row[drug_col_a], row[drug_col_b], train_drugs)
        for _, row in df.iterrows()
    ]

    # Assign strict drug-disjoint splits
    def assign_split(row):
        da = row[drug_col_a]
        db = row[drug_col_b]
        if da in test_drugs and db in test_drugs:
            return "test"
        elif da in val_drugs and db in val_drugs:
            return "val"
        elif da in train_drugs and db in train_drugs:
            return "train"
        elif (da in test_drugs) or (db in test_drugs):
            return "test_scenario_2"
        elif (da in val_drugs) or (db in val_drugs):
            return "val_scenario_2"
        else:
            return "train"

    df["split"] = df.apply(assign_split, axis=1)

    # Verification on strict drug-disjoint splits (train, val, test)
    actual_train_drugs = set(df[df["split"] == "train"][drug_col_a]) | set(df[df["split"] == "train"][drug_col_b])
    actual_val_drugs = set(df[df["split"] == "val"][drug_col_a]) | set(df[df["split"] == "val"][drug_col_b])
    actual_test_drugs = set(df[df["split"] == "test"][drug_col_a]) | set(df[df["split"] == "test"][drug_col_b])

    check_drug_overlap(actual_train_drugs, actual_val_drugs, actual_test_drugs)

    print(f"Split distribution: {df['split'].value_counts().to_dict()}")
    print(f"Scenario distribution: {df['scenario'].value_counts().to_dict()}")
    return df


def validate_existing_splits(
    df: pd.DataFrame,
    split_col: str = "split",
    drug_col_a: str = "drug_a",
    drug_col_b: str = "drug_b",
) -> Dict[str, Any]:
    """Validate an existing split column in the dataset."""
    if split_col not in df.columns:
        raise ValueError(f"Split column '{split_col}' not found in dataframe.")

    splits = df[split_col].unique()
    train_drugs = set(df[df[split_col] == "train"][drug_col_a]) | set(df[df[split_col] == "train"][drug_col_b])
    val_drugs = set(df[df[split_col] == "val"][drug_col_a]) | set(df[df[split_col] == "val"][drug_col_b]) if "val" in splits else set()
    test_drugs = set(df[df[split_col] == "test"][drug_col_a]) | set(df[df[split_col] == "test"][drug_col_b]) if "test" in splits else set()

    overlap = check_drug_overlap(train_drugs, val_drugs, test_drugs)

    # Compute scenario distribution
    df["scenario"] = [
        categorize_scenario(row[drug_col_a], row[drug_col_b], train_drugs)
        for _, row in df.iterrows()
    ]

    return {
        "splits": df[split_col].value_counts().to_dict(),
        "scenarios": df["scenario"].value_counts().to_dict(),
        "overlap": overlap,
    }
