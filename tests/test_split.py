"""Unit tests for drug-disjoint dataset splitting and scenario categorization."""

import pandas as pd
from cancer_combo_brics.data.splitting import create_drug_disjoint_splits, check_drug_overlap, categorize_scenario


def test_drug_disjoint_splitting():
    # Synthetic DataFrame of combinations
    records = []
    drugs = [f"Drug_{i}" for i in range(10)]
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            records.append({
                "drug_a": drugs[i],
                "drug_b": drugs[j],
                "cell_line": "MCF7",
            })
    df = pd.DataFrame(records)

    split_df = create_drug_disjoint_splits(df, test_ratio=0.2, val_ratio=0.2, seed=42)

    assert "split" in split_df.columns
    assert "scenario" in split_df.columns

    train_drugs = set(split_df[split_df["split"] == "train"]["drug_a"]) | set(split_df[split_df["split"] == "train"]["drug_b"])
    val_drugs = set(split_df[split_df["split"] == "val"]["drug_a"]) | set(split_df[split_df["split"] == "val"]["drug_b"])
    test_drugs = set(split_df[split_df["split"] == "test"]["drug_a"]) | set(split_df[split_df["split"] == "test"]["drug_b"])

    overlap = check_drug_overlap(train_drugs, val_drugs, test_drugs)
    assert overlap["is_strictly_disjoint"] == 1
    assert overlap["train_test_overlap"] == 0
    assert overlap["train_val_overlap"] == 0
    assert overlap["val_test_overlap"] == 0


def test_categorize_scenario():
    train_drugs = {"Drug_A", "Drug_B"}

    # Both seen -> Scenario 1
    assert categorize_scenario("Drug_A", "Drug_B", train_drugs) == 1

    # One seen -> Scenario 2
    assert categorize_scenario("Drug_A", "Drug_C", train_drugs) == 2

    # Both unseen -> Scenario 3
    assert categorize_scenario("Drug_C", "Drug_D", train_drugs) == 3
