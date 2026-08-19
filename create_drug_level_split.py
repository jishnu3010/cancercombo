"""
Drug-Level Disjoint Dataset Split Generator for CancerCombo Scenario 3.

Reads data/scenario3_drug1.csv, validates SMILES strings using RDKit,
partitions unique drugs into disjoint sets (Train, Val, Test), and creates
data/scenario3_drug_level.csv where:
    - Train drugs ∩ Val drugs = ∅
    - Train drugs ∩ Test drugs = ∅
    - Val drugs ∩ Test drugs = ∅

Outputs a complete split report including drug counts, combination counts,
and invalid SMILES counts.
"""

import os
import csv
import random
from collections import defaultdict
from typing import Set, Dict, List, Tuple

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def validate_and_parse_csv(
    csv_path: str
) -> Tuple[List[dict], Set[str], Dict[str, int]]:
    """
    Parses dataset CSV and audits SMILES validity.

    Returns:
        valid_rows: List of valid row dicts.
        valid_drugs: Set of valid SMILES strings.
        report: Dict containing total, valid, invalid counts and percentage.
    """
    total_molecules_checked = 0
    invalid_smiles_set = set()
    valid_smiles_set = set()
    valid_rows = []

    smiles_cache = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_a = str(row.get("smiles_a") or row.get("smiles_A") or "").strip()
            smiles_b = str(row.get("smiles_b") or row.get("smiles_B") or "").strip()

            # Check molecule A and B
            for smiles in (smiles_a, smiles_b):
                if not smiles or smiles.lower() in ("nan", "none", "null"):
                    invalid_smiles_set.add(smiles)
                    continue

                if smiles not in smiles_cache:
                    total_molecules_checked += 1
                    if RDKIT_AVAILABLE:
                        mol = Chem.MolFromSmiles(smiles)
                        smiles_cache[smiles] = mol is not None
                    else:
                        smiles_cache[smiles] = True

                if smiles_cache[smiles]:
                    valid_smiles_set.add(smiles)
                else:
                    invalid_smiles_set.add(smiles)

            # Skip row if either drug is invalid
            valid_a = smiles_a and smiles_cache.get(smiles_a, False)
            valid_b = smiles_b and smiles_cache.get(smiles_b, False)
            if not valid_a or not valid_b:
                continue

            valid_rows.append(row)

    total_unique = len(valid_smiles_set | invalid_smiles_set)
    n_invalid = len(invalid_smiles_set)
    n_valid = len(valid_smiles_set)
    invalid_pct = (n_invalid / max(total_unique, 1)) * 100.0

    report = {
        "total_molecules": total_unique,
        "valid_molecules": n_valid,
        "invalid_molecules": n_invalid,
        "invalid_percentage": invalid_pct,
        "valid_rows_count": len(valid_rows)
    }

    return valid_rows, valid_smiles_set, report


def generate_drug_level_split(
    input_csv: str = "data/scenario3_drug1.csv",
    output_csv: str = "data/scenario3_drug_level.csv",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
):
    print("=" * 75)
    print("  CancerCombo — Scenario 3 Drug-Level Split Generator & Audit")
    print("=" * 75)

    valid_rows, valid_drugs, audit_report = validate_and_parse_csv(input_csv)

    print("\n--- PREPROCESSING SMILES VALIDATION REPORT ---")
    print(f"Total Unique Molecules Checked : {audit_report['total_molecules']}")
    print(f"Valid Unique Molecules          : {audit_report['valid_molecules']}")
    print(f"Invalid Unique Molecules        : {audit_report['invalid_molecules']}")
    print(f"Invalid Percentage              : {audit_report['invalid_percentage']:.2f}%")
    print(f"Valid Dataset Rows              : {audit_report['valid_rows_count']}")

    sorted_drugs = sorted(list(valid_drugs))
    rng = random.Random(seed)
    rng.shuffle(sorted_drugs)

    n_total_drugs = len(sorted_drugs)
    n_train_drugs = int(n_total_drugs * train_ratio)
    n_val_drugs = int(n_total_drugs * val_ratio)

    train_drugs = set(sorted_drugs[:n_train_drugs])
    val_drugs = set(sorted_drugs[n_train_drugs:n_train_drugs + n_val_drugs])
    test_drugs = set(sorted_drugs[n_train_drugs + n_val_drugs:])

    # Assert strict disjointness
    assert train_drugs.isdisjoint(val_drugs), "Train and Val drugs overlap!"
    assert train_drugs.isdisjoint(test_drugs), "Train and Test drugs overlap!"
    assert val_drugs.isdisjoint(test_drugs), "Val and Test drugs overlap!"

    # Categorize rows into splits according to standard CancerCombo convention:
    # 1 = Train, 2 = Val, 3 = Test
    train_rows = []
    val_rows = []
    test_rows = []

    for row in valid_rows:
        sa = str(row.get("smiles_a") or row.get("smiles_A") or "").strip()
        sb = str(row.get("smiles_b") or row.get("smiles_B") or "").strip()

        row_copy = dict(row)
        if sa in train_drugs and sb in train_drugs:
            row_copy["split"] = "1"
            train_rows.append(row_copy)
        elif sa in val_drugs and sb in val_drugs:
            row_copy["split"] = "2"
            val_rows.append(row_copy)
        elif sa in test_drugs and sb in test_drugs:
            row_copy["split"] = "3"
            test_rows.append(row_copy)

    output_rows = train_rows + val_rows + test_rows

    # Count unique combinations
    def count_combos(rows):
        combos = set()
        for r in rows:
            sa = str(r.get("smiles_a") or r.get("smiles_A") or "").strip()
            sb = str(r.get("smiles_b") or r.get("smiles_B") or "").strip()
            combos.add(tuple(sorted([sa, sb])))
        return len(combos)

    print("\n--- DRUG-LEVEL SCENARIO 3 SPLIT REPORT ---")
    print(f"Number of Train Drugs        : {len(train_drugs)}")
    print(f"Number of Validation Drugs   : {len(val_drugs)}")
    print(f"Number of Test Drugs         : {len(test_drugs)}")
    print(f"Train Combinations           : {count_combos(train_rows)} (Rows: {len(train_rows)})")
    print(f"Validation Combinations      : {count_combos(val_rows)} (Rows: {len(val_rows)})")
    print(f"Test Combinations            : {count_combos(test_rows)} (Rows: {len(test_rows)})")
    print(f"Train & Val Drug Overlap     : {len(train_drugs & val_drugs)}")
    print(f"Train & Test Drug Overlap    : {len(train_drugs & test_drugs)}")
    print(f"Val & Test Drug Overlap      : {len(val_drugs & test_drugs)}")

    # Save to output CSV
    fieldnames = list(valid_rows[0].keys())
    if "split" not in fieldnames:
        fieldnames.append("split")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nSaved new drug-level disjoint split to '{output_csv}'.")
    print("=" * 75)


if __name__ == "__main__":
    generate_drug_level_split()
