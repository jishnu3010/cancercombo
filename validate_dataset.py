"""
Dataset Validation Command Line Diagnostic Script for CancerCombo.

Prints complete dataset summary:
    - Row count, unique drugs, unique cell lines
    - Dose grid shapes and concentration ranges
    - Expression matrix dimension (976 genes)
    - Leakage audit (Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅)
    - Canonical SMILES leakage check
"""

import os
import sys
from rdkit import Chem
import config
from cancer_combo_brics import load_cancer_combo_from_csv
from cancer_combo_brics.cell_expression import CellExpressionLoader


def canonicalize(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else smiles
    except Exception:
        return smiles


def main():
    print("=" * 75)
    print("  CancerCombo — Complete Dataset Diagnostic Audit")
    print("=" * 75)

    csv_path = config.DATA_CSV
    cell_csv = os.path.join("data", "cell_line_gene_expr.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset CSV '{csv_path}' not found!")
        sys.exit(1)

    # 1. Audit Cell Line Expression Source
    expr_loader = CellExpressionLoader(csv_path=cell_csv, gene_dim=config.GENE_DIM)
    expr_loader.print_summary()

    # 2. Load Datasets
    print(f"\n[1] Loading dataset splits from '{csv_path}'...")
    train_ds = load_cancer_combo_from_csv(csv_path, cell_expr_csv=cell_csv, split=config.TRAIN_SPLIT)
    val_ds = load_cancer_combo_from_csv(csv_path, cell_expr_csv=cell_csv, split=config.VAL_SPLIT)
    test_ds = load_cancer_combo_from_csv(csv_path, cell_expr_csv=cell_csv, split=config.TEST_SPLIT)

    def get_drugs_and_can(ds):
        raw_drugs, can_drugs = set(), set()
        for sample in ds:
            sa, sb = sample["smiles_A"], sample["smiles_B"]
            raw_drugs.add(sa)
            raw_drugs.add(sb)
            can_drugs.add(canonicalize(sa))
            can_drugs.add(canonicalize(sb))
        return raw_drugs, can_drugs

    tr_raw, tr_can = get_drugs_and_can(train_ds)
    val_raw, val_can = get_drugs_and_can(val_ds)
    te_raw, te_can = get_drugs_and_can(test_ds)

    print("\n[2] Drug Split & Leakage Audit:")
    print(f"  - Unique Train Drugs: {len(tr_raw)} (Canonical: {len(tr_can)})")
    print(f"  - Unique Val Drugs  : {len(val_raw)} (Canonical: {len(val_can)})")
    print(f"  - Unique Test Drugs : {len(te_raw)} (Canonical: {len(te_can)})")

    tv_overlap = len(tr_can & val_can)
    tt_overlap = len(tr_can & te_can)
    vt_overlap = len(val_can & te_can)

    print(f"  - Train AND Val Overlap : {tv_overlap}")
    print(f"  - Train AND Test Overlap: {tt_overlap}")
    print(f"  - Val AND Test Overlap  : {vt_overlap}")

    if tv_overlap > 0 or tt_overlap > 0 or vt_overlap > 0:
        print("\n[CRITICAL FAIL] Canonical SMILES Drug Leakage Detected across splits!")
        sys.exit(1)
    else:
        print("\n[PASS] ZERO DRUG LEAKAGE VERIFIED! Scenario 3 drug-disjoint condition holds.")

    print("=" * 75)


if __name__ == "__main__":
    main()
