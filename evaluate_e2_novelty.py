"""
Chemical Novelty Analysis Script for CancerCombo E2 Baseline Evaluation.

Calculates chemical similarity between test drugs and training drugs:
    1. Morgan Fingerprint (ECFP4) Tanimoto Similarity (Nearest Neighbor)
    2. Bemis-Murcko Scaffold Similarity
    3. BRICS Fragment Overlap (Jaccard Similarity)

Stratifies Scenario 3 test performance into High, Medium, and Low novelty tiers.
Evaluation only — no test drug information is used during training.
"""

import os
import json
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from torch.utils.data import DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

import config
from cancer_combo_brics import (
    CancerComboBRICSSymmetric,
    load_cancer_combo_from_csv,
    collate_cancer_combo_batch,
    decompose_smiles_to_brics
)
from train_dgx import evaluate_full, compute_evaluation_metrics, move_batch_to_device


def get_morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold_mol)
    except Exception:
        return ""


def get_brics_set(smiles: str) -> Set[str]:
    try:
        return set(decompose_smiles_to_brics(smiles))
    except Exception:
        return set()


def analyze_test_drug_novelty(train_drugs: Set[str], test_drugs: Set[str]) -> Dict[str, dict]:
    print("=" * 75)
    print("  Calculating Test Drug Chemical Novelty vs Training Set")
    print("=" * 75)

    train_fps = {d: get_morgan_fp(d) for d in train_drugs if get_morgan_fp(d) is not None}
    train_scaffolds = {d: get_scaffold(d) for d in train_drugs}
    train_brics = {d: get_brics_set(d) for d in train_drugs}

    novelty_results = {}

    for test_d in test_drugs:
        test_fp = get_morgan_fp(test_d)
        test_scaff = get_scaffold(test_d)
        test_br = get_brics_set(test_d)

        best_tanimoto = 0.0
        best_tanimoto_match = ""

        best_scaff_sim = 0.0
        best_scaff_match = ""

        best_brics_sim = 0.0
        best_brics_match = ""

        for train_d, tr_fp in train_fps.items():
            # Tanimoto similarity
            if test_fp is not None and tr_fp is not None:
                sim = DataStructs.TanimotoSimilarity(test_fp, tr_fp)
                if sim > best_tanimoto:
                    best_tanimoto = sim
                    best_tanimoto_match = train_d

            # Scaffold similarity (exact match or FP Tanimoto on scaffold)
            tr_scaff = train_scaffolds[train_d]
            if test_scaff and tr_scaff:
                scaff_sim = 1.0 if test_scaff == tr_scaff else 0.0
                if scaff_sim > best_scaff_sim:
                    best_scaff_sim = scaff_sim
                    best_scaff_match = train_d

            # BRICS fragment overlap Jaccard similarity
            tr_br = train_brics[train_d]
            if test_br or tr_br:
                inter = len(test_br & tr_br)
                union = len(test_br | tr_br)
                jaccard = inter / max(union, 1)
                if jaccard > best_brics_sim:
                    best_brics_sim = jaccard
                    best_brics_match = train_d

        # Determine Tier
        if best_tanimoto >= 0.5:
            tier = "High similarity"
        elif best_tanimoto >= 0.3:
            tier = "Medium similarity"
        else:
            tier = "Low similarity"

        novelty_results[test_d] = {
            "nearest_train_drug": best_tanimoto_match,
            "tanimoto_similarity": float(best_tanimoto),
            "nearest_scaffold_drug": best_scaff_match,
            "scaffold_similarity": float(best_scaff_sim),
            "nearest_brics_drug": best_brics_match,
            "brics_overlap": float(best_brics_sim),
            "tier": tier
        }

        print(f"Test Drug: {test_d[:30]}...")
        print(f"  Nearest Train Drug: {best_tanimoto_match[:30]}...")
        print(f"  Tanimoto Sim: {best_tanimoto:.4f} | Scaffold Sim: {best_scaff_sim:.4f} | BRICS Overlap: {best_brics_sim:.4f} | Tier: {tier}")

    return novelty_results


def evaluate_stratified_performance(
    model: torch.nn.Module,
    test_dataset,
    novelty_results: Dict[str, dict],
    device: torch.device
) -> Dict[str, dict]:
    print("\n" + "=" * 75)
    print("  Evaluating Scenario 3 Performance Stratified by Chemical Novelty Tiers")
    print("=" * 75)

    tier_samples = defaultdict(list)
    criterion = torch.nn.MSELoss()

    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            smiles_a = sample["smiles_A"]
            smiles_b = sample["smiles_B"]

            sim_a = novelty_results.get(smiles_a, {}).get("tanimoto_similarity", 0.0)
            sim_b = novelty_results.get(smiles_b, {}).get("tanimoto_similarity", 0.0)

            # Pair similarity: min of both drugs determines pair novelty constraint
            pair_sim = min(sim_a, sim_b)
            if pair_sim >= 0.5:
                tier = "High similarity"
            elif pair_sim >= 0.3:
                tier = "Medium similarity"
            else:
                tier = "Low similarity"

            tier_samples[tier].append(sample)

    stratified_metrics = {}

    for tier in ["High similarity", "Medium similarity", "Low similarity"]:
        samples = tier_samples[tier]
        if not samples:
            print(f"\n[{tier}] No samples in tier.")
            stratified_metrics[tier] = {"count": 0}
            continue

        loader = DataLoader(
            samples,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_cancer_combo_batch
        )

        metrics = evaluate_full(model, loader, criterion, device)
        metrics["count"] = len(samples)
        stratified_metrics[tier] = metrics

        print(f"\n[{tier}] Samples: {len(samples)}")
        print(f"  MSE Loss : {metrics['loss']:.6f}")
        print(f"  RMSE     : {metrics['rmse']:.4f}")
        print(f"  MAE      : {metrics['mae']:.4f}")
        print(f"  R²       : {metrics['r2']:.4f}")
        print(f"  Pearson  : {metrics['pearson']:.4f}")
        print(f"  Spearman : {metrics['spearman']:.4f}")

    return stratified_metrics


def main():
    data_csv = config.DATA_CSV
    device = config.DEVICE

    print(f"Loading dataset splits from '{data_csv}'...")
    train_dataset = load_cancer_combo_from_csv(data_csv, split=config.TRAIN_SPLIT)
    test_dataset = load_cancer_combo_from_csv(data_csv, split=config.TEST_SPLIT)

    train_drugs = set()
    for sample in train_dataset:
        train_drugs.add(sample["smiles_A"])
        train_drugs.add(sample["smiles_B"])

    test_drugs = set()
    for sample in test_dataset:
        test_drugs.add(sample["smiles_A"])
        test_drugs.add(sample["smiles_B"])

    print(f"Unique Train Drugs: {len(train_drugs)}")
    print(f"Unique Test Drugs:  {len(test_drugs)}")

    novelty_results = analyze_test_drug_novelty(train_drugs, test_drugs)

    # Load Model
    if os.path.exists(config.BEST_MODEL_PATH):
        model = CancerComboBRICSSymmetric().to(device)
        checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        stratified_metrics = evaluate_stratified_performance(model, test_dataset, novelty_results, device)
    else:
        print(f"Model checkpoint '{config.BEST_MODEL_PATH}' not found. Skipping model evaluation.")
        stratified_metrics = {}

    # Save Analysis Report
    os.makedirs(config.LOG_DIR, exist_ok=True)
    report_path = os.path.join(config.LOG_DIR, "e2_chemical_novelty_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "novelty_per_test_drug": novelty_results,
            "stratified_performance": stratified_metrics
        }, f, indent=2)

    print(f"\nSaved Chemical Novelty Report to '{report_path}'.")
    print("=" * 75)


if __name__ == "__main__":
    main()
