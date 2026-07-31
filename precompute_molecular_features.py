#!/usr/bin/env python3
"""
precompute_molecular_features.py - Precomputes and Saves Molecular Features for CancerCombo

Generates static pre-extracted feature dictionaries for all unique SMILES in a dataset:
  1. Morgan Fingerprints (2048-bit bit vector)
  2. Physical Descriptors (200 continuous RDKit descriptors, Z-score normalized)
  3. Tokenized SMILES sequences (Token IDs & Attention Masks for MolFormer)

Outputs saved to PyTorch checkpoint format (.pt) and Pickle (.pkl) for instant O(1) dataset loading.

Usage:
    python precompute_molecular_features.py --input_csv data/DrugCombination_with_SMILES.zip --output_file data/features/drug_features.pt
"""

import os
for _k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "1"

from helpers import enforce_single_thread
enforce_single_thread()

import sys
import argparse
import logging
import zipfile
import pickle
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from preprocessor import MolecularPreprocessor
from dataset import SMILESTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FeaturePrecomputer")


def extract_unique_smiles(input_path: str) -> List[str]:
    """Extract all unique SMILES strings from input CSV or ZIP archive.

    Args:
        input_path: Path to CSV or ZIP archive.

    Returns:
        List[str]: Sorted list of unique SMILES strings.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: '{input_path}'")
        
    if input_path.endswith(".zip"):
        with zipfile.ZipFile(input_path, "r") as z:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_files:
                raise ValueError("No CSV found in ZIP archive.")
            dfs = []
            for csv_file in csv_files:
                with z.open(csv_file) as f:
                    dfs.append(pd.read_csv(f))
            df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(input_path)
        
    smiles_set = set()
    for col in ["Drug1_SMILES", "Drug2_SMILES", "smiles_a", "smiles_b", "smiles1", "smiles2", "SMILES_A", "SMILES_B"]:
        if col in df.columns:
            smiles_set.update(df[col].dropna().astype(str).str.strip().tolist())
            
    unique_smiles = sorted(list(smiles_set))
    logger.info(f"Extracted {len(unique_smiles)} unique SMILES strings from '{input_path}'.")
    return unique_smiles


# ============================================================
# ABLATION 1 - MOLFORMER ONLY
# DISABLED FOR ABLATION 2
# ============================================================
# from blocks.molformer_encoder import MolFormerEncoder
#
# def precompute_drug_features(
#     smiles_list: List[str],
#     max_length: int = 128,
#     d_model: int = 256,
#     batch_size: int = 64,
#     device: str = "cpu"
# ) -> Dict[str, Dict[str, torch.Tensor]]:
#     tokenizer = SMILESTokenizer(max_len=max_length)
#     molformer_encoder = MolFormerEncoder(d_model=d_model, vocab_size=100)
#     ...

# ============================================================
# ABLATION 2 - MORGAN + RDKIT DESCRIPTORS ONLY
# ACTIVE
# ============================================================
def precompute_drug_features(
    smiles_list: List[str],
    morgan_bits: int = 2048,
    n_descriptors: int = 200
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Precompute Morgan fingerprints (2048-bit) and RDKit descriptors (200 continuous) for all unique SMILES.

    Args:
        smiles_list: List of unique SMILES strings.
        morgan_bits: Dimension of Morgan fingerprint bit vector (default=2048).
        n_descriptors: Number of physical descriptors (default=200).

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Feature dictionary keyed by SMILES string containing 'morgan' and 'descriptors'.
    """
    preprocessor = MolecularPreprocessor(morgan_nbits=morgan_bits, morgan_radius=2)
    feature_store: Dict[str, Dict[str, torch.Tensor]] = {}

    logger.info(f"Computing Morgan ({morgan_bits}-bit) + RDKit Descriptors ({n_descriptors}) for {len(smiles_list)} unique SMILES...")

    for idx, smiles in enumerate(smiles_list):
        if (idx + 1) % 500 == 0 or (idx + 1) == len(smiles_list):
            logger.info(f"Processing SMILES [{idx + 1}/{len(smiles_list)}]...")

        morgan, desc, _ = preprocessor.process_smiles(smiles)
        morgan_t = torch.from_numpy(morgan).float() if isinstance(morgan, np.ndarray) else morgan.float()
        desc_t = torch.from_numpy(desc).float() if isinstance(desc, np.ndarray) else desc.float()

        feature_store[smiles] = {
            "morgan": morgan_t,
            "descriptors": desc_t
        }

    return feature_store


def main():
    parser = argparse.ArgumentParser(description="Precompute and Save Morgan + RDKit Molecular Features for CancerCombo (Ablation 2)")
    parser.add_argument("--input_csv", type=str, default="data/scenario1_combination_50k.csv", help="Path to input CSV/ZIP.")
    parser.add_argument("--output_file", type=str, default="data/features/morgan_rdkit_only/drug_features_morgan_rdkit.pt", help="Output .pt feature file.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    smiles_list = extract_unique_smiles(args.input_csv)
    if not smiles_list:
        logger.warning("No SMILES found.")
        return

    feature_store = precompute_drug_features(smiles_list)

    # Save as PyTorch .pt file
    torch.save(feature_store, args.output_file)
    logger.info(f"Successfully saved PyTorch feature store to '{args.output_file}' ({len(feature_store)} unique drugs).")

    # Save as Pickle .pkl file
    pkl_file = args.output_file.replace(".pt", ".pkl")
    with open(pkl_file, "wb") as f:
        pickle.dump(feature_store, f)
    logger.info(f"Successfully saved Pickle feature store to '{pkl_file}'.")


if __name__ == "__main__":
    main()


