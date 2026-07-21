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


def precompute_drug_features(
    smiles_list: List[str],
    morgan_bits: int = 2048,
    n_descriptors: int = 200,
    max_length: int = 128
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Precompute Morgan fingerprints, descriptors, and SMILES token IDs for all unique SMILES.

    Args:
        smiles_list: List of unique SMILES strings.
        morgan_bits: Bit vector length (default=2048).
        n_descriptors: Number of continuous physical descriptors (default=200).
        max_length: Token sequence max length (default=128).

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: Feature dictionary keyed by SMILES string.
    """
    preprocessor = MolecularPreprocessor(morgan_nbits=morgan_bits, morgan_radius=2)
    tokenizer = SMILESTokenizer(max_len=max_length)
    
    feature_store: Dict[str, Dict[str, torch.Tensor]] = {}
    descriptor_matrix = []
    
    logger.info("Computing RDKit Morgan fingerprints, descriptors, and SMILES tokens...")
    for idx, smiles in enumerate(smiles_list):
        if (idx + 1) % 500 == 0 or idx == len(smiles_list) - 1:
            logger.info(f"Processing SMILES [{idx + 1}/{len(smiles_list)}]...")
            
        morgan, desc, _ = preprocessor.process_smiles(smiles)
        morgan = torch.from_numpy(morgan) if isinstance(morgan, np.ndarray) else morgan
        desc = torch.from_numpy(desc) if isinstance(desc, np.ndarray) else desc
        ids, mask = tokenizer.tokenize(smiles)
        ids = torch.tensor(ids, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        feature_store[smiles] = {
            "morgan": morgan.cpu(),
            "descriptors": desc.cpu(),
            "token_ids": ids.cpu(),
            "token_mask": mask.cpu()
        }
        descriptor_matrix.append(desc.numpy())
        
    # Apply global Z-score normalization to continuous descriptors
    if descriptor_matrix:
        desc_arr = np.array(descriptor_matrix) # Shape: (N, 200)
        mean = np.mean(desc_arr, axis=0)
        std = np.std(desc_arr, axis=0)
        std[std == 0] = 1.0 # Protect against zero variance
        
        logger.info("Applying global Z-score normalization to physical descriptors...")
        for smiles in feature_store:
            raw_desc = feature_store[smiles]["descriptors"].numpy()
            norm_desc = (raw_desc - mean) / std
            feature_store[smiles]["descriptors"] = torch.tensor(norm_desc, dtype=torch.float32)

    return feature_store


def main():
    parser = argparse.ArgumentParser(description="Precompute and Save Molecular Features for CancerCombo")
    parser.add_argument("--input_csv", type=str, default="data/DrugCombination_with_SMILES.zip", help="Path to input CSV/ZIP.")
    parser.add_argument("--output_file", type=str, default="data/features/drug_features.pt", help="Output .pt feature file.")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    smiles_list = extract_unique_smiles(args.input_csv)
    if not smiles_list:
        logger.warning("No SMILES found. Using mock SMILES list for demonstration...")
        smiles_list = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34"]
        
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
