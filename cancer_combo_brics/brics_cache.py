"""
Deterministic BRICS Feature Precomputation and Caching Module for CancerCombo.

Avoids repeated dynamic RDKit BRICS decomposition and Morgan fingerprint generation
during training epochs by precomputing and caching unique drug features ONCE.

Cache Data Structure:
    canonical_smiles -> {
        "brics_fragments": List[str],
        "fingerprints": np.ndarray (N_frags, 2048),
        "num_fragments": int
    }
"""

import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

from .brics_utils import decompose_smiles_to_brics, fragment_to_morgan_fp


class BRICSCache:
    """
    In-memory and persistent disk cache for BRICS fragment decomposition and Morgan fingerprints.
    """

    def __init__(self, cache_file: Optional[str] = os.path.join("data", "brics_cache.json"), n_bits: int = 2048):
        self.cache_file = cache_file
        self.n_bits = n_bits

        # In-memory lookup tables
        self.smiles_to_frags: Dict[str, List[str]] = {}
        self.smiles_to_fps: Dict[str, np.ndarray] = {}

        if cache_file and os.path.exists(cache_file):
            self.load_cache(cache_file)

    def canonicalize_smiles(self, smiles: str) -> str:
        if not smiles or not isinstance(smiles, str) or smiles.lower() in ("nan", "none", "null"):
            raise ValueError(f"Invalid SMILES string encountered: '{smiles}'")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed to parse invalid SMILES string: '{smiles}'")
        return Chem.MolToSmiles(mol, canonical=True)

    def get_or_compute_brics(self, smiles: str) -> Tuple[List[str], np.ndarray]:
        """
        Retrieves cached BRICS fragments and Morgan fingerprints for a SMILES string,
        or computes and caches them on first occurrence.
        """
        can_smi = self.canonicalize_smiles(smiles)

        if can_smi in self.smiles_to_frags:
            return self.smiles_to_frags[can_smi], self.smiles_to_fps[can_smi]

        # Compute BRICS decomposition and Morgan fingerprints
        frags = decompose_smiles_to_brics(can_smi)
        fps_list = []
        for f in frags:
            fp_arr = fragment_to_morgan_fp(f, n_bits=self.n_bits)
            fps_list.append(fp_arr)

        fps_matrix = np.array(fps_list, dtype=np.float32)

        # Store in cache
        self.smiles_to_frags[can_smi] = frags
        self.smiles_to_fps[can_smi] = fps_matrix

        return frags, fps_matrix

    def precompute_dataset_drugs(self, smiles_list: List[str]):
        """Precomputes and caches BRICS features for all unique SMILES in dataset."""
        unique_smiles = set(smiles_list)
        count = 0
        for smi in unique_smiles:
            try:
                self.get_or_compute_brics(smi)
                count += 1
            except Exception as e:
                print(f"Warning: BRICSCache failed to process SMILES '{smi}': {e}")
        print(f"BRICSCache: Precomputed and cached features for {count} unique drugs.")
        if self.cache_file:
            self.save_cache(self.cache_file)

    def save_cache(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        serializable_data = {}
        for smi, frags in self.smiles_to_frags.items():
            fps_matrix = self.smiles_to_fps[smi]
            serializable_data[smi] = {
                "brics_fragments": frags,
                "fingerprints": fps_matrix.tolist()
            }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f)
        print(f"Saved BRICSCache to '{file_path}'.")

    def load_cache(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for smi, content in data.items():
            self.smiles_to_frags[smi] = content["brics_fragments"]
            self.smiles_to_fps[smi] = np.array(content["fingerprints"], dtype=np.float32)
        print(f"Loaded {len(self.smiles_to_frags)} cached drug entries from '{file_path}'.")
