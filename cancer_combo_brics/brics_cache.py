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
import threading
from typing import Dict, List, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

from .brics_utils import decompose_smiles_to_brics, fragment_to_morgan_fp


class BRICSCache:
    """
    In-memory and persistent disk cache for BRICS fragment decomposition and Morgan fingerprints.
    Tracks lookup statistics and provides thread-safe access across batch loading.
    """

    def __init__(
        self,
        cache_file: Optional[str] = os.path.join("data", "brics_cache.json"),
        n_bits: int = 2048
    ):
        self.cache_file = cache_file
        self.n_bits = n_bits

        # In-memory lookup tables
        self.smiles_to_frags: Dict[str, List[str]] = {}
        self.smiles_to_fps: Dict[str, np.ndarray] = {}

        # Thread safety lock
        self._lock = threading.Lock()

        # Cache statistics instrumentation
        self.requests = 0
        self.hits = 0
        self.misses = 0

        if cache_file and os.path.exists(cache_file):
            self.load_cache(cache_file)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    @property
    def lock(self) -> threading.Lock:
        if not hasattr(self, "_lock") or self._lock is None:
            self._lock = threading.Lock()
        return self._lock

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
        with self.lock:
            self.requests += 1

            # Fast path 1: Direct exact SMILES match in cache
            if smiles in self.smiles_to_frags:
                self.hits += 1
                return self.smiles_to_frags[smiles], self.smiles_to_fps[smiles]

        # Canonicalize SMILES
        can_smi = self.canonicalize_smiles(smiles)

        with self.lock:
            # Fast path 2: Canonical SMILES match in cache
            if can_smi in self.smiles_to_frags:
                frags = self.smiles_to_frags[can_smi]
                fps = self.smiles_to_fps[can_smi]
                # Alias raw SMILES to canonical entry for future fast path 1 hits
                self.smiles_to_frags[smiles] = frags
                self.smiles_to_fps[smiles] = fps
                self.hits += 1
                return frags, fps

            self.misses += 1

        # Compute BRICS decomposition and Morgan fingerprints (uncached miss)
        frags = decompose_smiles_to_brics(can_smi)
        fps_list = []
        for f in frags:
            fp_arr = fragment_to_morgan_fp(f, n_bits=self.n_bits)
            fps_list.append(fp_arr)

        if fps_list:
            fps_matrix = np.array(fps_list, dtype=np.float32)
        else:
            fps_matrix = np.zeros((0, self.n_bits), dtype=np.float32)

        # Store in cache under both canonical and raw SMILES
        with self.lock:
            self.smiles_to_frags[can_smi] = frags
            self.smiles_to_fps[can_smi] = fps_matrix
            self.smiles_to_frags[smiles] = frags
            self.smiles_to_fps[smiles] = fps_matrix

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

    def get_stats(self) -> Dict[str, float]:
        """Returns dictionary of cache usage statistics."""
        with self.lock:
            unique_drugs = len(set(id(v) for v in self.smiles_to_frags.values()))
            hit_rate = (self.hits / self.requests * 100.0) if self.requests > 0 else 0.0
            return {
                "unique_drugs": unique_drugs,
                "requests": self.requests,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }

    def print_stats(self):
        """Prints a human-readable summary of BRICSCache performance statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 45)
        print("         BRICS Cache Statistics")
        print("=" * 45)
        print(f"Unique drugs in cache: {stats['unique_drugs']}")
        print(f"Total lookup requests: {stats['requests']}")
        print(f"Cache hits           : {stats['hits']}")
        print(f"Cache misses         : {stats['misses']}")
        print(f"Cache hit rate       : {stats['hit_rate']:.2f}%")
        print("=" * 45 + "\n")

    def reset_stats(self):
        """Resets request/hit/miss counters to zero."""
        with self.lock:
            self.requests = 0
            self.hits = 0
            self.misses = 0

    def save_cache(self, file_path: str):
        with self.lock:
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

    def load_cache(self, file_path: str, verbose: bool = False):
        with self.lock:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for smi, content in data.items():
                self.smiles_to_frags[smi] = content["brics_fragments"]
                self.smiles_to_fps[smi] = np.array(content["fingerprints"], dtype=np.float32)
            if verbose:
                print(f"Loaded {len(self.smiles_to_frags)} cached drug entries from '{file_path}'.")


# Global singleton cache instance
_GLOBAL_BRICS_CACHE: Optional[BRICSCache] = None


def get_global_brics_cache(
    cache_file: Optional[str] = os.path.join("data", "brics_cache.json"),
    n_bits: int = 2048
) -> BRICSCache:
    """Returns or initializes global BRICSCache singleton instance."""
    global _GLOBAL_BRICS_CACHE
    if _GLOBAL_BRICS_CACHE is None:
        _GLOBAL_BRICS_CACHE = BRICSCache(cache_file=cache_file, n_bits=n_bits)
    return _GLOBAL_BRICS_CACHE


def set_global_brics_cache(cache: BRICSCache):
    """Sets global BRICSCache singleton instance."""
    global _GLOBAL_BRICS_CACHE
    _GLOBAL_BRICS_CACHE = cache


def reset_global_brics_cache():
    """Resets global BRICSCache singleton instance."""
    global _GLOBAL_BRICS_CACHE
    _GLOBAL_BRICS_CACHE = None

