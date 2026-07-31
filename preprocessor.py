import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from typing import Tuple, Optional
from functools import lru_cache

try:
    from rdkit.Chem import rdFingerprintGenerator
    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False

class MolecularPreprocessor:
    """RDKit chemistry preprocessor mapping SMILES strings to numerical representations with LRU caching."""
    
    def __init__(self, morgan_nbits: int = 2048, morgan_radius: int = 2, cache_size: int = 10000):
        self.morgan_nbits = morgan_nbits
        self.morgan_radius = morgan_radius
        self.descriptor_names = [desc[0] for desc in Descriptors._descList][:200]
        self.cache_size = cache_size
        self._cache = {}
        self._generator = None

    def _get_generator(self):
        if self._generator is None and HAS_GENERATOR:
            self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=self.morgan_radius, fpSize=self.morgan_nbits)
        return self._generator

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_generator"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._generator = None
        
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES to RDKit Mol object.

        Args:
            smiles: Raw SMILES string.

        Returns:
            Optional[Chem.Mol]: RDKit Mol object or None if invalid.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
            return mol
        except Exception:
            return None

# ============================================================
# ABLATION 1 - MOLFORMER ONLY
# DISABLED FOR ABLATION 2
# ============================================================
#     def get_morgan_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
#         """Disabled for MolFormer-only ablation."""
#         raise NotImplementedError("Morgan fingerprints are disabled in MolFormer-only ablation.")
#
#     def get_physical_descriptors(self, mol: Chem.Mol) -> np.ndarray:
#         """Disabled for MolFormer-only ablation."""
#         raise NotImplementedError("RDKit physical descriptors are disabled in MolFormer-only ablation.")
#
#     def process_smiles(self, smiles: str) -> bool:
#         """MolFormer-only SMILES validation check."""
#         mol = self.smiles_to_mol(smiles)
#         return mol is not None

# ============================================================
# ABLATION 2 - MORGAN + RDKIT DESCRIPTORS ONLY
# ACTIVE
# ============================================================
    def get_morgan_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Generate 2048-bit Morgan fingerprint vector (radius=2)."""
        if mol is None:
            return np.zeros(self.morgan_nbits, dtype=np.float32)
        try:
            generator = self._get_generator()
            if generator is not None:
                fp = generator.GetFingerprint(mol)
            else:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius=self.morgan_radius, nBits=self.morgan_nbits
                )
            arr = np.zeros((1,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr.astype(np.float32)
        except Exception:
            return np.zeros(self.morgan_nbits, dtype=np.float32)

    def get_physical_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """Generate 200 continuous RDKit physical descriptor values."""
        desc_vec = []
        if mol is None:
            return np.zeros(len(self.descriptor_names), dtype=np.float32)
        for name in self.descriptor_names:
            try:
                func = getattr(Descriptors, name)
                val = func(mol)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                else:
                    val = float(np.clip(val, -1e6, 1e6))
                desc_vec.append(val)
            except Exception:
                desc_vec.append(0.0)
        res = np.array(desc_vec, dtype=np.float32)
        if len(res) < 200:
            res = np.pad(res, (0, 200 - len(res)))
        elif len(res) > 200:
            res = res[:200]
        return res

    def process_smiles(self, smiles: str) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Processes SMILES string into Morgan fingerprint and continuous descriptors."""
        if smiles in self._cache:
            m_fp, desc, ok = self._cache[smiles]
            return m_fp.copy(), desc.copy(), ok
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            res = (np.zeros(self.morgan_nbits, dtype=np.float32), np.zeros(200, dtype=np.float32), False)
        else:
            morgan_fp = self.get_morgan_fingerprint(mol)
            descriptors = self.get_physical_descriptors(mol)
            res = (morgan_fp, descriptors, True)
        if len(self._cache) < self.cache_size:
            self._cache[smiles] = res
        return res[0].copy(), res[1].copy(), res[2]



