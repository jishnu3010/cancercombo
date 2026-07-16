import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from typing import Tuple, Optional

class MolecularPreprocessor:
    """RDKit chemistry preprocessor mapping SMILES strings to numerical representations."""
    
    def __init__(self, morgan_nbits: int = 2048, morgan_radius: int = 2):
        self.morgan_nbits = morgan_nbits
        self.morgan_radius = morgan_radius
        self.descriptor_names = [desc[0] for desc in Descriptors._descList][:200]
        
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

    def get_morgan_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Compute binary Morgan fingerprint vector from Mol.

        Args:
            mol: RDKit Mol object.

        Returns:
            np.ndarray: Binary array of shape (morgan_nbits,).
        """
        if mol is None:
            return np.zeros(self.morgan_nbits, dtype=np.float32)
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=self.morgan_radius, nBits=self.morgan_nbits
            )
            arr = np.zeros((1,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr.astype(np.float32)
        except Exception:
            return np.zeros(self.morgan_nbits, dtype=np.float32)

    def get_physical_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """Compute continuous descriptors from Mol.

        Args:
            mol: RDKit Mol object.

        Returns:
            np.ndarray: Vector of shape (200,).
        """
        desc_vec = []
        if mol is None:
            return np.zeros(len(self.descriptor_names), dtype=np.float32)
            
        for name in self.descriptor_names:
            try:
                func = getattr(Descriptors, name)
                val = func(mol)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                desc_vec.append(float(val))
            except Exception:
                desc_vec.append(0.0)
                
        res = np.array(desc_vec, dtype=np.float32)
        
        # Format padding/slicing to match config
        if len(res) < 200:
            res = np.pad(res, (0, 200 - len(res)))
        elif len(res) > 200:
            res = res[:200]
        return res

    def process_smiles(self, smiles: str) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Runs the complete chemical processing pipeline on a SMILES string.

        Args:
            smiles: Raw SMILES string.

        Returns:
            Tuple[np.ndarray, np.ndarray, bool]: Morgan vector, descriptor vector, success flag.
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return (
                np.zeros(self.morgan_nbits, dtype=np.float32),
                np.zeros(200, dtype=np.float32),
                False
            )
        
        morgan_fp = self.get_morgan_fingerprint(mol)
        descriptors = self.get_physical_descriptors(mol)
        return morgan_fp, descriptors, True
