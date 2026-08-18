"""
BRICS Preprocessing Utilities for CancerCombo-BRICS.

Decomposes input SMILES strings into BRICS fragments using RDKit,
converts fragments to Morgan Fingerprints (ECFP4), and handles batch collation
with padding and boolean attention masks.
"""

from typing import List, Tuple, Union
import torch
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import BRICS, AllChem
    try:
        from rdkit.Chem import rdFingerprintGenerator
        _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    except Exception:
        _MORGAN_GEN = None
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    _MORGAN_GEN = None


def decompose_smiles_to_brics(smiles: str) -> List[str]:
    """
    Decomposes a SMILES string into BRICS fragments using RDKit.

    Args:
        smiles: SMILES string of the drug molecule.

    Returns:
        List of fragment SMILES strings. If decomposition yields no fragments or fails,
        returns [smiles].
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for BRICS decomposition. Please install rdkit.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Fallback for invalid SMILES
        return [smiles if smiles else "C"]

    try:
        # BRICSDecompose breaks strategically valid bonds
        fragments = list(BRICS.BRICSDecompose(mol, keepIntermediate=False))
        if not fragments:
            fragments = [smiles]
        return fragments
    except Exception:
        # Fallback to full SMILES if BRICS fails
        return [smiles]


def fragment_to_morgan_fp(frag_smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """
    Computes Morgan Fingerprint (ECFP4) for a fragment SMILES string.
    Uses modern rdFingerprintGenerator if available to prevent deprecation warnings.

    Args:
        frag_smiles: Fragment SMILES string.
        n_bits: Length of fingerprint vector (default: 2048).
        radius: Fingerprint radius (default: 2 -> ECFP4).

    Returns:
        Numpy array of shape (n_bits,) containing binary fingerprint.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for fingerprint calculation.")

    mol = Chem.MolFromSmiles(frag_smiles)
    fp_array = np.zeros((n_bits,), dtype=np.float32)
    if mol is not None:
        try:
            from rdkit.DataStructs import ConvertToNumpyArray
            if _MORGAN_GEN is not None and n_bits == 2048 and radius == 2:
                fp = _MORGAN_GEN.GetFingerprint(mol)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            ConvertToNumpyArray(fp, fp_array)
        except Exception:
            pass
    return fp_array


def collate_brics_fragments(
    smiles_list: List[str],
    n_bits: int = 2048,
    radius: int = 2,
    device: Union[torch.device, str] = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    """
    Collates a list of molecule SMILES into a padded tensor of fragment fingerprints
    and a corresponding boolean padding mask.

    Args:
        smiles_list: List of B SMILES strings.
        n_bits: Morgan FP dimension (2048).
        radius: Fingerprint radius (2).
        device: PyTorch device.

    Returns:
        fp_tensor: Padded tensor of shape (B, n_max, n_bits).
        mask_tensor: Boolean mask tensor of shape (B, n_max) where True indicates real fragment, False indicates padding.
        all_fragments: List of fragment lists for each molecule in batch.
    """
    all_fragments: List[List[str]] = []
    max_len = 0

    for smiles in smiles_list:
        frags = decompose_smiles_to_brics(smiles)
        all_fragments.append(frags)
        if len(frags) > max_len:
            max_len = len(frags)

    max_len = max(max_len, 1)
    batch_size = len(smiles_list)

    fp_tensor = torch.zeros(batch_size, max_len, n_bits, dtype=torch.float32)
    mask_tensor = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, frags in enumerate(all_fragments):
        for j, frag_smi in enumerate(frags):
            fp_arr = fragment_to_morgan_fp(frag_smi, n_bits=n_bits, radius=radius)
            fp_tensor[i, j] = torch.from_numpy(fp_arr)
            mask_tensor[i, j] = True

    return fp_tensor.to(device), mask_tensor.to(device), all_fragments
