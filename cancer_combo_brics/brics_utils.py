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
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


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
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            # Convert RDKit ExplicitBitVect to numpy array
            from rdkit.DataStructs import ConvertToNumpyArray
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
        n_bits: Morgan fingerprint dimension.
        radius: Morgan fingerprint radius.
        device: PyTorch device.

    Returns:
        fp_tensor: Tensor of shape (B, N_max, n_bits) containing fragment fingerprints.
        padding_mask: BoolTensor of shape (B, N_max), where True indicates a VALID fragment,
                      and False indicates PADDING.
        batch_fragments: List of lists containing raw fragment SMILES for each molecule.
    """
    batch_fragments = [decompose_smiles_to_brics(s) for s in smiles_list]
    batch_size = len(smiles_list)
    max_frags = max(len(frags) for frags in batch_fragments)
    # Ensure at least max_frags >= 1
    max_frags = max(max_frags, 1)

    fp_tensor = torch.zeros((batch_size, max_frags, n_bits), dtype=torch.float32)
    padding_mask = torch.zeros((batch_size, max_frags), dtype=torch.bool)

    for i, frags in enumerate(batch_fragments):
        for j, frag_smi in enumerate(frags):
            fp = fragment_to_morgan_fp(frag_smi, n_bits=n_bits, radius=radius)
            fp_tensor[i, j] = torch.from_numpy(fp)
            padding_mask[i, j] = True  # Valid fragment

    return fp_tensor.to(device), padding_mask.to(device), batch_fragments
