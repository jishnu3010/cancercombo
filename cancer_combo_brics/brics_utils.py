"""
BRICS Preprocessing Utilities for CancerCombo-BRICS.

Decomposes input SMILES strings into BRICS fragments using RDKit,
converts fragments to Morgan Fingerprints (ECFP4), and handles batch collation
with padding and boolean attention masks.
"""

from typing import List, Tuple, Union, Optional
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
        List of fragment SMILES strings.
    Raises:
        ValueError: If SMILES is invalid or cannot be parsed by RDKit.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for BRICS decomposition. Please install rdkit.")

    if not smiles or not isinstance(smiles, str) or smiles.lower() in ("nan", "none", "null"):
        raise ValueError(f"Invalid SMILES string encountered: '{smiles}'")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse invalid SMILES string: '{smiles}'")

    try:
        # BRICSDecompose breaks strategically valid bonds
        fragments = list(BRICS.BRICSDecompose(mol, keepIntermediate=False))
        if not fragments:
            fragments = [smiles]
        return fragments
    except Exception as e:
        # Fallback to full SMILES if BRICS decomposition fails but mol is valid
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
    Raises:
        ValueError: If fragment SMILES cannot be parsed by RDKit.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for fingerprint calculation.")

    if not frag_smiles or frag_smiles.lower() in ("nan", "none", "null"):
        raise ValueError(f"Invalid fragment SMILES string: '{frag_smiles}'")

    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse fragment SMILES string: '{frag_smiles}'")

    fp_array = np.zeros((n_bits,), dtype=np.float32)
    try:
        from rdkit.DataStructs import ConvertToNumpyArray
        if _MORGAN_GEN is not None and n_bits == 2048 and radius == 2:
            fp = _MORGAN_GEN.GetFingerprint(mol)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        ConvertToNumpyArray(fp, fp_array)
    except Exception as e:
        raise ValueError(f"Failed to generate fingerprint for fragment '{frag_smiles}': {e}")

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


def print_drug_fragments(smiles: str, fragments: List[str], drug_label: str):
    """
    Debug printing helper for single drug SMILES and BRICS fragment list.

    Args:
        smiles: Original SMILES string.
        fragments: Pre-existing list of BRICS fragment SMILES strings.
        drug_label: Label string (e.g. 'Drug A' or 'Drug B').
    """
    if not RDKIT_AVAILABLE or not smiles or Chem.MolFromSmiles(smiles) is None:
        print(f"{drug_label}: Invalid SMILES")
        return
    print(f"  {drug_label}:")
    print(f"    SMILES: {smiles}")
    print("    BRICS Fragments:")
    prefix = "A" if "A" in drug_label else ("B" if "B" in drug_label else "")
    for i, frag in enumerate(fragments, 1):
        print(f"      {prefix}{i}: {frag}")


def print_batch_drug_fragments(
    smiles_A_list: List[str],
    frags_A_list: List[List[str]],
    smiles_B_list: List[str],
    frags_B_list: List[List[str]]
):
    """
    Debug printing helper for Drug A and Drug B original SMILES and BRICS fragments across a batch.

    Args:
        smiles_A_list: List of original SMILES for Drug A in batch.
        frags_A_list: List of pre-existing BRICS fragment SMILES lists for Drug A in batch.
        smiles_B_list: List of original SMILES for Drug B in batch.
        frags_B_list: List of pre-existing BRICS fragment SMILES lists for Drug B in batch.
    """
    batch_size = max(len(smiles_A_list), len(smiles_B_list))
    for i in range(batch_size):
        smiles_a = smiles_A_list[i] if i < len(smiles_A_list) else ""
        frags_a = frags_A_list[i] if i < len(frags_A_list) else []
        smiles_b = smiles_B_list[i] if i < len(smiles_B_list) else ""
        frags_b = frags_B_list[i] if i < len(frags_B_list) else []

        print("=" * 60)
        print(f"Sample {i + 1}")
        print("=" * 60)

        # Drug A
        mol_a = Chem.MolFromSmiles(smiles_a) if (RDKIT_AVAILABLE and smiles_a and smiles_a.lower() not in ("nan", "none", "null")) else None
        if mol_a is None:
            print("Drug A: Invalid SMILES")
        else:
            print("Drug A:")
            print(f"  SMILES: {smiles_a}")
            print("  BRICS Fragments:")
            for idx, frag in enumerate(frags_a, 1):
                print(f"    A{idx}: {frag}")

        # Drug B
        mol_b = Chem.MolFromSmiles(smiles_b) if (RDKIT_AVAILABLE and smiles_b and smiles_b.lower() not in ("nan", "none", "null")) else None
        if mol_b is None:
            print("Drug B: Invalid SMILES")
        else:
            print("Drug B:")
            print(f"  SMILES: {smiles_b}")
            print("  BRICS Fragments:")
            for idx, frag in enumerate(frags_b, 1):
                print(f"    B{idx}: {frag}")
