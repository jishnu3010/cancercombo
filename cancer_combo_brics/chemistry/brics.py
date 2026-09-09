"""BRICS chemical decomposition module with deterministic canonicalization and dummy handling."""

from __future__ import annotations

import re
from typing import List, Optional
import rdkit.Chem as Chem
from rdkit.Chem import BRICS


def clean_brics_fragment(frag_smiles: str, clean_mode: str = "remove") -> Optional[str]:
    """Clean BRICS dummy attachment points from fragment SMILES.

    Args:
        frag_smiles: Fragment SMILES with dummy atoms (e.g. '[6*]C(=O)O')
        clean_mode: One of 'remove' (strip dummies), 'cap_h' (replace with H),
                   'star' (replace with generic '*'), or 'raw' (keep as-is).

    Returns:
        Canonicalized clean fragment SMILES, or None if fragment is empty/invalid.
    """
    if not frag_smiles:
        return None

    if clean_mode == "raw":
        return frag_smiles

    if clean_mode in ["remove", "cap_h"]:
        # Parse fragment with RDKit
        mol = Chem.MolFromSmiles(frag_smiles)
        if mol is None:
            # Fallback to regex cleaning if raw parse fails
            cleaned = re.sub(r"\[\d*\*\]|\*", "", frag_smiles)
            cleaned = re.sub(r"\(\)", "", cleaned)
            return cleaned if cleaned else None

        # Replace dummy atoms (atomic num 0) with Hydrogen and clear isotope
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
                atom.SetAtomicNum(1)

        # Remove explicit hydrogens to yield canonical fragment
        try:
            clean_mol = Chem.RemoveHs(mol)
            return Chem.MolToSmiles(clean_mol, isomericSmiles=True, canonical=True)
        except Exception:
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    if clean_mode == "star":
        # Normalize dummy labels to generic '*'
        cleaned = re.sub(r"\[\d*\*\]", "*", frag_smiles)
        return cleaned

    raise ValueError(f"Unknown clean_mode: {clean_mode}")


def decompose_smiles(
    smiles: str,
    clean_mode: str = "remove",
    min_fragment_size: int = 1,
) -> List[str]:
    """Deterministically decompose a molecule SMILES into BRICS fragments.

    Args:
        smiles: Input SMILES string.
        clean_mode: Mode to clean dummy attachment points ('remove', 'cap_h', 'star', 'raw').
        min_fragment_size: Minimum number of heavy atoms for a fragment to be retained.

    Returns:
        Sorted list of canonical fragment SMILES strings.
        If molecule has no BRICS bonds, returns [canonicalized_smiles].
    """
    if not smiles or not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Invalid SMILES: Input is empty or not a string.")

    cleaned_input = smiles.strip()
    mol = Chem.MolFromSmiles(cleaned_input)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: '{cleaned_input}'")

    canonical_full = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    try:
        raw_fragments = BRICS.BRICSDecompose(mol, returnMols=False, singlePass=False)
    except Exception:
        # Fallback if decomposition fails
        return [canonical_full]

    if not raw_fragments:
        return [canonical_full]

    processed_frags: List[str] = []
    for f in raw_fragments:
        cleaned_frag = clean_brics_fragment(f, clean_mode=clean_mode)
        if cleaned_frag:
            # Check heavy atom count if min_fragment_size > 1
            if min_fragment_size > 1:
                frag_mol = Chem.MolFromSmiles(cleaned_frag)
                if frag_mol is not None and frag_mol.GetNumHeavyAtoms() < min_fragment_size:
                    continue
            processed_frags.append(cleaned_frag)

    # Deduplicate and sort deterministically
    unique_sorted = sorted(list(set(processed_frags)))
    if not unique_sorted:
        return [canonical_full]

    return unique_sorted
