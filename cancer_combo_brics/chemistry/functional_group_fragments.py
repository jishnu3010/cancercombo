"""RDKit functional-group fragmentation module with context preservation and robust aromaticity handling.

Identifies functional groups in a molecule using RDKit SMARTS matching and extracts
each functional group along with its surrounding molecular/attachment context (1-hop neighbors).

Context Preservation Strategy:
  - Identifies core functional group atom indices (e.g. C(=O)NH for amides, -OH for alcohols/phenols).
  - Expands the core atom set by including 1-hop neighbor atoms to capture attachment and electronic context.
  - Extracts the sub-molecule for the expanded atom set using RDKit with kekulization/aromaticity sanitization.
  - Converts sub-molecules into RDKit-validated canonical SMILES fragment strings.

Fallback Handling (CASE A & CASE B):
  - CASE A: If a molecule has no recognized functional group SMARTS match, returns [canonical_full_molecule_smiles].
  - CASE B: If functional groups match but every extracted candidate fragment is invalid, returns [canonical_full_molecule_smiles].
  - No valid drug molecule is ever discarded or replaced with dummy/fabricated SMILES.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple
import rdkit.Chem as Chem

logger = logging.getLogger(__name__)

# Standard functional-group SMARTS patterns
FUNCTIONAL_GROUP_SMARTS: Dict[str, str] = {
    "carboxylic_acid": "[CX3](=O)[OX1H,OX1-]",
    "ester": "[CX3](=O)[O][CX4,a]",
    "amide": "[CX3](=O)[NX3]",
    "carbonyl": "[CX3]=O",
    "amine_primary": "[NX3;H2]",
    "amine_secondary": "[NX3;H1;!$(N*=*)]",
    "amine_tertiary": "[NX3;H0;!$(N*=*);!$(N=*)]",
    "alcohol_phenol": "[OX2H]",
    "ether": "[OD2]([c,C])([c,C])",
    "nitro": "[$([NX3](=O)=O),$([NX3+]([O-])=O)]",
    "sulfonyl": "[SX4](=O)(=O)",
    "halide": "[F,Cl,Br,I]",
    "nitrile": "[C]#[N]",
    "aromatic_ring": "c1ccccc1",
    "hetero_aromatic": "[a;!c]",
}


def _compile_smarts_patterns() -> Dict[str, Chem.Mol]:
    compiled = {}
    for name, pattern in FUNCTIONAL_GROUP_SMARTS.items():
        pat_mol = Chem.MolFromSmarts(pattern)
        if pat_mol is not None:
            compiled[name] = pat_mol
    return compiled


COMPILED_FG_PATTERNS: Dict[str, Chem.Mol] = _compile_smarts_patterns()


def validate_and_canonicalize_fragment(frag_smiles: str) -> Optional[str]:
    """Validate that a fragment SMILES parses, sanitizes, and canonicalizes cleanly in RDKit.

    Returns:
        Canonical fragment SMILES string if valid, or None if invalid / un-kekulizable.
    """
    if not frag_smiles or not isinstance(frag_smiles, str) or not frag_smiles.strip():
        return None

    try:
        t_mol = Chem.MolFromSmiles(frag_smiles.strip())
        if t_mol is None:
            return None

        Chem.SanitizeMol(t_mol)
        canon = Chem.MolToSmiles(t_mol, isomericSmiles=True, canonical=True)
        # Verify re-parsing of canonical string
        if Chem.MolFromSmiles(canon) is None:
            return None
        return canon
    except Exception:
        return None


def extract_fragment_with_context(
    mol: Chem.Mol,
    match_indices: Tuple[int, ...],
    radius: int = 1,
) -> Optional[str]:
    """Extract sub-molecule SMILES for a matched core plus its 1-hop context with robust aromaticity handling.

    Args:
        mol: Parent RDKit Mol object.
        match_indices: Atom indices of matched functional group core.
        radius: Number of topological hops to expand context (default 1).

    Returns:
        RDKit-validated canonical fragment SMILES string with context, or None if candidate is invalid.
    """
    if not match_indices:
        return None

    # Expand atom set to include neighbor context up to radius hops
    atoms_to_include: Set[int] = set(match_indices)
    for _ in range(radius):
        new_atoms: Set[int] = set()
        for idx in atoms_to_include:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                new_atoms.add(neighbor.GetIdx())
        atoms_to_include.update(new_atoms)

    if len(atoms_to_include) == 1:
        idx = list(atoms_to_include)[0]
        sym = mol.GetAtomWithIdx(idx).GetSymbol()
        return validate_and_canonicalize_fragment(sym)

    # Collect bonds between included atoms
    bonds_to_include: List[int] = []
    for bond in mol.GetBonds():
        if (
            bond.GetBeginAtomIdx() in atoms_to_include
            and bond.GetEndAtomIdx() in atoms_to_include
        ):
            bonds_to_include.append(bond.GetIdx())

    if not bonds_to_include:
        idx = list(atoms_to_include)[0]
        sym = mol.GetAtomWithIdx(idx).GetSymbol()
        return validate_and_canonicalize_fragment(sym)

    # Strategy 1: Kekulize a copy of parent mol before submolecule extraction
    try:
        mol_kek = Chem.Mol(mol)
        try:
            Chem.Kekulize(mol_kek, clearAromaticFlags=True)
        except Exception:
            pass

        submol = Chem.PathToSubmol(mol_kek, bonds_to_include)
        frag_smiles = Chem.MolToSmiles(submol, isomericSmiles=True, canonical=True)
        val_canon = validate_and_canonicalize_fragment(frag_smiles)
        if val_canon is not None:
            return val_canon
    except Exception:
        pass

    # Strategy 2: Raw submolecule extraction with strict validation
    try:
        submol_raw = Chem.PathToSubmol(mol, bonds_to_include)
        frag_smiles_raw = Chem.MolToSmiles(submol_raw, isomericSmiles=True, canonical=True)
        val_canon_raw = validate_and_canonicalize_fragment(frag_smiles_raw)
        if val_canon_raw is not None:
            return val_canon_raw
    except Exception:
        pass

    # Candidate fragment failed validity/kekulization checks -> reject candidate
    return None


def extract_functional_group_fragments(
    smiles: str,
    radius: int = 1,
    min_fragment_size: int = 1,
) -> List[str]:
    """Extract context-preserving functional-group fragments from a SMILES string.

    Guarantees:
      - Every returned fragment SMILES is strictly RDKit-valid and sanitizable.
      - Fallback CASE A: Returns [canonical_full] if no SMARTS pattern matches.
      - Fallback CASE B: Returns [canonical_full] if all candidate extracted fragments are invalid.

    Args:
        smiles: Input SMILES string.
        radius: Context expansion radius (default 1).
        min_fragment_size: Minimum heavy atom count (default 1).

    Returns:
        Sorted, unique list of RDKit-valid canonical fragment SMILES strings.
    """
    if not smiles or not isinstance(smiles, str) or not smiles.strip() or str(smiles).strip().lower() == "nan":
        logger.warning(f"Invalid or missing SMILES string '{smiles}'.")
        return []

    cleaned_input = smiles.strip()
    mol = Chem.MolFromSmiles(cleaned_input)
    if mol is None:
        logger.warning(f"RDKit failed to parse SMILES '{cleaned_input}'.")
        return []

    canonical_full = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    extracted_fragments: List[str] = []

    # Run SMARTS matching for all functional groups
    for fg_name, pat_mol in COMPILED_FG_PATTERNS.items():
        matches = mol.GetSubstructMatches(pat_mol)
        for match_indices in matches:
            frag_smiles = extract_fragment_with_context(
                mol, match_indices=match_indices, radius=radius
            )
            if frag_smiles:
                if min_fragment_size > 1:
                    frag_mol = Chem.MolFromSmiles(frag_smiles)
                    if frag_mol is not None and frag_mol.GetNumHeavyAtoms() < min_fragment_size:
                        continue
                extracted_fragments.append(frag_smiles)

    # Deduplicate and sort deterministically
    unique_fragments = sorted(list(set(extracted_fragments)))

    # Fallback to full canonical SMILES if no functional groups match (CASE A)
    # or if all extracted candidate fragments were invalid (CASE B)
    if not unique_fragments:
        return [canonical_full]

    return unique_fragments
