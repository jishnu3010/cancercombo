"""
CancerCombo-BRICS-Symmetric Package.

Provides PyTorch modules for predicting full 2D dose-response viability surfaces
for cancer drug combinations with mathematical drug-order invariance.
"""

from .cell_line_encoder import CellLineEncoder
from .fragment_encoder import FragmentEncoder
from .film_conditioning import FiLMConditioning
from .cross_attention import MaskedBidirectionalCrossAttention, BidirectionalCrossAttention, ManualMultiHeadCrossAttention
from .symmetric_fusion import SymmetricFusion
from .cell_fusion import CellFusion
from .parameter_heads import ParameterHeads
from .constraint_transform import ConstraintTransform
from .bivariate_hill_solver import BivariateHillSolver
from .model import CancerComboBRICSSymmetric, CancerComboBRICS
from .dataset import CancerComboDataset, collate_cancer_combo_batch, load_cancer_combo_from_csv
from .brics_utils import (
    decompose_smiles_to_brics,
    fragment_to_morgan_fp,
    collate_brics_fragments
)

__all__ = [
    "CellLineEncoder",
    "FragmentEncoder",
    "FiLMConditioning",
    "MaskedBidirectionalCrossAttention",
    "BidirectionalCrossAttention",
    "ManualMultiHeadCrossAttention",
    "SymmetricFusion",
    "CellFusion",
    "ParameterHeads",
    "ConstraintTransform",
    "BivariateHillSolver",
    "CancerComboBRICSSymmetric",
    "CancerComboBRICS",
    "CancerComboDataset",
    "collate_cancer_combo_batch",
    "load_cancer_combo_from_csv",
    "decompose_smiles_to_brics",
    "fragment_to_morgan_fp",
    "collate_brics_fragments"
]
