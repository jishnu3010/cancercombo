"""Interaction modules for CancerCombo."""

from cancer_combo_brics.interaction.pairwise_interaction import ExplicitPairwiseFragmentInteraction
from cancer_combo_brics.interaction.drug_cell import DrugCellInteraction

__all__ = [
    "ExplicitPairwiseFragmentInteraction",
    "DrugCellInteraction",
]
