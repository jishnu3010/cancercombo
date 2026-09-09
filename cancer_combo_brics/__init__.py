"""CancerCombo-BRICS package."""

from cancer_combo_brics.config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from cancer_combo_brics.model import CancerComboBRICS
from cancer_combo_brics.losses import SurfaceRegressionLoss
from cancer_combo_brics.metrics import compute_surface_metrics, evaluate_predictions_grouped

__version__ = "1.0.0"
__all__ = [
    "CancerComboBRICS",
    "ExperimentConfig",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "SurfaceRegressionLoss",
    "compute_surface_metrics",
    "evaluate_predictions_grouped",
]
