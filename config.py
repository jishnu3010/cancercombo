"""
Configuration Settings for CancerCombo-BRICS-Symmetric.

Centralizes model architecture hyperparameters, dataset file paths,
DGX GPU settings, and training hyperparameters.

Parses config.yaml if available, or falls back to default values.
"""

import os
import torch

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


CONFIG_YAML_PATH = "config.yaml"

# Default configuration parameters
_defaults = {
    "dataset": {
        "data_csv": "data/scenario3_drug_level.csv",
        "original_csv": "data/scenario3_drug1.csv",
        "train_split": 3,
        "val_split": 2,
        "test_split": 1,
        "max_samples": None,
    },
    "model": {
        "gene_dim": 976,
        "cell_dim": 512,
        "frag_fp_dim": 2048,
        "d_dim": 128,
        "num_attn_heads": 4,
        "dropout_rate": 0.2,
        "shared_attn_weights": True,
    },
    "training": {
        "batch_size": 64,
        "epochs": 50,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "num_workers": 8,
        "use_amp": True,
    },
    "checkpoints": {
        "dir": "checkpoints",
        "best_model_name": "best_cancer_combo_brics.pt",
    },
    "logging": {
        "log_dir": "logs"
    }
}


def _load_config():
    cfg = {sec: dict(vals) for sec, vals in _defaults.items()}
    if YAML_AVAILABLE and os.path.exists(CONFIG_YAML_PATH):
        try:
            with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as f:
                yaml_cfg = yaml.safe_load(f)
                if isinstance(yaml_cfg, dict):
                    for sec, vals in yaml_cfg.items():
                        if sec in cfg and isinstance(vals, dict):
                            cfg[sec].update(vals)
        except Exception:
            pass
    return cfg


_cfg = _load_config()

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================
DATA_DIR = "data"
_csv_candidate = str(_cfg["dataset"]["data_csv"])
_orig_candidate = str(_cfg["dataset"]["original_csv"])
DATA_CSV = _csv_candidate if os.path.exists(_csv_candidate) else _orig_candidate

TRAIN_SPLIT = int(_cfg["dataset"]["train_split"])
VAL_SPLIT = int(_cfg["dataset"]["val_split"])
TEST_SPLIT = int(_cfg["dataset"]["test_split"])
MAX_SAMPLES = _cfg["dataset"]["max_samples"]

# ==============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# ==============================================================================
GENE_DIM = int(_cfg["model"]["gene_dim"])            # NCI-60 / LINCS L1000 landmark gene expression dimension
CELL_DIM = int(_cfg["model"]["cell_dim"])            # Cell feature vector representation dimension (c)
FRAG_FP_DIM = int(_cfg["model"]["frag_fp_dim"])        # Morgan fingerprint vector dimension (ECFP4)
D_DIM = int(_cfg["model"]["d_dim"])               # Shared fragment embedding dimension (d)
NUM_ATTN_HEADS = int(_cfg["model"]["num_attn_heads"])        # Multi-head cross-attention heads
DROPOUT_RATE = float(_cfg["model"]["dropout_rate"])        # Encoders dropout rate
SHARED_ATTN_WEIGHTS = bool(_cfg["model"]["shared_attn_weights"])# Shared cross-attention weights for exact order invariance

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = int(_cfg["training"]["batch_size"])
EPOCHS = int(_cfg["training"]["epochs"])
LEARNING_RATE = float(_cfg["training"]["learning_rate"])
WEIGHT_DECAY = float(_cfg["training"]["weight_decay"])
NUM_WORKERS = int(_cfg["training"]["num_workers"])
USE_AMP = bool(_cfg["training"]["use_amp"])            # Automatic Mixed Precision (FP16) on NVIDIA Tensor Cores

# ==============================================================================
# OUTPUT & CHECKPOINT CONFIGURATION
# ==============================================================================
CHECKPOINT_DIR = str(_cfg["checkpoints"]["dir"])
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, str(_cfg["checkpoints"]["best_model_name"]))
LOG_DIR = str(_cfg["logging"]["log_dir"])

# ==============================================================================
# HARDWARE & DEVICE CONFIGURATION
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if DEVICE.type == "cuda" else False
