"""
Configuration Settings for CancerCombo-BRICS-Symmetric.

Centralizes model architecture hyperparameters, dataset file paths,
DGX GPU settings, and training hyperparameters.
"""

import os
import torch

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================
DATA_DIR = "data"
DEFAULT_DRUG_LEVEL_CSV = os.path.join(DATA_DIR, "scenario3_drug_level.csv")
DEFAULT_ORIGINAL_CSV = os.path.join(DATA_DIR, "scenario3_drug1.csv")
DATA_CSV = DEFAULT_DRUG_LEVEL_CSV if os.path.exists(DEFAULT_DRUG_LEVEL_CSV) else DEFAULT_ORIGINAL_CSV
TRAIN_SPLIT = 3
VAL_SPLIT = 2
TEST_SPLIT = 1
MAX_SAMPLES = None  # Set integer for fast debugging, or None for full dataset

# ==============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# ==============================================================================
GENE_DIM = 976            # NCI-60 / LINCS L1000 landmark gene expression dimension
CELL_DIM = 512            # Cell feature vector representation dimension (c)
FRAG_FP_DIM = 2048        # Morgan fingerprint vector dimension (ECFP4)
D_DIM = 128               # Shared fragment embedding dimension (d)
NUM_ATTN_HEADS = 4        # Multi-head cross-attention heads
DROPOUT_RATE = 0.2        # Encoders dropout rate
SHARED_ATTN_WEIGHTS = True# Shared cross-attention weights for exact order invariance

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
USE_AMP = True            # Automatic Mixed Precision (FP16) on NVIDIA Tensor Cores

# ==============================================================================
# OUTPUT & CHECKPOINT CONFIGURATION
# ==============================================================================
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_cancer_combo_brics.pt")
LOG_DIR = "logs"

# ==============================================================================
# HARDWARE & DEVICE CONFIGURATION
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if DEVICE.type == "cuda" else False
