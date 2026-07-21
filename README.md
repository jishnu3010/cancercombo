# CancerCombo: Deep Learning Framework for Cancer Drug Combination Dose-Response Surface Prediction

**CancerCombo** is a modular, production-ready, biophysically constrained deep learning architecture designed to predict 2D dose-response viability matrices for cancer drug combinations. By integrating multi-modal drug representations (SMILES sequence embeddings, Morgan fingerprints, and physical descriptors) with cell-line gene expression profiles and a differentiable bivariate Hill equation solver, CancerCombo predicts continuous, smooth viability surfaces under strict permutation invariance.

---

## 🌟 Key Architectural Features

- **Multi-Modal Drug Representations**: Combines HuggingFace MoLFormer sequence Transformer embeddings, 2048-bit RDKit Morgan fingerprints, and 200 physical descriptors via Multi-Head Self-Attention Fusion.
- **Drug-Cell Cross-Attention**: Conditions drug representations on cell-line transcriptomics pathway tokens.
- **Permutation Invariance**: Enforces mathematical symmetry $f(\text{Drug}_A, \text{Drug}_B) = f(\text{Drug}_B, \text{Drug}_A)$ via symmetric combination pooling.
- **Differentiable Bivariate Hill Equation**: Predicts 8 log-space pharmacological parameters ($\log C_1, \log C_2, E_1, E_2, E_3, h_1, h_2, \alpha$) to reconstruct smooth 2D viability surfaces with autograd backpropagation.
- **Numerical Safety Guarantees**: Built-in protection against `log(0)`, single-precision float overflow, and division by zero.
- **Zero-Leakage Dataset Partitioning**: Precomputes Scenario 1 (Combination-wise), Scenario 2 (Cell-wise), and Scenario 3 (Drug-wise) splits.

---

## 📁  Structure

```
cancercombo/
├── blocks/                           # Modular PyTorch Neural Network Building Blocks
│   ├── cell_encoder.py               # Transcriptomics GEX pathway encoder
│   ├── descriptor_encoder.py         # Physical descriptor MLP encoder
│   ├── drug_cell_attention.py        # Drug-Cell cross-attention module
│   ├── drug_drug_attention.py        # Mutual drug interaction cross-attention
│   ├── fusion.py                     # Multi-representation self-attention fusion
│   ├── hill_equation.py              # Differentiable 2D Bivariate Hill solver
│   ├── molformer_encoder.py          # MoLFormer HuggingFace sequence encoder
│   ├── morgan_encoder.py             # Morgan fingerprint MLP encoder
│   ├── prediction_heads.py           # Log-space 8-parameter prediction heads
│   └── shared_feature.py             # Symmetric combination fusion pooling
├── data/                             # Dataset archives & pre-extracted features
│   ├── DrugCombination_with_SMILES.zip # 1,000,000 sample benchmark archive
│   ├── features/                     # Pre-extracted features
│   │   ├── NCI-60_landmark_gex.csv   # NCI-60 landmark transcriptomics matrix
│   │   ├── drug_features.pkl         # Precomputed Pickle drug feature store
│   │   └── drug_features.pt          # Precomputed PyTorch drug feature store
│   └── splits/                       # Precomputed 60/20/20 scenario splits
│       ├── scenario1_combination.csv # Combination-wise split
│       ├── scenario2_cell.csv        # Cell-wise split
│       └── scenario3_drug.csv        # Unseen drug-wise split
├── cancercombo.py                    # Top-level PyTorch neural network module
├── config.py                         # Dataclass model & training configuration
├── config.yaml                       # Primary hyperparameter YAML file
├── dataset.py                        # Dataset loader & regex SMILES tokenizer
├── evaluate.py                       # Checkpoint evaluation script
├── experimenter.py                   # Multi-experiment hyperparameter sweep runner
├── helpers.py                        # Seed setter & mock simulation generator
├── logger.py                         # Formatted console logger setup
├── losses.py                         # Composite loss function (MSE + Margin Ranking + Aux)
├── main.py                           # Primary unified CLI router (train, evaluate, predict)
├── metrics.py                        # Regression & synergy evaluation metrics
├── precompute_molecular_features.py  # Standalone feature pre-extraction script
├── predictor.py                      # Single & batch inference prediction engine
├── preprocessor.py                   # RDKit Morgan & descriptor preprocessor
├── requirements.txt                  # Pinned environment dependencies
├── split_dataset.py                  # Scenario 1, 2, 3 dataset splitter script
└── test_suite.py                     # Consolidated PyTest verification suite
```

---

## ⚙️ Installation

1. Create and activate a Python 3.10+ virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. Install pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Running Unit & Integration Tests

Verify system installation, tensor shape propagation, autograd stability, and permutation invariance:

```bash
python -m pytest
```

The consolidated test suite can also be executed directly:
```bash
python test_suite.py
```

---

## 📊 Dataset Partitioning & Feature Pre-computation

### 1. Partition Dataset into Scenarios (60/20/20)
Generate reproducible Scenario 1 (Combination-wise), Scenario 2 (Cell-wise), and Scenario 3 (Unseen Drug-wise) splits:

```bash
python split_dataset.py --input_csv data/DrugCombination_with_SMILES.zip --output_dir ./data/splits --seed 42
```

### 2. Precompute Drug Molecular Features
Pre-extract Morgan fingerprints, 200 physical descriptors (Z-score normalized), and SMILES tokens to eliminate CPU bottlenecks during training:

```bash
python precompute_molecular_features.py --input_csv data/DrugCombination_with_SMILES.zip --output_file data/features/drug_features.pt
```

---

## 🚀 Model Training, Evaluation & Inference

### 1. Model Training
Train CancerCombo using PyTorch Lightning with automatic mixed precision and GPU acceleration:

```bash
python main.py --mode train --config config.yaml
```

### 2. Checkpoint Evaluation
Evaluate a trained model checkpoint on test dataset splits:

```bash
python main.py --mode evaluate --checkpoint checkpoints/cancercombo_best.ckpt
```

### 3. Inference Prediction
Run single-pair or batch dose-response matrix predictions:

```bash
python main.py --mode predict --checkpoint checkpoints/cancercombo_best.ckpt
```

---

## 📈 Evaluation Metrics

CancerCombo logs the following metrics during training and evaluation:
- **MSE / RMSE / MAE**: Viability matrix surface reconstruction accuracy.
- **$R^2$ Score**: Proportion of variance explained.
- **Pearson ($r$) & Spearman ($r_s$)**: Linear and monotonic rank correlation.
- **Top-$K$ Precision, Recall & Hit Rate**: Synergistic combination retrieval performance.
