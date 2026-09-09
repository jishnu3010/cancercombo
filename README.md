# CancerCombo-BRICS

A clean, modular, and optimized PyTorch implementation for predicting complete **2D dose-response surfaces** of anticancer drug combinations with generalization to **completely unseen drugs**.

---

## Architecture

```text
                         CELL LINE
                    976-D GENE EXPRESSION
                              │
                              ▼
                    ┌──────────────────┐
                    │ Cell-Line Encoder│
                    └──────────────────┘
                              │
                              ▼
                         c ∈ R^512
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
          DRUG A                           DRUG B
           SMILES                            SMILES
              │                               │
              ▼                               ▼
            BRICS                           BRICS
              │                               │
              ▼                               ▼
     Shared pretrained                 Shared pretrained
         MoLFormer                         MoLFormer
              │                               │
              ▼                               ▼
       F_A ∈ R^(N×512)                  F_B ∈ R^(M×512)
              │                               │
              └──────────────┬────────────────┘
                             ▼
                  Cell-conditioned FiLM
                             │
                     F̃_A         F̃_B
                             │
                             ▼
             Bidirectional shared-weight
                  Fragment Cross-Attention
                             │
                             ▼
                          Pooling
                             │
                             ▼
                       r_AB ∈ R^2048
                             │
                             ▼
                     Projection → 512
                             │
                             ▼
                        r'_AB ∈ R^512
                             │
                ┌────────────┼────────────┐
                │            │            │
                │            ▼            │
                │            c            │
                │                         │
                ▼                         ▼
            r'_AB ⊙ c              |r'_AB - c|
                │                         │
                └────────────┬────────────┘
                             ▼
                  Drug–Cell Interaction
                             │
                             ▼
                     Sigmoid MLP Gate
                             │
                             ▼
                          r_gate
                             │
                             ▼
                 [r'_AB ; c ; r_gate]
                             │
                             ▼
                       r_DC ∈ R^1536
                             │
                             ▼
                  8 Pharmacological Heads
                             │
                             ▼
                    Constraint Transform
                             │
                             ▼
                    Bivariate Hill Solver
                             │
                             ▼
                    Dose-Dependent Bias
                             │
                             ▼
             COMPLETE 2D DOSE-RESPONSE SURFACE
```

---

## Key Features & Highlights

- **Strict Architecture 1 Adherence**: Cell line encoder ($976 \to 512$), deterministic RDKit BRICS decomposition with persistent caching, shared pretrained MoLFormer projection to 512-D, cell-conditioned FiLM with identity initialization, bidirectional shared-weight cross-attention, masked mean + max pooling ($r_{AB} \in \mathbb{R}^{2048} \to r'_{AB} \in \mathbb{R}^{512}$), explicit Drug-Cell interaction with Sigmoid MLP gate ($r_{DC} \in \mathbb{R}^{1536}$), 8 pharmacological parameter heads ($e_1, e_2, e_3, \log C_1, \log C_2, h_1, h_2, \alpha$), vectorized Bivariate Hill Solver ($e_0 = 1.0$), and bounded dose-dependent bias.
- **Normalized Viability Scale**: Single-pass conversion: $100\% \to 1.0$, guaranteeing internal mathematical consistency across all heads, constraints, and losses.
- **Unseen-Drug Generalization**: Strict drug-disjoint dataset splitting ($\text{Drugs}_{\text{train}} \cap \text{Drugs}_{\text{test}} = \emptyset$) and stratified evaluation across:
  - **Scenario 1**: Both drugs seen in training.
  - **Scenario 2**: One drug seen + one drug unseen.
  - **Scenario 3**: Both drugs unseen (primary research objective).
- **Production Performance**: Drug-level base fragment embedding caching, AMP mixed precision, unscaled gradient clipping, separate learning rate parameter groups for MoLFormer vs new layers, and zero-loop vectorized tensor operations.

---

## Directory Structure

```text
CancerCombo-BRICS/
│
├── data/                             # Dataset root (auto-discovered)
│   ├── brics_cache.sqlite            # Persistent SQLite BRICS fragment cache
│   └── ...
│
├── cancer_combo_brics/               # Core library
│   ├── __init__.py
│   ├── config.py                     # Typed dataclass and YAML configurations
│   ├── data/
│   │   ├── dataset.py                # Dataset and dynamic collation
│   │   ├── preprocessing.py          # Train-only cell expression standardization
│   │   ├── splitting.py              # Drug-disjoint splits and scenario classifier
│   │   └── validation.py             # Discovery, schema checks, and statistics
│   ├── chemistry/
│   │   ├── brics.py                  # RDKit BRICS decomposition with canonicalization
│   │   ├── cache.py                  # Thread-safe persistent & memory caching
│   │   └── fragment_utils.py         # Batch padding and fragment masks
│   ├── encoders/
│   │   ├── cell_encoder.py           # 976 -> 512 MLP with LayerNorm & GELU
│   │   ├── molformer.py              # MoLFormer backbone wrapper & mock mode
│   │   └── fragment_encoder.py       # Projection to 512-D and drug-level cache
│   ├── interaction/
│   │   ├── film.py                   # Cell-conditioned FiLM with identity init
│   │   ├── cross_attention.py        # Shared-weight bidirectional cross-attention
│   │   └── drug_cell.py              # z_DC (2048), Sigmoid gate, r_DC (1536)
│   ├── pharmacology/
│   │   ├── parameter_heads.py        # 8 scalar parameter heads
│   │   ├── constraints.py            # Biological domain transformations (e0=1.0)
│   │   ├── bivariate_hill.py         # Differentiable vectorized bivariate Hill solver
│   │   └── dose_bias.py              # Bounded dose-dependent bias
│   ├── model.py                      # Full end-to-end model
│   ├── losses.py                     # Huber, MSE, and MAE surface loss
│   ├── metrics.py                    # RMSE, MAE, R², Pearson, Spearman
│   ├── diagnostics.py                # Gradient, parameter, and surface monitoring
│   └── utils.py                      # Seed, checkpointing, and GPU profiling
│
├── scripts/
│   ├── prepare_data.py               # Data discovery, validation, split, and synthetic gen
│   ├── train.py                      # Production training with AMP and gradient checks
│   ├── evaluate.py                   # Checkpoint evaluation by scenario
│   └── inference.py                  # Predict 2D surfaces for unseen pairs
│
├── configs/
│   └── default.yaml                  # Baseline configuration file
│
├── tests/                            # Comprehensive PyTest test suite
│   ├── test_data.py
│   ├── test_brics.py
│   ├── test_encoders.py
│   ├── test_film.py
│   ├── test_attention.py
│   ├── test_drug_cell.py
│   ├── test_hill.py
│   ├── test_model.py
│   └── test_split.py
│
├── results/                          # Logs, metrics, and figures
├── checkpoints/                      # Saved models and preprocessing stats
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Required packages: `torch`, `transformers`, `rdkit`, `scipy`, `scikit-learn`, `pandas`, `numpy`, `pyyaml`, `tqdm`, `pytest`.

---

## Data Preparation & Validation

Place your dataset files in `./data/`. The system auto-discovers combination files (`.csv`, `.tsv`, `.parquet`) and cell expression profiles (`.csv`, `.tsv`, `.npz`).

To inspect and validate your dataset:
```bash
python scripts/prepare_data.py --config configs/default.yaml
```

To generate a synthetic cancer combination dataset for immediate end-to-end dry runs:
```bash
python scripts/prepare_data.py --synthetic --samples 100
```

---

## Training

Train with mixed precision, gradient monitoring, and separate learning rates:
```bash
python scripts/train.py --config configs/default.yaml
```

Options:
- `--epochs N`: Override epoch count
- `--batch_size N`: Override batch size
- `--device cuda|cpu`: Specify device

---

## Evaluation

Evaluate model checkpoint across the test set and breakdown performance by Scenario 1, Scenario 2, and Scenario 3:
```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt
```

---

## Inference

Predict the complete 2D dose-response surface and 8 pharmacological parameters for any pair of drug SMILES and cell line:
```bash
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pt \
    --smiles_a "CC(=O)Oc1ccccc1C(=O)O" \
    --smiles_b "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
    --cell_id "CELL_01" \
    --doses_a "0.0, 0.01, 0.05, 0.2, 1.0, 3.0, 10.0, 30.0" \
    --doses_b "0.0, 0.005, 0.02, 0.1, 0.5, 2.0, 8.0, 25.0"
```

---

## Running Unit & Integration Tests

```bash
pytest tests/ -v
```
