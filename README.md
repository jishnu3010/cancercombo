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
    RDKit Functional-Group           RDKit Functional-Group
    Fragmentation (1-hop)            Fragmentation (1-hop)
              │                               │
              ▼                               ▼
      Morgan Subgraphs                 Morgan Subgraphs
       (Mol2Vec d=300)                  (Mol2Vec d=300)
              │                               │
              ▼                               ▼
     Learnable Projection             Learnable Projection
              │                               │
              ▼                               ▼
       F_A ∈ R^(N×512)                  F_B ∈ R^(M×512)
              │                               │
              └──────────────┬────────────────┘
                             ▼
                   Explicit Pairwise Fragment
                          Interaction
                             │
                             ▼
                  Masked Mean Pooling
                             │
                             ▼
                       r_AB ∈ R^512
                             │
                ┌────────────┼────────────┐
                │            │            │
                │            ▼            │
                │            c            │
                │                         │
                ▼                         ▼
            r_AB ⊙ c                |r_AB - c|
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
                 [r_AB ; c ; r_gate]
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

- **Adherence to Functional-Group & Mol2Vec Architecture**:
  - Cell line encoder ($976 \to 512$).
  - RDKit Functional-Group decomposition with 1-hop context preservation and persistent SQLite caching (`fg_cache_v1`).
  - Mol2Vec Morgan subgraph environment vectorizer ($D=300$) with learnable 512-D projection.
  - Explicit Pairwise Fragment Interaction MLP ($z_{ij} \in \mathbb{R}^{2048} \to 512 \to 512$) with masked mean pooling ($r_{AB} \in \mathbb{R}^{512}$).
  - Explicit Drug-Cell interaction with Sigmoid MLP gate ($r_{\text{DC}} \in \mathbb{R}^{1536}$).
  - 8 pharmacological parameter heads ($e_1, e_2, e_3, \log C_1, \log C_2, h_1, h_2, \alpha$).
  - Vectorized Bivariate Hill Solver ($e_0 = 100.0$) and bounded dose-dependent bias.
- **Percentage Viability Scale**: Percentage viability target scale ($100.0 = 100\%$), guaranteeing internal mathematical consistency across all heads, constraints, and solver predictions.
- **Unseen-Drug Generalization**: Strict drug-disjoint dataset splitting ($\text{Drugs}_{\text{train}} \cap \text{Drugs}_{\text{test}} = \emptyset$) and stratified evaluation across:
  - **Scenario 1**: Both drugs seen in training.
  - **Scenario 2**: One drug seen + one drug unseen.
  - **Scenario 3**: Both drugs unseen (primary research objective).
- **Production Performance**: AMP mixed precision, unscaled gradient clipping, zero-loop vectorized tensor operations, and cross-platform picklable DataLoader collation.

---

## Directory Structure

```text
CancerCombo-BRICS/
│
├── data/                             # Dataset root (auto-discovered)
│   ├── scenario3_drug_level.csv      # Primary drug combination dataset
│   ├── cell_line_gene_expr.csv       # Cell line gene expression matrix
│   ├── fg_cache.sqlite               # Persistent SQLite functional group cache
│   └── ...
│
├── cancer_combo_brics/               # Core library
│   ├── __init__.py
│   ├── config.py                     # Typed dataclass and YAML configurations
│   ├── data/
│   │   ├── dataset.py                # Dataset and dynamic collation
│   │   ├── preprocessing.py          # Automatic CSV orientation & cell standardization
│   │   ├── splitting.py              # Drug-disjoint splits and scenario classifier
│   │   └── validation.py             # Discovery, schema checks, and statistics
│   ├── chemistry/
│   │   ├── functional_group_fragments.py # RDKit functional group matching (1-hop)
│   │   ├── cache.py                  # Thread-safe persistent & memory caching
│   │   └── fragment_utils.py         # Batch padding and fragment masks
│   ├── encoders/
│   │   ├── cell_encoder.py           # 976 -> 512 MLP with LayerNorm & GELU
│   │   ├── mol2vec_encoder.py        # Mol2Vec Morgan environment vectorizer (D=300 -> 512)
│   │   └── fragment_encoder.py       # Fragment encoder wrapper
│   ├── interaction/
│   │   ├── pairwise_interaction.py   # Explicit Pairwise Fragment Interaction (512)
│   │   └── drug_cell.py              # z_DC (2048), Sigmoid gate, r_DC (1536)
│   ├── pharmacology/
│   │   ├── parameter_heads.py        # 8 scalar parameter heads
│   │   ├── constraints.py            # Biological domain transformations (e0=100.0)
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
│   ├── default.yaml                  # Baseline configuration file
│   └── test_run.yaml                 # Fast verification test run configuration
│
├── tests/                            # Comprehensive PyTest test suite (40 tests)
│   ├── test_architecture_verification.py
│   ├── test_functional_groups.py
│   ├── test_mol2vec.py
│   ├── test_pairwise_interaction.py
│   └── ...
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

---

## Complete DGX / Execution Commands (Step-by-Step)

### 1. Update Repository
```bash
git fetch origin
git reset --hard origin/main
```

### 2. Verify Architecture & Data Pipeline via PyTest
```bash
python -m pytest
```

### 3. Run Short Verification Test Run (2 Epochs, Batch Size 16)
```bash
python scripts/train.py --config configs/test_run.yaml --device cuda
```

### 4. Run Full Production Training (50 Epochs, AMP Mixed Precision)
```bash
python scripts/train.py --config configs/default.yaml --device cuda
```

### 5. Evaluate Best Checkpoint Across Unseen-Drug Scenarios 1, 2, 3
```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --output results/evaluation_metrics.json --device cuda
```

### 6. Run Single Pair Inference Prediction
```bash
python scripts/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pt \
    --smiles_a "CC(=O)Oc1ccccc1C(=O)O" \
    --smiles_b "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
    --cell_id "MCF7" \
    --output results/inference_output.json \
    --device cuda
```
