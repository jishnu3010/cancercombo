# CancerCombo-BRICS Datasets Folder

This directory contains the primary benchmark dataset:
- **`scenario3_drug1.csv`** (74.4 MB): 154,705 drug combination viability surface matrices across NCI-60 cell lines.

---

## Dataset Schema (`scenario3_drug1.csv`)

| Column Name | Description | Example Value |
| :--- | :--- | :--- |
| `smiles_a` | SMILES string for Drug A | `C1=NC2=C(N=C(N=C2N1C3...` |
| `smiles_b` | SMILES string for Drug B | `CCC1=C2CN3C(=CC4=C...` |
| `cell_line_name` | NCI-60 cell line identifier | `7860`, `A549`, `HCT116` |
| `doses_a` | JSON array of Drug A concentrations | `"[0.0, 6e-08, 6e-07, 6e-06]"` |
| `doses_b` | JSON array of Drug B concentrations | `"[0.0, 1e-10, 1e-09, 1e-08]"` |
| `viability_matrix` | $4 \times 4$ JSON 2D viability array | `"[[100.0, 98.52, ...], ...]"` |
| `split` | Train/Val/Test split index | `3` |

---

## Loading `scenario3_drug1.csv` in PyTorch

Use `load_cancer_combo_from_csv` in `cancer_combo_brics.dataset`:

```python
from torch.utils.data import DataLoader
from cancer_combo_brics import load_cancer_combo_from_csv, collate_cancer_combo_batch

# Load dataset directly from uploaded scenario3_drug1.csv
dataset = load_cancer_combo_from_csv("data/scenario3_drug1.csv")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_cancer_combo_batch
)
```
