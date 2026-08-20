# CancerCombo — Google Colab Training Guide

Follow these step-by-step instructions to train the **CancerCombo** model on Google Colab with GPU acceleration (T4 / V100 / A100).

---

## Step 1: Open Colab & Enable GPU
1. Open [Google Colab](https://colab.research.google.com).
2. Create a **New Notebook**.
3. Enable GPU Acceleration:
   - Go to **Runtime** > **Change runtime type**.
   - Under **Hardware accelerator**, select **GPU** (e.g., T4, V100, or A100).

---

## Step 2: Clone Repository & Change Directory
In the first Colab code cell, run:

```bash
!git clone https://github.com/jishnu3010/cancercombo.git
%cd cancercombo
```

---

## Step 3: Install Dependencies
Install RDKit and PyYAML:

```bash
!pip install rdkit pyyaml
```

---

## Step 4: Verify GPU & Dataset Integrity
Run the diagnostic scripts to verify GPU device detection, model gradients, and dataset splits:

```bash
!python validate_model.py
!python validate_dataset.py
```

---

## Step 5: Start Model Training
Run `train_dgx.py` to start training:

```bash
!python train_dgx.py --batch_size 128 --epochs 500
```

*Note: You can pass custom arguments like `--epochs 100` or `--batch_size 64` if using a free T4 GPU.*

---

## Step 6: Save Checkpoints to Google Drive (Optional)
To save your trained model checkpoints (`best_cancer_combo_brics.pt`) directly to Google Drive so they persist after Colab disconnects:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints to Google Drive
!mkdir -p /content/drive/MyDrive/CancerCombo_Checkpoints
!cp -r checkpoints/* /content/drive/MyDrive/CancerCombo_Checkpoints/
```

---

## Step 7: Run Evaluation & Inference in Colab
Evaluate the trained model on test metrics or run inference:

```bash
# Run baseline evaluation
!python baselines_and_ablations.py

# Run sample inference on unseen drug combinations
!python inference.py
```
