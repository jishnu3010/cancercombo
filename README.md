# CancerCombo

CancerCombo is a modular, production-ready deep learning framework designed to predict complete dose-response matrices for cancer drug combinations. It utilizes differentiable pharmacological modeling constraints (the bivariate Hill equation) to output continuous, biophysically sound viability surfaces.

## Structure
* `blocks/`: Contains the modular component layers.
  * `molformer_encoder.py`: Semantic SMILES encoder.
  * `morgan_encoder.py`: Topological Morgan fingerprint encoder.
  * `descriptor_encoder.py`: Molecular descriptor encoder.
  * `fusion.py`: Self-attention multi-modal fusion.
  * `cell_encoder.py`: Cellular pathway encoder.
  * `drug_cell_attention.py`: Cell-conditioning cross-attention.
  * `drug_drug_attention.py`: Mutual drug interaction cross-attention.
  * `shared_feature.py`: Symmetric pooling projection.
  * `prediction_heads.py`: Eight parameter estimation heads.
  * `hill_equation.py`: Differentiable Hill solver.
* `config.yaml` / `config.py`: Configuration models.
* `dataset.py` / `preprocessor.py`: Data loaders and feature engineering helpers.
* `cancercombo.py`: Combined wrapper model.
* `losses.py` / `metrics.py`: Loss functions and correlation metrics.
* `trainer.py` / `train.py`: PyTorch Lightning wrappers and scripts.
* `evaluate.py` / `evaluator.py`: Validation evaluation scripts.
* `predictor.py`: Inference prediction pipeline.
* `experimenter.py`: Ablation study pipeline.
* `logger.py` / `helpers.py`: Logger and utilities.
* `test_model.py` / `test_forward.py` / `test_hill.py`: Unit tests.
* `main.py`: General command-line interface entry point.

## Installation
Set up your virtual environment and install all packages:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Running Tests
Verify your system has correct dependencies and check dimensions and gradients:
```bash
python -m pytest
```

## Training and Inference
Run the training pipeline:
```bash
python main.py --mode train
```

Run validation on a trained model:
```bash
python main.py --mode evaluate --checkpoint checkpoints/cancercombo_best.ckpt
```

Run single drug-pair inference:
```bash
python main.py --mode predict --checkpoint checkpoints/cancercombo_best.ckpt
```
