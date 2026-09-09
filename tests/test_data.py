"""Unit tests for dataset, parsing, and cell expression preprocessing."""

import numpy as np
import pytest
import torch
from cancer_combo_brics.data.dataset import parse_dose_array, parse_viability_matrix
from cancer_combo_brics.data.preprocessing import CellExpressionPreprocessor


def test_parse_dose_array():
    # JSON list
    doses_json = "[0.0, 0.1, 1.0, 10.0]"
    arr = parse_dose_array(doses_json)
    assert arr.shape == (4,)
    assert arr[0] == 0.0
    assert arr[-1] == 10.0

    # Comma separated
    doses_str = "0.0, 0.5, 2.5"
    arr2 = parse_dose_array(doses_str)
    assert arr2.shape == (3,)
    assert arr2[1] == 0.5


def test_parse_viability_matrix():
    # 2x2 matrix from JSON
    mat_json = "[[100.0, 95.0], [90.0, 60.0]]"
    mat = parse_viability_matrix(mat_json, shape=(2, 2))
    assert mat.shape == (2, 2)
    assert mat[0, 0] == 100.0
    assert mat[1, 1] == 60.0


def test_cell_preprocessor_isolation():
    # Train data with 976 landmark genes
    np.random.seed(42)
    train_expr = np.random.randn(20, 976).astype(np.float32)
    train_expr[0, 0] = np.nan  # inject missing value

    preprocessor = CellExpressionPreprocessor(expected_dim=976)
    preprocessor.fit(train_expr)

    assert preprocessor.is_fitted
    assert preprocessor.means.shape == (976,)
    assert preprocessor.stds.shape == (976,)

    # Transform train
    norm_train = preprocessor.transform(train_expr)
    assert norm_train.shape == (20, 976)
    assert not np.isnan(norm_train).any()

    # Val data normalized using TRAIN stats
    val_expr = np.random.randn(5, 976).astype(np.float32)
    norm_val = preprocessor.transform(val_expr)
    assert norm_val.shape == (5, 976)
    assert not np.isnan(norm_val).any()

    # Check save and load
    save_path = "./data/test_preprocessor.npz"
    preprocessor.save(save_path)
    loaded = CellExpressionPreprocessor.load(save_path)
    assert np.allclose(loaded.means, preprocessor.means)
    assert np.allclose(loaded.stds, preprocessor.stds)


def test_cancer_combo_dataset_dose_conversion_and_scaling():
    import pandas as pd
    from cancer_combo_brics.data.dataset import CancerComboDataset

    # Mock dataset with Molar doses (6e-8 M) and percentage viability (100.0)
    data = {
        "smiles_a": ["CCO"],
        "smiles_b": ["CC(=O)O"],
        "cell_line_name": ["MCF7"],
        "doses_a": ["[0.0, 6e-08, 6e-07, 6e-06]"],
        "doses_b": ["[0.0, 5e-08, 5e-07, 5e-06]"],
        "viability_matrix": ["[[100.0, 95.0, 90.0, 80.0], [98.0, 90.0, 85.0, 70.0], [95.0, 85.0, 75.0, 60.0], [90.0, 75.0, 60.0, 40.0]]"],
    }
    df = pd.DataFrame(data)
    cell_expressions = {"MCF7": np.ones(976, dtype=np.float32)}

    ds = CancerComboDataset(df, cell_expressions=cell_expressions)
    sample = ds[0]

    # Doses converted M -> uM (* 1e6)
    np.testing.assert_allclose(sample["doses_a"], np.array([0.0, 0.06, 0.6, 6.0], dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(sample["doses_b"], np.array([0.0, 0.05, 0.5, 5.0], dtype=np.float32), rtol=1e-5)

    # Viability remains on percentage scale: 100.0 is 100.0 (not 1.0)
    assert sample["viability_matrix"].shape == (4, 4)
    assert sample["viability_matrix"][0, 0] == 100.0
    assert sample["viability_matrix"].max() <= 120.0
    assert sample["viability_matrix"].min() >= 0.0

