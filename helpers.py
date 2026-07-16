import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Any

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across packages.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_mock_data(num_samples: int = 64) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Generate mock data based on continuous 2D Hill equation for execution simulation.

    Args:
        num_samples: Number of sample dicts to generate.

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]: Mock datasets list and cell features.
    """
    smiles_pool = [
        "CC1=CC(=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5", # Imatinib
        "CC1=C(C(=CC=C1)Cl)C(=O)NC2=C(C=C(S2)C(=O)NC3=NC(=CS3)C)C",                 # Dasatinib
        "CC(=O)NC1=CC=C(C=C1)O",                                                   # Acetaminophen
        "COC1=CC=CC2=C1C(=C(C(=O)O2)C)C",                                          # Mock
        "CC1=CC(=C(S1)C(=O)N2CC(NC(=O)OC(C)(C)C)C2)C"                             # Mock
    ]
    
    cell_names = ["MCF7", "A549", "HELA", "K562"]
    
    cell_line_features = {
        name: np.random.randn(20000).astype(np.float32) for name in cell_names
    }
    
    data_list = []
    for _ in range(num_samples):
        s_a = np.random.choice(smiles_pool)
        s_b = np.random.choice(smiles_pool)
        cell = np.random.choice(cell_names)
        
        doses_a = [0.0, 0.1, 1.0, 10.0]
        doses_b = [0.0, 0.2, 2.0, 20.0]
        
        mat = np.zeros((4, 4))
        for i, da in enumerate(doses_a):
            for j, db in enumerate(doses_b):
                val = 100.0 / (1.0 + (da / 2.0)**1.2 + (db / 3.0)**1.5 + 0.5 * (da / 2.0)**1.2 * (db / 3.0)**1.5)
                val += np.random.normal(0, 2.0)
                mat[i, j] = np.clip(val, 0.0, 100.0)
                
        data_list.append({
            "smiles_a": s_a,
            "smiles_b": s_b,
            "cell_line_name": cell,
            "doses_a": doses_a,
            "doses_b": doses_b,
            "viability_matrix": mat.tolist()
        })
        
    return data_list, cell_line_features
