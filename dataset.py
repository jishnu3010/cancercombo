import os
import re
import json
import zipfile
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from preprocessor import MolecularPreprocessor
from typing import List, Dict, Tuple, Any

SMILES_REGEX = r"(\[[^\]]+\]|Br|Cl|Si|Se|B|C|N|O|P|S|F|I|b|c|n|o|p|s|==|#|%[0-9]{2}|[0-9]|\+|-|=|/|\\|\@|\.|\(|\)|~|\*)"

class SMILESTokenizer:
    """Robust regex-based tokenizer for SMILES chemical structures."""
    
    def __init__(self, max_len: int = 128):
        self.max_len = max_len
        self.regex = re.compile(SMILES_REGEX)
        tokens = [
            '[PAD]', '[UNK]', 'C', 'c', 'O', 'o', 'N', 'n', 'S', 's', 'F', 'P', 'p',
            'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'H', '=', '#', '(', ')', '[', ']',
            '/', '\\', '+', '-', '@', '.', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
        ]
        self.vocab = {t: idx for idx, t in enumerate(tokens)}
        self.pad_idx = self.vocab['[PAD]']
        self.unk_idx = self.vocab['[UNK]']

    def tokenize(self, smiles: str) -> Tuple[List[int], List[int]]:
        """Tokenize a SMILES string using regex matching, mapping tokens to IDs.

        Args:
            smiles: Raw SMILES string.

        Returns:
            Tuple[List[int], List[int]]: Token IDs and attention mask.
        """
        matched_tokens = self.regex.findall(smiles)
        ids = [self.vocab.get(t, self.unk_idx) for t in matched_tokens]
        
        # Padding & Truncation
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            attention_mask = [1] * self.max_len
        else:
            padding_len = self.max_len - len(ids)
            attention_mask = [1] * len(ids) + [0] * padding_len
            ids = ids + [self.pad_idx] * padding_len
            
        return ids, attention_mask


def load_nci60_gex(csv_path: str = "data/features/NCI-60_landmark_gex.csv", target_dim: int = 20000) -> Dict[str, np.ndarray]:
    """Load cell line gene expression matrix from NCI-60 CSV file.

    Args:
        csv_path: Path to NCI-60 gene expression CSV file.
        target_dim: Dimension size to pad/trim to match model cell_in_dim.

    Returns:
        Dict[str, np.ndarray]: Dict mapping cell line names to float32 expression vectors.
    """
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        cell_lines = df.columns[1:]
        gex_dict = {}
        for cell in cell_lines:
            vec = df[cell].values.astype(np.float32)
            if len(vec) < target_dim:
                vec = np.pad(vec, (0, target_dim - len(vec)))
            elif len(vec) > target_dim:
                vec = vec[:target_dim]
            norm_key = cell.replace('-', '_').replace('/', '_').upper()
            gex_dict[cell] = vec
            gex_dict[norm_key] = vec
        return gex_dict
    except Exception:
        return {}


def load_synergy_dataset(zip_or_csv_path: str = "data/DrugCombination_with_SMILES.zip") -> List[Dict[str, Any]]:
    """Load drug combination dataset from ZIP or CSV archive.

    Args:
        zip_or_csv_path: Path to dataset ZIP archive or CSV file.

    Returns:
        List[Dict[str, Any]]: Parsed list of sample dictionaries.
    """
    if not os.path.exists(zip_or_csv_path):
        return []
    try:
        if zip_or_csv_path.endswith('.zip'):
            with zipfile.ZipFile(zip_or_csv_path, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return []
                dfs = []
                for csv_file in csv_files:
                    with z.open(csv_file) as f:
                        dfs.append(pd.read_csv(f))
                df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(zip_or_csv_path)
            
        records = df.to_dict('records')
        data_list = []
        default_d_a = [0.0, 0.1, 1.0, 10.0]
        default_d_b = [0.0, 0.2, 2.0, 20.0]
        
        for row in records:
            s_a = row.get('Drug1_SMILES', row.get('smiles_a', row.get('smiles1', row.get('SMILES_A', ''))))
            s_b = row.get('Drug2_SMILES', row.get('smiles_b', row.get('smiles2', row.get('SMILES_B', ''))))
            cell = row.get('Sample', row.get('cell_line_name', row.get('cell', row.get('CELL_NAME', 'MCF7'))))
            
            d_a = row.get('Drug1_Dose', row.get('doses_a', default_d_a))
            d_b = row.get('Drug2_Dose', row.get('doses_b', default_d_b))
            if isinstance(d_a, str):
                try: d_a = json.loads(d_a)
                except Exception: d_a = default_d_a
            if isinstance(d_b, str):
                try: d_b = json.loads(d_b)
                except Exception: d_b = default_d_b
                
            viab = row.get('Response', row.get('viability_matrix', None))
            if isinstance(viab, str):
                try: viab = json.loads(viab)
                except Exception: viab = np.zeros((len(d_a), len(d_b))).tolist()
            elif viab is None:
                viab = np.zeros((len(d_a), len(d_b))).tolist()
                
            data_list.append({
                "smiles_a": str(s_a),
                "smiles_b": str(s_b),
                "cell_line_name": str(cell),
                "doses_a": d_a,
                "doses_b": d_b,
                "viability_matrix": viab
            })
        return data_list
    except Exception:
        return []


class DrugComboDataset(Dataset):
    """PyTorch Dataset class representing cell line and drug combination inputs."""
    
    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        cell_line_features: Dict[str, np.ndarray],
        drug_feature_store: Dict[str, Dict[str, Any]] | None = None,
        max_smiles_len: int = 128
    ):
        """
        Args:
            data_list: List of dictionaries of samples.
            cell_line_features: Dictionary mapping cell line name to expression vector.
            max_smiles_len: Maximum token sequence length.
        """
        self.data = data_list
        self.cell_line_features = cell_line_features
        self.drug_feature_store = drug_feature_store or {}
        self.tokenizer = SMILESTokenizer(max_len=max_smiles_len)
        self.preprocessor = MolecularPreprocessor()
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        smiles_a = item['smiles_a']
        smiles_b = item['smiles_b']
        cell_name = item['cell_line_name']
        doses_a = np.array(item['doses_a'], dtype=np.float32)
        doses_b = np.array(item['doses_b'], dtype=np.float32)
        viability = np.array(item['viability_matrix'], dtype=np.float32)
        
        # Calculate or retrieve chemical representations
        feat_a = self.drug_feature_store.get(smiles_a)
        feat_b = self.drug_feature_store.get(smiles_b)

        if feat_a is not None:
            morgan_a = feat_a["morgan"] if isinstance(feat_a, dict) else feat_a[0]
            desc_a = feat_a["descriptors"] if isinstance(feat_a, dict) else feat_a[1]
            ids_a = feat_a["token_ids"] if isinstance(feat_a, dict) else feat_a[2]
            mask_a = feat_a["token_mask"] if isinstance(feat_a, dict) else feat_a[3]
        else:
            morgan_a, desc_a, _ = self.preprocessor.process_smiles(smiles_a)
            ids_a, mask_a = self.tokenizer.tokenize(smiles_a)

        if feat_b is not None:
            morgan_b = feat_b["morgan"] if isinstance(feat_b, dict) else feat_b[0]
            desc_b = feat_b["descriptors"] if isinstance(feat_b, dict) else feat_b[1]
            ids_b = feat_b["token_ids"] if isinstance(feat_b, dict) else feat_b[2]
            mask_b = feat_b["token_mask"] if isinstance(feat_b, dict) else feat_b[3]
        else:
            morgan_b, desc_b, _ = self.preprocessor.process_smiles(smiles_b)
            ids_b, mask_b = self.tokenizer.tokenize(smiles_b)
        
        # Get biological profile
        norm_cell = cell_name.replace('-', '_').replace('/', '_').upper()
        cell_vec = self.cell_line_features.get(cell_name, self.cell_line_features.get(norm_cell))
        if cell_vec is None:
            cell_vec = np.zeros(20000, dtype=np.float32)
            
        res_dict = {
            "drug_a_ids": torch.tensor(ids_a, dtype=torch.long),
            "drug_a_mask": torch.tensor(mask_a, dtype=torch.float32),
            "drug_a_morgan": torch.tensor(morgan_a, dtype=torch.float32),
            "drug_a_desc": torch.tensor(desc_a, dtype=torch.float32),
            
            "drug_b_ids": torch.tensor(ids_b, dtype=torch.long),
            "drug_b_mask": torch.tensor(mask_b, dtype=torch.float32),
            "drug_b_morgan": torch.tensor(morgan_b, dtype=torch.float32),
            "drug_b_desc": torch.tensor(desc_b, dtype=torch.float32),
            
            "cell_line": torch.tensor(cell_vec, dtype=torch.float32),
            "doses_a": torch.tensor(doses_a, dtype=torch.float32),
            "doses_b": torch.tensor(doses_b, dtype=torch.float32),
            "viability": torch.tensor(viability, dtype=torch.float32)
        }
        
        # Include optional Hill parameters for auxiliary supervision if present
        for p in ["e1", "e2", "e3", "log_c1", "log_c2", "h1", "h2", "alpha"]:
            if p in item:
                res_dict[p] = torch.tensor([float(item[p])], dtype=torch.float32)
                
        return res_dict


def load_precomputed_drug_features(feature_path: str) -> Dict[str, Dict[str, Any]]:
    """Load a saved drug feature store from .pt or .pkl."""
    if not os.path.exists(feature_path):
        return {}

    try:
        if feature_path.endswith(".pt"):
            return torch.load(feature_path, map_location="cpu")
        if feature_path.endswith(".pkl"):
            with open(feature_path, "rb") as f:
                return pickle.load(f)
    except Exception:
        return {}

    return {}

