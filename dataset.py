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
from typing import List, Dict, Tuple, Any, Optional

def _to_tensor(x: Any, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if x.dtype != dtype else x
    return torch.tensor(x, dtype=dtype)

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


def load_nci60_gex(csv_path: str = "data/features/NCI-60_landmark_gex.csv", target_dim: int = 976) -> Dict[str, np.ndarray]:
    """Load cell line gene expression matrix from NCI-60 CSV file.

    Args:
        csv_path: Path to NCI-60 gene expression CSV file.
        target_dim: Expected dimension size (for validation only).

    Returns:
        Dict[str, np.ndarray]: Dict mapping cell line names to float32 expression vectors.
    """
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        # First column is gene names, remaining columns are canonical cell line names
        cell_lines = df.columns[1:]
        # Impute missing gene values with gene-wise median across cell lines
        df[cell_lines] = df[cell_lines].apply(lambda row: row.fillna(row.median()), axis=1)
        df[cell_lines] = df[cell_lines].fillna(0.0)
        
        gex_dict = {}
        for cell in cell_lines:
            vec = df[cell].values.astype(np.float32)
            if target_dim and len(vec) != target_dim:
                raise ValueError(f"Gene expression vector for {cell} has dimension {len(vec)}, expected {target_dim}")
            
            # Map canonical name and normalized name
            gex_dict[cell] = vec
            norm_key = re.sub(r'[^A-Z0-9]', '', str(cell).upper())
            gex_dict[norm_key] = vec
        return gex_dict
    except Exception as e:
        print(f"Error loading gene expression: {e}")
        return {}


def resolve_cell_line(sample_str: str, known_cells_map: Dict[str, str]) -> Optional[str]:
    """Robustly extract biological cell line name from a complex Sample string and return canonical name.
    
    Args:
        sample_str: The full sample string (e.g., 'DrugA_DrugB_786-0', 'DrugA_DrugB_A549/ATCC', 'DrugA_DrugB_HL-60(TB)')
        known_cells_map: Mapping from normalized cell line string to canonical cell line name.
        
    Returns:
        Optional[str]: The canonical cell line name if matched, else None.
    """
    if not isinstance(sample_str, str) or not sample_str.strip():
        return None
    s_clean = re.sub(r'\(.*?\)', '', str(sample_str)).strip()
    
    # Protect NCI/ADR-RES while handling provider slash suffixes (e.g. A549/ATCC -> A549)
    if '/' in s_clean and 'NCI/ADR' not in s_clean.upper():
        s_clean = s_clean.split('/')[0].strip()
        
    norm_sample = re.sub(r'[^A-Z0-9]', '', s_clean.upper())
    
    # Sort normalized keys by length descending to match longest cell line name first (preventing prefix collision)
    sorted_keys = sorted(known_cells_map.keys(), key=len, reverse=True)
    for norm_k in sorted_keys:
        if norm_sample.endswith(norm_k):
            return known_cells_map[norm_k]
            
    return None


def match_cell_line(sample_str: str, known_cells_map: Dict[str, str]) -> Optional[str]:
    """Alias to resolve_cell_line for backward compatibility."""
    return resolve_cell_line(sample_str, known_cells_map)


def parse_dataframe_to_records(df: pd.DataFrame, known_gex_dict: Dict[str, np.ndarray] = None) -> List[Dict[str, Any]]:
    """Parse pandas DataFrame into a list of 2D dose-response matrix records grouped by Sample.
    
    Args:
        df: Input dataframe containing drug and dose-response data.
        known_gex_dict: Dictionary mapping cell lines to gene expression vectors.
        
    Returns:
        List[Dict]: List of parsed sample records containing 4x4 viability matrices.
    """
    if df.empty:
        return []
        
    # Build normalized lookup for cell matching
    known_cells_map = {}
    if known_gex_dict:
        for k in known_gex_dict.keys():
            norm_k = re.sub(r'[^A-Z0-9]', '', str(k).upper())
            known_cells_map[norm_k] = k

    # First check if the dataframe is already pre-grouped records (e.g., has doses_a as list or matrix)
    if 'viability_matrix' in df.columns or 'doses_a' in df.columns:
        records = df.to_dict('records')
        data_list = []
        for row in records:
            s_a = row.get('smiles_a', row.get('Drug1_SMILES', ''))
            s_b = row.get('smiles_b', row.get('Drug2_SMILES', ''))
            cell_raw = str(row.get('cell_line_name', row.get('Sample', '')))
            
            if pd.isna(s_a) or pd.isna(s_b) or not str(s_a).strip() or not str(s_b).strip():
                continue
                
            cell = match_cell_line(cell_raw, known_cells_map) if known_cells_map else cell_raw
            if known_cells_map and (cell is None or cell not in known_gex_dict):
                continue
                
            d_a = row.get('doses_a')
            d_b = row.get('doses_b')
            viab = row.get('viability_matrix', row.get('viability'))
            
            if isinstance(d_a, str):
                try: d_a = json.loads(d_a)
                except Exception: continue
            if isinstance(d_b, str):
                try: d_b = json.loads(d_b)
                except Exception: continue
            if isinstance(viab, str):
                try: viab = json.loads(viab)
                except Exception: continue
                
            data_list.append({
                "smiles_a": str(s_a),
                "smiles_b": str(s_b),
                "cell_line_name": str(cell),
                "doses_a": d_a,
                "doses_b": d_b,
                "viability_matrix": viab
            })
        return data_list

    # Vectorized fast grouping for raw 16-row per Sample dataframe
    df_clean = df.dropna(subset=['Response']).copy()
    df_sorted = df_clean.sort_values(['Sample', 'Drug1_Dose', 'Drug2_Dose']).reset_index(drop=True)
    
    # Extract unique sample metadata headers
    first_rows = df_sorted.drop_duplicates(subset=['Sample'], keep='first').copy()
    
    matched_count = 0
    unmatched_count = 0
    unique_cells = set()
    unmatched_names = set()
    
    sample_to_cell = {}
    for sample_name in first_rows['Sample'].values:
        if known_cells_map:
            cell = match_cell_line(sample_name, known_cells_map)
            if cell is None or cell not in known_gex_dict:
                unmatched_count += 1
                unmatched_names.add(sample_name)
                continue
            matched_count += 1
            unique_cells.add(cell)
            sample_to_cell[sample_name] = cell
        else:
            sample_to_cell[sample_name] = str(sample_name)

    valid_samples = [s for s in first_rows['Sample'].values if s in sample_to_cell]
    valid_set = set(valid_samples)
    
    df_valid = df_sorted[df_sorted['Sample'].isin(valid_set)].copy()
    
    # Verify exact 16-row matrix requirement per sample
    sample_counts = df_valid['Sample'].value_counts()
    valid_16_samples = set(sample_counts[sample_counts == 16].index)
    df_valid = df_valid[df_valid['Sample'].isin(valid_16_samples)].copy()
    
    first_rows_valid = first_rows[first_rows['Sample'].isin(valid_16_samples)].copy()
    
    s_a_col = 'Drug1_SMILES' if 'Drug1_SMILES' in first_rows_valid.columns else 'Drug1'
    s_b_col = 'Drug2_SMILES' if 'Drug2_SMILES' in first_rows_valid.columns else 'Drug2'
    
    # Verify 4x4 dose grid and build explicit Cartesian coordinate response matrix per sample
    grouped = df_valid.groupby('Sample', sort=False)
    data_list = []
    
    for sample_name, sample_df in grouped:
        if sample_name not in sample_to_cell:
            continue
        cell = sample_to_cell[sample_name]
        
        row_first = sample_df.iloc[0]
        s_a = str(row_first[s_a_col])
        s_b = str(row_first[s_b_col])
        if pd.isna(s_a) or pd.isna(s_b) or not s_a.strip() or not s_b.strip():
            continue
            
        d1_doses = np.sort(sample_df['Drug1_Dose'].unique())
        d2_doses = np.sort(sample_df['Drug2_Dose'].unique())
        
        if len(d1_doses) != 4 or len(d2_doses) != 4:
            continue
            
        coord_map = {}
        valid_grid = True
        for da_val, db_val, resp_val in zip(sample_df['Drug1_Dose'].values, sample_df['Drug2_Dose'].values, sample_df['Response'].values):
            da_f, db_f, resp_f = float(da_val), float(db_val), float(resp_val)
            if np.isnan(resp_f) or np.isinf(resp_f):
                valid_grid = False
                break
            coord_map[(da_f, db_f)] = resp_f
            
        if not valid_grid or len(coord_map) != 16:
            continue
            
        matrix = np.zeros((4, 4), dtype=np.float32)
        complete = True
        for i, da in enumerate(d1_doses):
            for j, db in enumerate(d2_doses):
                coord_key = (float(da), float(db))
                if coord_key not in coord_map:
                    complete = False
                    break
                matrix[i, j] = coord_map[coord_key]
            if not complete:
                break
                
        if not complete:
            continue
            
        data_list.append({
            "smiles_a": s_a,
            "smiles_b": s_b,
            "cell_line_name": cell,
            "doses_a": d1_doses.tolist(),
            "doses_b": d2_doses.tolist(),
            "viability_matrix": matrix.tolist()
        })

    if known_cells_map:
        total_samples = len(first_rows)
        match_pct = (matched_count / total_samples * 100) if total_samples > 0 else 0
        print("\n" + "=" * 60)
        print("CELL-LINE EXTRACTION & PARSING REPORT")
        print("=" * 60)
        print(f"Total Raw Unique Samples : {total_samples}")
        print(f"Successfully Matched     : {matched_count} ({match_pct:.2f}%)")
        print(f"Unmatched Samples        : {unmatched_count} ({100 - match_pct:.2f}%)")
        print(f"Unique Matched Cell Lines: {len(unique_cells)}")
        if unmatched_names:
            print(f"Sample Unmatched Names   : {list(unmatched_names)[:5]}")
        print("=" * 60 + "\n")
        
    return data_list


def load_synergy_dataset(zip_or_csv_path: str = "data/DrugCombination_with_SMILES.zip", known_cells: set = None) -> List[Dict[str, Any]]:
    """Load drug combination dataset from ZIP or CSV archive.

    Args:
        zip_or_csv_path: Path to dataset ZIP archive or CSV file.
        known_cells: Optional set of known cell lines.

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
            
        return parse_dataframe_to_records(df, known_cells)
    except Exception as e:
        print(f"Error loading dataset: {e}")
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
        self._dynamic_cache = {} # Thread-safe cache for dynamic feature generation
        
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
            if smiles_a not in self._dynamic_cache:
                m_a, d_a, _ = self.preprocessor.process_smiles(smiles_a)
                i_a, mk_a = self.tokenizer.tokenize(smiles_a)
                self._dynamic_cache[smiles_a] = (m_a, d_a, i_a, mk_a)
            morgan_a, desc_a, ids_a, mask_a = self._dynamic_cache[smiles_a]

        if feat_b is not None:
            morgan_b = feat_b["morgan"] if isinstance(feat_b, dict) else feat_b[0]
            desc_b = feat_b["descriptors"] if isinstance(feat_b, dict) else feat_b[1]
            ids_b = feat_b["token_ids"] if isinstance(feat_b, dict) else feat_b[2]
            mask_b = feat_b["token_mask"] if isinstance(feat_b, dict) else feat_b[3]
        else:
            if smiles_b not in self._dynamic_cache:
                m_b, d_b, _ = self.preprocessor.process_smiles(smiles_b)
                i_b, mk_b = self.tokenizer.tokenize(smiles_b)
                self._dynamic_cache[smiles_b] = (m_b, d_b, i_b, mk_b)
            morgan_b, desc_b, ids_b, mask_b = self._dynamic_cache[smiles_b]
        
        # Get biological profile
        norm_cell = re.sub(r'[^A-Z0-9]', '', str(cell_name).upper())
        cell_vec = self.cell_line_features.get(cell_name, self.cell_line_features.get(norm_cell))
        if cell_vec is None:
            raise KeyError(f"Cell line features for '{cell_name}' not found. Do not zero-pad.")
        assert cell_vec.shape[-1] == 976, f"Expected 976-dim gene expression vector, got {cell_vec.shape[-1]}"
            
        res_dict = {
            "drug_a_ids": _to_tensor(ids_a, dtype=torch.long),
            "drug_a_mask": _to_tensor(mask_a, dtype=torch.float32),
            "drug_a_morgan": _to_tensor(morgan_a, dtype=torch.float32),
            "drug_a_desc": _to_tensor(desc_a, dtype=torch.float32),
            
            "drug_b_ids": _to_tensor(ids_b, dtype=torch.long),
            "drug_b_mask": _to_tensor(mask_b, dtype=torch.float32),
            "drug_b_morgan": _to_tensor(morgan_b, dtype=torch.float32),
            "drug_b_desc": _to_tensor(desc_b, dtype=torch.float32),
            
            "cell_line": _to_tensor(cell_vec, dtype=torch.float32),
            "doses_a": _to_tensor(doses_a, dtype=torch.float32),
            "doses_b": _to_tensor(doses_b, dtype=torch.float32),
            "viability": _to_tensor(viability, dtype=torch.float32)
        }
        
        # Include optional Hill parameters for auxiliary supervision if present
        for p in ["e1", "e2", "e3", "log_c1", "log_c2", "h1", "h2", "alpha"]:
            if p in item:
                res_dict[p] = _to_tensor([float(item[p])], dtype=torch.float32)
                
        return res_dict


def load_precomputed_drug_features(feature_path: str) -> Dict[str, Dict[str, Any]]:
    """Load a saved drug feature store from .pt or .pkl."""
    if not os.path.exists(feature_path):
        return {}

    try:
        if feature_path.endswith(".pt"):
            return torch.load(feature_path, map_location="cpu", weights_only=True)
        if feature_path.endswith(".pkl"):
            with open(feature_path, "rb") as f:
                return pickle.load(f)
    except Exception:
        return {}

    return {}

