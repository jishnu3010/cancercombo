import torch
from torch.utils.data import Dataset
import numpy as np
from preprocessor import MolecularPreprocessor
from typing import List, Dict, Tuple, Any

class SMILESTokenizer:
    """Character-level tokenizer for SMILES strings with PAD/UNK vocabulary."""
    
    def __init__(self, max_len: int = 128):
        self.max_len = max_len
        chars = ['[PAD]', '[UNK]', 'C', 'O', 'N', '=', '(', ')', '[', ']', 'c', 'o', 'n', 'S', 's', 'F', 'Cl', 'Br', 'I', 'H', '@', '/', '\\', '+', '-', '1', '2', '3', '4', '5', '6', '7', '#', '.']
        self.vocab = {char: idx for idx, char in enumerate(chars)}
        self.pad_idx = self.vocab['[PAD]']
        self.unk_idx = self.vocab['[UNK]']

    def tokenize(self, smiles: str) -> Tuple[List[int], List[int]]:
        """Tokenize a SMILES string character by character, mapping to indices.

        Args:
            smiles: Raw SMILES string.

        Returns:
            Tuple[List[int], List[int]]: Token IDs and the attention mask.
        """
        tokens = []
        i = 0
        while i < len(smiles):
            # Check for two-character symbols
            if i + 1 < len(smiles) and smiles[i:i+2] in ['Cl', 'Br']:
                tokens.append(smiles[i:i+2])
                i += 2
            else:
                tokens.append(smiles[i])
                i += 1
                
        ids = [self.vocab.get(t, self.unk_idx) for t in tokens]
        
        # Padding & Truncation
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            attention_mask = [1] * self.max_len
        else:
            padding_len = self.max_len - len(ids)
            attention_mask = [1] * len(ids) + [0] * padding_len
            ids = ids + [self.pad_idx] * padding_len
            
        return ids, attention_mask


class DrugComboDataset(Dataset):
    """PyTorch Dataset class representing cell line and drug combination inputs."""
    
    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        cell_line_features: Dict[str, np.ndarray],
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
        
        # Calculate chemical representations
        morgan_a, desc_a, _ = self.preprocessor.process_smiles(smiles_a)
        morgan_b, desc_b, _ = self.preprocessor.process_smiles(smiles_b)
        
        ids_a, mask_a = self.tokenizer.tokenize(smiles_a)
        ids_b, mask_b = self.tokenizer.tokenize(smiles_b)
        
        # Get biological profile
        cell_vec = self.cell_line_features.get(cell_name)
        if cell_vec is None:
            cell_vec = np.zeros(20000, dtype=np.float32)
            
        return {
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
