"""MoLFormer backbone wrapper for encoding chemical fragments."""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class MockMoLFormer(nn.Module):
    """Fast deterministic mock for MoLFormer backbone used in unit tests."""

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        # A simple linear projection from hash/character counts or random embed
        self.char_embed = nn.Embedding(256, hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Any:
        # input_ids: (B, seq_len)
        safe_ids = torch.clamp(input_ids, 0, 255)
        embeds = self.char_embed(safe_ids)  # (B, seq_len, 768)
        return type("Output", (), {"last_hidden_state": embeds})()


class MoLFormerWrapper(nn.Module):
    """Wraps pretrained MoLFormer model for encoding SMILES fragments into 768-D vectors."""

    def __init__(
        self,
        model_name_or_path: str = "ibm/MoLFormer-XL-both-10pct",
        hidden_dim: int = 768,
        mock: bool = False,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.hidden_dim = hidden_dim
        self.mock = mock

        self.tokenizer = None
        if self.mock:
            self.model = MockMoLFormer(hidden_dim=hidden_dim)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    def freeze(self) -> None:
        """Freeze all backbone parameters."""
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        for p in self.model.parameters():
            p.requires_grad = True

    def encode_smiles_list(
        self,
        smiles_list: List[str],
        device: torch.device,
        max_length: int = 128,
    ) -> torch.Tensor:
        """Encode a list of SMILES strings into pooled fragment representations.

        Args:
            smiles_list: List of K SMILES strings.
            device: Target torch device.
            max_length: Max token length for tokenizer.

        Returns:
            Tensor of shape (K, 768).
        """
        if not smiles_list:
            return torch.zeros((0, self.hidden_dim), device=device)

        if self.mock or self.tokenizer is None:
            # Deterministic tokenization for mock
            max_len = min(max_length, max(len(s) for s in smiles_list) + 2)
            input_ids = torch.zeros((len(smiles_list), max_len), dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(smiles_list), max_len), dtype=torch.float32, device=device)
            for i, s in enumerate(smiles_list):
                chars = [ord(c) % 250 + 1 for c in s[:max_len - 2]]
                tokens = [1] + chars + [2]  # [CLS] ... [SEP]
                input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
                attention_mask[i, :len(tokens)] = 1.0

            out = self.model(input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state  # (K, seq_len, 768)
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1)
            sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-6)
            pooled = sum_hidden / sum_mask
            return pooled

        # Production HuggingFace tokenizer
        encoded = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (K, seq_len, 768)

        # Masked mean pooling over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-6)
        pooled = sum_hidden / sum_mask
        return pooled
