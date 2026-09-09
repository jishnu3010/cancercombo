"""Utilities for padding, masking, and batching chemical fragments."""

from __future__ import annotations

from typing import List, Tuple, Dict
import torch


def pad_fragment_strings(
    fragments_list: List[List[str]],
    max_fragments: int = 32,
    pad_str: str = "",
) -> Tuple[List[List[str]], torch.Tensor]:
    """Pad variable-length lists of fragment strings and create a boolean validity mask.

    Args:
        fragments_list: List of length B, where each item is a list of N_i fragment SMILES.
        max_fragments: Maximum fragment length to clamp / pad.
        pad_str: String representation of padding fragment.

    Returns:
        padded_frags: List of length B with exactly max_len items each.
        mask: Float or Bool Tensor of shape (B, max_len), 1.0 for valid fragments, 0.0 for padding.
    """
    batch_size = len(fragments_list)
    actual_max = max((len(f) for f in fragments_list), default=1)
    target_len = min(max_fragments, max(1, actual_max))

    padded_frags: List[List[str]] = []
    mask = torch.zeros((batch_size, target_len), dtype=torch.float32)

    for i, frags in enumerate(fragments_list):
        truncated = frags[:target_len] if frags else [pad_str]
        valid_count = len(frags[:target_len]) if frags else 0
        if valid_count == 0:
            # If completely empty, at least 1 dummy pad
            truncated = [pad_str]

        pad_needed = target_len - len(truncated)
        padded = truncated + [pad_str] * pad_needed
        padded_frags.append(padded)
        mask[i, :valid_count] = 1.0

    return padded_frags, mask
