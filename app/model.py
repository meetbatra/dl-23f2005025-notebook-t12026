# BiLSTM model definition and preprocessing utilities for MCQ scoring.

from __future__ import annotations

import re
from typing import Dict, List

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

MAX_LEN = 90
VOCAB_SIZE = 2983
EMBED_DIM = 100
HIDDEN_SIZE = 128
NUM_OPTIONS = 5
PADDING_IDX = 0
UNKNOWN_IDX = 1


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def combine_text(prompt: str, option: str) -> str:
    return f"{prompt} [SEP] {option}"


def encode(text: str, word2idx: Dict[str, int], max_len: int = MAX_LEN) -> List[int]:
    tokens = tokenize(text)
    encoded = [word2idx.get(token, UNKNOWN_IDX) for token in tokens[:max_len]]
    if len(encoded) < max_len:
        encoded.extend([PADDING_IDX] * (max_len - len(encoded)))
    return encoded


def get_lengths(x: Tensor) -> Tensor:
    return (x != 0).sum(dim=-1)


class BiLSTMScorer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        hidden_size: int = HIDDEN_SIZE,
        padding_idx: int = PADDING_IDX,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, 5, 90), got {tuple(x.shape)}")

        batch_size, num_options, seq_len = x.shape
        if num_options != NUM_OPTIONS:
            raise ValueError(f"Expected 5 options per question, got {num_options}")

        x_flat = x.reshape(batch_size * num_options, seq_len)
        if lengths is None:
            lengths = get_lengths(x)

        lengths_flat = lengths.reshape(-1).clamp(min=1).to(device="cpu")
        embedded = self.embedding(x_flat)
        packed = pack_padded_sequence(
            embedded,
            lengths_flat,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        features = self.dropout(features)
        logits = self.fc(features)
        return logits.reshape(batch_size, num_options)

