# Model loading and MCQ prediction helpers for the FastAPI service.

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .model import BiLSTMScorer, MAX_LEN, VOCAB_SIZE, combine_text, encode, get_lengths

LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E")
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "bilstm_model.pth"
VOCAB_PATH = MODELS_DIR / "word2idx.json"

_WORD2IDX: Dict[str, int] | None = None
_MODEL: BiLSTMScorer | None = None


def _require_model_files() -> None:
    missing = [path.name for path in (MODEL_PATH, VOCAB_PATH) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Model files are missing. Place bilstm_model.pth and word2idx.json in the models/ folder."
        )


def load_resources() -> BiLSTMScorer:
    global _WORD2IDX, _MODEL

    if _MODEL is not None and _WORD2IDX is not None:
        return _MODEL

    _require_model_files()

    with VOCAB_PATH.open("r", encoding="utf-8") as handle:
        _WORD2IDX = {str(key): int(value) for key, value in json.load(handle).items()}

    model = BiLSTMScorer(vocab_size=VOCAB_SIZE)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    _MODEL = model
    return model


def get_loaded_model() -> BiLSTMScorer:
    if _MODEL is None:
        return load_resources()
    return _MODEL


def get_loaded_vocab() -> Dict[str, int]:
    if _WORD2IDX is None:
        load_resources()
    assert _WORD2IDX is not None
    return _WORD2IDX


def predict(prompt: str, a: str, b: str, c: str, d: str, e: str) -> Dict[str, object]:
    model = get_loaded_model()
    word2idx = get_loaded_vocab()

    options = {"A": a, "B": b, "C": c, "D": d, "E": e}
    encoded_options: List[List[int]] = [
        encode(combine_text(prompt, option_text), word2idx, max_len=MAX_LEN)
        for option_text in options.values()
    ]

    input_tensor = torch.tensor([encoded_options], dtype=torch.long)
    lengths = get_lengths(input_tensor)

    with torch.no_grad():
        logits = model(input_tensor, lengths=lengths)
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)

    score_map = {
        label: round(float(probabilities[index].item()), 4)
        for index, label in enumerate(LABELS)
    }
    ranking = [
        label
        for label, _ in sorted(
            (
                (label, float(probabilities[index].item()))
                for index, label in enumerate(LABELS)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    return {
        "top3": ranking[:3],
        "scores": score_map,
    }
