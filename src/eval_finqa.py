"""Evaluation utilities for FinQA baseline runs."""

from __future__ import annotations

import re
from typing import Iterable


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_answer(answer: str) -> str:
    """Normalize numeric strings for exact-match comparison."""
    answer = _WHITESPACE_RE.sub(" ", str(answer)).strip().lower()
    answer = answer.replace(",", "")
    if answer.endswith(".0"):
        answer = answer[:-2]
    return answer


def compute_accuracy(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Compute simple exact-match accuracy."""
    preds = list(predictions)
    refs = list(references)
    if not preds or len(preds) != len(refs):
        raise ValueError("Prediction and reference collections must be non-empty and of equal length.")

    correct = 0
    for pred, ref in zip(preds, refs):
        if _normalize_answer(pred) == _normalize_answer(ref):
            correct += 1
    return correct / len(preds)
