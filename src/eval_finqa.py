"""Evaluation utilities for FinQA baseline runs (aligned with official scripts)."""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional


NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text)).strip()


def _strip_currency(text: str) -> str:
    return (
        text.replace("$", "")
        .replace("usd", "")
        .replace("eur", "")
        .replace("£", "")
        .replace("¥", "")
    )


def _str_to_num(text: str) -> Optional[float]:
    """Port of FinQA's ``str_to_num`` with minor extensions."""
    if text is None:
        return None

    cleaned = _strip_currency(text).replace(",", "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]

    try:
        return float(cleaned)
    except ValueError:
        if cleaned.endswith("%"):
            try:
                return float(cleaned[:-1]) / 100.0
            except ValueError:
                return None
        if cleaned.lower().startswith("const_"):
            token = cleaned.lower().replace("const_", "")
            if token == "m1":
                token = "-1"
            try:
                return float(token)
            except ValueError:
                return None
    return None


def _extract_numeric_candidate(text: str) -> Optional[float]:
    """Attempt to recover a numeric answer from free-form text."""
    num = _str_to_num(text)
    if num is not None:
        return num

    sanitized = _strip_currency(text).replace(",", "")
    matches = list(NUM_RE.finditer(sanitized))
    for match in reversed(matches):
        candidate = match.group()
        suffix_index = match.end()
        if suffix_index < len(sanitized) and sanitized[suffix_index] == "%":
            candidate = candidate + "%"
        num = _str_to_num(candidate)
        if num is not None:
            return num
    return None


def _answers_match(pred: str, gold: str, *, atol: float = 1e-2, rtol: float = 1e-3) -> bool:
    pred_num = _extract_numeric_candidate(pred)
    gold_num = _extract_numeric_candidate(gold)

    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=rtol, abs_tol=atol)

    return _normalize_whitespace(pred).lower() == _normalize_whitespace(gold).lower()


def compute_accuracy(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Compute FinQA-style execution accuracy."""
    preds = list(predictions)
    refs = list(references)
    if not preds or len(preds) != len(refs):
        raise ValueError("Prediction and reference collections must be non-empty and of equal length.")

    matched = sum(1 for pred, ref in zip(preds, refs) if _answers_match(pred, ref))
    return matched / len(preds)
