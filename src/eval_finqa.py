"""FinQA evaluation utilities.

This module provides functions to normalize and compare model predictions
against FinQA ground-truth answers. It supports:
- Whitespace normalization and case-insensitive text comparison.
- Robust numeric parsing that strips currency symbols, percent signs, and
  thousand separators; also handles forms like "(123)" → -123 and
  FinQA-style constants (e.g., ``const_m1``).
- Magnitude alignment to tolerate scientific-scale mismatches when both
  sides are numeric.
- Extract answer from model outputs wrapped in ``\\boxed{...}``.

Key entry point:
- ``compute_accuracy``: computes FinQA-style execution accuracy and also
  returns the parsed predictions and a match mask for downstream analysis.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional, List, Tuple



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
                return float(cleaned[:-1])
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


def _normalize_magnitude(num1: float, num2: float) -> List[float]:
    if not math.isfinite(num1) or not math.isfinite(num2) or num1 == 0 or num2 == 0:
        return [num1, num2]

    mag1 = math.floor(math.log10(abs(num1)))
    mag2 = math.floor(math.log10(abs(num2)))
    diff = mag2 - mag1

    if abs(diff) >= 1:
        if diff > 0:
            num1 *= 10**diff
        else:
            num1 *= 10**(-diff)

    return [num1, num2] 


def _extract_numeric_candidate(text: str) -> Optional[float]:
    """Attempt to recover a numeric answer from free-form text."""
    num = _str_to_num(text)
    if num is not None:
        return num

    sanitized = _strip_currency(text).replace(",", "")
    matches = list(NUM_RE.finditer(sanitized))
    for match in reversed(matches):
        candidate = match.group()
        num = _str_to_num(candidate)
        if num is not None:
            return num
    return None


def _extract_boxed_prediction(pred: str) -> Optional[float]:
    """Extract numerical prediction from model output."""
    if "\\boxed{" in pred:
        pred = pred.split("\\boxed{")[1].split("}")[0].strip().lower()
        pred_num = _extract_numeric_candidate(pred)
        if pred_num is not None:
            return pred_num
    return pred


def _answers_match(pred: str, gold: str, *, atol: float = 5e-1, rtol: float = 1e-3) -> bool:
    pred_num = pred
    gold_num = _extract_numeric_candidate(gold)

    if not isinstance(pred_num, str) and gold_num is not None:
        pred_num, gold_num = _normalize_magnitude(pred_num, gold_num)
        return math.isclose(pred_num, gold_num, rel_tol=rtol, abs_tol=atol)

    return _normalize_whitespace(pred).lower() == _normalize_whitespace(gold).lower()


def compute_accuracy(generations: Iterable[str], references: Iterable[str]) -> float:
    """Compute FinQA-style execution accuracy."""
    preds = [_extract_boxed_prediction(gen) for gen in list(generations)]
    refs = list(references)
    if not preds or len(preds) != len(refs):
        raise ValueError("Prediction and reference collections must be non-empty and of equal length.")

    matches = [1 if _answers_match(pred, ref) else 0 for pred, ref in zip(preds, refs)]
    return sum(matches) / len(preds), preds, matches
