"""Utilities for loading and parsing the FinQA dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


def _program_tokenization(program_text: str) -> List[str]:
    """Convert the FinQA program string into a list of tokens (copy of official logic)."""
    program_text = program_text or ""
    tokens = []
    current = ""

    for char in program_text.split(", "):
        cur_tok = ""
        for c in char:
            if c == ")":
                if cur_tok:
                    tokens.append(cur_tok)
                    cur_tok = ""
            cur_tok += c
            if c in ("(", ")"):
                tokens.append(cur_tok)
                cur_tok = ""
        if cur_tok:
            tokens.append(cur_tok)
    tokens.append("EOF")
    return tokens


def _flatten_text(field) -> str:
    if isinstance(field, list):
        return " ".join(str(item).strip() for item in field if str(item).strip())
    if field is None:
        return ""
    return str(field)


@dataclass
class FinQASample:
    """Container for one FinQA example with rich context."""

    sample_id: str
    question: str
    answer: str
    program_text: str
    program_tokens: Sequence[str]
    table: Sequence[Sequence[str]]
    pre_text: str
    post_text: str
    supporting_facts: Sequence[str]
    metadata: dict


def load_finqa_split(dataset_dir: Path, split: str) -> List[FinQASample]:
    """Load a FinQA split (train/dev/test) from ``dataset_dir``."""

    dataset_dir = Path(dataset_dir)
    split_path = dataset_dir / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Could not find FinQA split '{split}' at {split_path}. "
            "Make sure you've copied the dataset files from https://github.com/czyssrs/FinQA."
        )

    with split_path.open("r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    examples: List[FinQASample] = []
    for entry in raw_samples:
        qa_blob = entry.get("qa", {}) or {}
        program_text = qa_blob.get("program", "")
        example = FinQASample(
            sample_id=str(entry.get("id", "")),
            question=qa_blob.get("question", ""),
            answer=str(qa_blob.get("answer", "")),
            program_text=program_text,
            program_tokens=_program_tokenization(program_text),
            table=[list(map(str, row)) for row in entry.get("table", [])],
            pre_text=_flatten_text(entry.get("pre_text", "")),
            post_text=_flatten_text(entry.get("post_text", "")),
            supporting_facts=list(qa_blob.get("supporting_facts") or []),
            metadata=entry,
        )
        examples.append(example)
    return examples


def iter_answers(samples: Iterable[FinQASample]) -> Iterable[str]:
    """Yield ground-truth answers as strings."""
    for sample in samples:
        yield sample.answer
