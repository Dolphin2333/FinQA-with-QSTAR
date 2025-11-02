"""Utilities for loading the FinQA dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class FinQASample:
    """Container for one FinQA example."""

    question: str
    answer: str
    program: Sequence[str]
    evidence: Sequence[str]
    metadata: dict


def load_finqa_split(dataset_dir: Path, split: str) -> List[FinQASample]:
    """Load a FinQA split (train/dev/test) from ``dataset_dir``.

    Parameters
    ----------
    dataset_dir:
        Path pointing to the root of the FinQA dataset repository.
        We expect JSON files following the official naming convention,
        e.g. ``train.json``, ``dev.json``, and ``test.json``.
    split:
        Dataset split name. Common values are ``\"train\"``, ``\"dev\"``, ``\"test\"``.

    Returns
    -------
    List[FinQASample]
        Parsed samples with question text, ground-truth answer, program steps, and metadata.
    """
    dataset_dir = Path(dataset_dir)
    split_path = dataset_dir / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Could not find FinQA split '{split}' at {split_path}. "
            "Make sure you've copied the dataset files from https://github.com/czyssrs/FinQA."
        )

    with split_path.open("r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    samples: List[FinQASample] = []
    for example in raw_samples:
        question = example.get("question", "")
        answer = example.get("answer", "")
        program = example.get("program", []) or []
        evidence = example.get("evidence", []) or []
        samples.append(
            FinQASample(
                question=question,
                answer=str(answer),
                program=list(program),
                evidence=list(evidence),
                metadata=example,
            )
        )
    return samples


def iter_questions(samples: Iterable[FinQASample]) -> Iterable[str]:
    """Yield question text from a collection of :class:`FinQASample` objects."""
    for example in samples:
        yield example.question


def iter_answers(samples: Iterable[FinQASample]) -> Iterable[str]:
    """Yield ground-truth answers as strings."""
    for example in samples:
        yield example.answer
