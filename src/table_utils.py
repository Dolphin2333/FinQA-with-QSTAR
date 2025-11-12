"""Table-to-text helpers for FinQA.

Converts tabular data (header + rows) into short natural-language
statements so that tables can be injected into LLM prompts.

Main function:
- ``table_to_text``: emits sentences like
  "For <row0>, the <col1> is <val1>; the <col2> is <val2>." per row.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


def _clean_cell(cell: str) -> str:
    return " ".join(str(cell).split())


def table_to_text(table: Sequence[Sequence[str]]) -> str:
    """Convert a table (header + rows) into natural language statements."""
    if not table:
        return ""
    header = [_clean_cell(h) for h in table[0]]
    rows = [[_clean_cell(cell) for cell in row] for row in table[1:]]

    sentences: List[str] = []
    for row in rows:
        if not row:
            continue
        subject = row[0]
        fragments = []
        for head, cell in zip(header[1:], row[1:]):
            if head and cell:
                fragments.append(f"the {head} is {cell}")
        if fragments:
            sentences.append(f"For {subject}, " + "; ".join(fragments) + ".")
    return "\n".join(sentences)
