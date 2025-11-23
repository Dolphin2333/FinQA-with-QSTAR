"""
evaluate.py — Evaluate predictions from run_baseline.py
Computes:
- exact_match accuracy
- formatting accuracy (# predictions with exactly one \boxed{...})
- rationale length statistics
"""

import json
import re
from pathlib import Path
from typing import Optional  # <-- Python 3.9 compatible


BOX_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def load_predictions(path: str):
    """Load the JSON predictions file produced by run_baseline.py."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_boxed(text: str) -> Optional[str]:
    """Extract the first \boxed{...} value."""
    matches = BOX_PATTERN.findall(text)
    return matches[0] if matches else None


def compute_formatting_accuracy(preds):
    """Check how many predictions contain exactly one \boxed{}."""
    correct = 0
    for p in preds:
        matches = BOX_PATTERN.findall(p)
        if len(matches) == 1:
            correct += 1
    return correct / len(preds)


def compute_rationale_length(pred: str) -> int:
    """Count tokens inside <think>...</think> if present."""
    if "<think>" in pred and "</think>" in pred:
        inner = pred.split("<think>")[1].split("</think>")[0]
        return len(inner.split())
    return 0


def evaluate(pred_file: str):
    """Evaluate predictions and print metrics."""
    data = load_predictions(pred_file)

    preds = [d["generation"] for d in data]
    targets = [d["ground_truth"] for d in data]

    # 1. exact-match accuracy
    correct = 0
    for p, t in zip(preds, targets):
        boxed = extract_boxed(p)
        if boxed is not None and boxed.strip() == str(t).strip():
            correct += 1

    accuracy = correct / len(preds)

    # 2. formatting accuracy
    fmt_acc = compute_formatting_accuracy(preds)

    # 3. rationale length statistics
    lengths = [compute_rationale_length(p) for p in preds]
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    print("==========================================")
    print(f"Evaluating: {pred_file}")
    print("==========================================")
    print(f"Exact-match accuracy      : {accuracy:.4f}")
    print(f"Formatting accuracy       : {fmt_acc:.4f}")
    print(f"Rationale length (avg)    : {avg_len:.1f} tokens")
    print(f"Rationale length (min/max): {min_len} / {max_len}")
    print("==========================================")

    return {
        "accuracy": accuracy,
        "fmt_acc": fmt_acc,
        "avg_len": avg_len,
        "min_len": min_len,
        "max_len": max_len,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True,
                        help="Path to predictions JSON file.")
    args = parser.parse_args()

    evaluate(args.pred)
