"""
evaluate.py — Evaluate predictions from run_baseline.py

Computes ONLY:
- formatting accuracy (# predictions with exactly one \boxed{...})
- rationale length statistics

Accuracy is computed in run_baseline.py, NOT here.
"""

import json
import re
from typing import Optional

from eval_finq import compute_accuracy


BOX_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


# ---------------------------------------------------------
# LOAD / UTILS
# ---------------------------------------------------------
def load_predictions(path: str):
    """Load the JSON predictions file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_boxed(text: str) -> Optional[str]:
    """Extract the first \boxed{...} value."""
    matches = BOX_PATTERN.findall(text)
    return matches[0] if matches else None


# ---------------------------------------------------------
# TWO NEW FUNCTIONS AS REQUIRED
# ---------------------------------------------------------
def compute_formatting_accuracy(preds):
    """Percentage of predictions containing exactly ONE valid \\boxed{...}."""
    correct = 0
    for p in preds:
        matches = BOX_PATTERN.findall(p)
        if len(matches) == 1:
            correct += 1
    return correct / len(preds)


def compute_rationale_stats(preds):
    """Return (avg_len, min_len, max_len) for rationale lengths."""
    lengths = []
    for p in preds:
        if "<think>" in p and "</think>" in p:
            inner = p.split("<think>")[1].split("</think>")[0]
            lengths.append(len(inner.split()))
        else:
            lengths.append(0)

    avg_len = sum(lengths) / len(lengths)
    return avg_len, min(lengths), max(lengths)


# ---------------------------------------------------------
# MAIN EVALUATION 
# ---------------------------------------------------------
def evaluate(pred_file: str):
    """Evaluate formatting + rationale statistics only."""
    data = load_predictions(pred_file)

    preds = [d["generation"] for d in data]
    true = [d["ground_truth"] for d in data]

    # Answer accuracy
    accuracy, _, _ = compute_accuracy(preds, true)

    # Formatting accuracy
    fmt_acc = compute_formatting_accuracy(preds)

    # Rationale stats
    avg_len, min_len, max_len = compute_rationale_stats(preds)

    print("==========================================")
    print(f"Evaluating: {pred_file}")
    print("==========================================")
    print(f"Answer accuracy.          : {accuracy * 100:.1f}%")
    print(f"Formatting accuracy       : {fmt_acc * 100:.1f}%")
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
