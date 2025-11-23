"""
evaluate.py — Evaluate predictions from run_baseline.py
Computes:
- exact_match accuracy (existing logic, inline)
- formatting accuracy (# predictions with exactly one \boxed{...})
- rationale length statistics
"""

import json
import re
from typing import Optional

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


# ---------------------------------------------------------
# ONLY TWO NEW FUNCTIONS AS REQUESTED
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
# MAIN EVALUATION (inline accuracy logic preserved)
# ---------------------------------------------------------

def evaluate(pred_file: str):
    """Evaluate predictions and print metrics."""
    data = load_predictions(pred_file)

    preds = [d["generation"] for d in data]
    targets = [d["ground_truth"] for d in data]

    # 1️⃣ Exact-match accuracy (existing logic preserved)
    correct = 0
    for p, t in zip(preds, targets):
        boxed = extract_boxed(p)
        if boxed is not None and boxed.strip() == str(t).strip():
            correct += 1
    accuracy = correct / len(preds)

    # 2️⃣ Formatting accuracy
    fmt_acc = compute_formatting_accuracy(preds)

    # 3️⃣ Rationale length statistics
    avg_len, min_len, max_len = compute_rationale_stats(preds)

    # Print results
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
