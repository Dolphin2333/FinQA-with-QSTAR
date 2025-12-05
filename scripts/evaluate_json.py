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

from src.eval_finqa import compute_accuracy, compute_formatting_accuracy, compute_rationale_stats


# ---------------------------------------------------------
# MAIN EVALUATION 
# ---------------------------------------------------------
def evaluate(pred_file: str):
    """Evaluate formatting + rationale statistics only."""
    with open(pred_file, "r", encoding="utf-8") as f:
        data = json.load(f)

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
    parser.add_argument("--json-file", type=str, required=True,
                        help="Path to predictions JSON file.")
    args = parser.parse_args()

    evaluate(args.json_file)
