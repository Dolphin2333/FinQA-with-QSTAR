"""
Unified evaluation script for FinQA generations.

This script compares model-generated predictions (from run_baseline.py)
against FinQA ground truth answers. It computes:

- Exact-match accuracy
- Boxed answer formatting accuracy
- Average & range of rationale length (if <think>...</think> exists)
- Lists mismatches for inspection

Usage:
    python src/evaluate.py --pred outputs/finr1_P1_20.json \
                           --samples data/FinQA/test.json
"""

import json
import argparse
import re
from pathlib import Path
from statistics import mean


# ------------------------------------------------------------
# EXTRACT BOXED ANSWER
# ------------------------------------------------------------

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")

def extract_boxed(text: str) -> str | None:
    """Return the content inside \\boxed{...} or None."""
    match = BOXED_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


# ------------------------------------------------------------
# EXTRACT THINK LENGTH
# ------------------------------------------------------------

def extract_reasoning_length(text: str) -> int:
    """Count tokens inside <think>...</think> (approx via word count)."""
    if "<think>" not in text or "</think>" not in text:
        return 0
    try:
        inner = text.split("<think>")[1].split("</think>")[0]
        return len(inner.split())
    except:
        return 0


# ------------------------------------------------------------
# LOAD JSON FILES
# ------------------------------------------------------------

def load_predictions(pred_path: Path):
    """Loads predictions.json from run_baseline.py."""
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = {}
    for item in data:
        preds[item["id"]] = {
            "generation": item["generation"],
            "prediction_text": item["prediction"],
            "ground_truth": item["ground_truth"],
        }
    return preds


def load_samples(samples_path: Path):
    """Loads Test/Dev FinQA data."""
    with open(samples_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def compute_accuracy(preds):
    total = len(preds)
    correct = sum(1 for _, p in preds.items() if p["match"])
    return correct / total if total > 0 else 0


def compute_format_accuracy(preds):
    total = len(preds)
    correct = 0
    for _, p in preds.items():
        if extract_boxed(p["generation"]) is not None:
            correct += 1
    return correct / total


def compute_rationale_stats(preds):
    lengths = [p["reasoning_length"] for _, p in preds.items()]
    if not lengths:
        return 0, 0, 0
    return mean(lengths), min(lengths), max(lengths)


# ------------------------------------------------------------
# MAIN EVALUATION LOGIC
# ------------------------------------------------------------

def evaluate(pred_path: Path, sample_path: Path):

    raw_preds = load_predictions(pred_path)
    raw_samples = load_samples(sample_path)

    # Build lookup for ground truths in FinQA format
    gt_lookup = {item["id"]: item["answer"] for item in raw_samples}

    processed = {}

    # Build processed dataset
    for sample_id, item in raw_preds.items():
        gt = gt_lookup.get(sample_id, None)
        gen = item["generation"]
        boxed = extract_boxed(gen)
        reasoning_len = extract_reasoning_length(gen)

        match = False
        if boxed is not None and gt is not None:
            # Normalize both values
            try:
                match = float(boxed) == float(gt)
            except:
                match = boxed.strip() == str(gt).strip()

        processed[sample_id] = {
            "ground_truth": gt,
            "prediction": boxed,
            "match": match,
            "generation": gen,
            "reasoning_length": reasoning_len,
        }

    # Compute metrics
    acc = compute_accuracy(processed)
    fmt_acc = compute_format_accuracy(processed)
    mean_len, min_len, max_len = compute_rationale_stats(processed)

    # Print summary table
    print("\n================ Evaluation Summary ================")
    print(f"Prediction file : {pred_path}")
    print(f"Sample file     : {sample_path}")
    print(f"Total samples   : {len(processed)}")
    print("---------------------------------------------------")
    print(f"Accuracy                    : {acc*100:.2f}%")
    print(f"Formatting accuracy         : {fmt_acc*100:.2f}%")
    print(f"Avg rationale length        : {mean_len:.1f} tokens")
    print(f"Rationale length range      : {min_len} – {max_len}")
    print("===================================================\n")

    # Print mismatches
    print("MISMATCHES:")
    for sid, item in processed.items():
        if not item["match"]:
            print(f"- ID {sid}: predicted={item['prediction']}  |  gt={item['ground_truth']}")

    return processed


# ------------------------------------------------------------
# CLI ENTRY
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True,
                        help="Path to predictions JSON (from run_baseline.py)")
    parser.add_argument("--samples", type=Path, required=True,
                        help="Path to FinQA split JSON (test.json or dev.json)")
    args = parser.parse_args()

    evaluate(args.pred, args.samples)
