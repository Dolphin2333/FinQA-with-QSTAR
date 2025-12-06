"""Evaluate training checkpoint."""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
from accelerate.utils import set_seed
from peft import PeftModel

from src.eval_finqa import compute_accuracy, compute_formatting_accuracy, compute_rationale_stats
from src.infer import run_inference
from src.load_data import iter_answers, load_finqa_split
from src.load_model import load_qstar_for_training


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the FinQA baseline runner.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including dataset location, split, model id,
        generation settings, optional sample offset and limit, output path,
        and random seed.
    """
    parser = argparse.ArgumentParser(description="Run FinR1 baseline on the FinQA dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default="data",
        help="Path to the FinQA dataset directory containing train/dev/test JSON files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (train/dev/test). Defaults to test.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model identifier to load. Defaults to the Qwen 7B Instruct.",
    )
    parser.add_argument(
        "--peft-dir",
        type=Path,
        default="../.cache/qstar/1764960658",
        help="Hugging Face model identifier to load. Defaults to the FinR1 checkpoint.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum number of tokens generated for each answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for nucleus sampling during generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p value for nucleus sampling during generation.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty to apply during generation.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Optional offset on the samples evaluated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of samples evaluated (useful for smoke tests).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write predictions as JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument("--n-ahead", type=int, default=10)
    parser.add_argument("--n-ahead-talk", type=int, default=4)
    parser.add_argument("--n-passes", type=int, default=1)
    parser.add_argument("--root-prefix", type=str, default="..")
    return parser.parse_args()


def main() -> None:
    """Run the baseline pipeline end-to-end.

    Loads the requested FinQA split, runs generation with the baseline
    model, computes accuracy against ground-truth answers, and optionally
    writes raw generations and a detailed predictions file to disk.
    """
    args = parse_args()
    set_seed(args.seed)

    samples = load_finqa_split(args.dataset_dir, args.split)
    if args.offset:
        samples = samples[args.offset:]
    if args.limit:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} samples from FinQA {args.split} split.")

    model, tokenizer = load_qstar_for_training(args.model_name, params=args)
    model.eval()
    results = []

    for checkpoint in [f"checkpoint-{i}" for i in range(10, 61, 10)]:
        adapter_path = args.peft_dir / checkpoint
        peft_model = PeftModel.from_pretrained(model, adapter_path)

        generations = run_inference(
            peft_model,
            tokenizer,
            samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(generations)

        # Answer accuracy
        accuracy, preds, matches = compute_accuracy(generations, list(iter_answers(samples)))

        # Formatting accuracy
        fmt_acc = compute_formatting_accuracy(generations)

        # Rationale stats
        avg_len, min_len, max_len = compute_rationale_stats(generations)

        result = {
            "checkpoint": checkpoint,
            "accuracy": accuracy,
            "format_accuracy": fmt_acc,
            "rational_length_avg": avg_len,
            "rational_length_min": min_len,
            "rational_length_max": max_len
        }
        results.append(result)
        print(result)

    if args.output:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Wrote evaluation to {args.output}")


if __name__ == "__main__":
    main()
