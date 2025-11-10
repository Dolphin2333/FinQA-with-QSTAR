"""Entry point to execute the FinQA baseline pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from accelerate.utils import set_seed

from src.eval_finqa import compute_accuracy
from src.infer import run_inference
from src.load_data import FinQASample, iter_answers, load_finqa_split
from src.load_model import DEFAULT_MODEL_ID, load_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FinR1 baseline on the FinQA dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the FinQA dataset directory containing train/dev/test JSON files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Dataset split to evaluate (train/dev/test). Defaults to dev.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model identifier to load. Defaults to the FinR1 checkpoint.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)

    samples = load_finqa_split(args.dataset_dir, args.split)
    if args.limit:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} samples from FinQA {args.split} split.")

    model, tokenizer = load_baseline(args.model_name)
    generations = run_inference(
        model,
        tokenizer,
        samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    accuracy, preds, matches = compute_accuracy(generations, list(iter_answers(samples)))
    print(f"Accuracy: {accuracy * 100:.2f}% ({accuracy:.4f})")

    if args.output:
        serializable = [
            {
                "id": sample.sample_id,
                "question": sample.question,
                "ground_truth": sample.answer,
                "prediction": pred,
                "match": match,
                "generation": gen,
                "program_text": sample.program_text,
                "pre_text": sample.pre_text,
                "post_text": sample.post_text,
                "table": sample.table,
            }
            for sample, gen, pred, match in zip(samples, generations, preds, matches)
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
