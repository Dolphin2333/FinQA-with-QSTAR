"""Entry point to execute the FinQA baseline pipeline."""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
from accelerate.utils import set_seed

from src.eval_finqa import compute_accuracy
from src.infer import run_inference
from src.load_data import FinQASample, iter_answers, load_finqa_split
from src.load_model import DEFAULT_MODEL_ID, load_baseline


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
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model identifier to load. Defaults to the FinR1 checkpoint.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4000,
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

    model, tokenizer = load_baseline(args.model_name)
    generations = run_inference(
        model,
        tokenizer,
        samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    if args.output:
        temp_output = args.output.with_suffix(".raw.json")
        print(f"Backing up raw generations to {temp_output}...")

        raw_serializable = [
            {"id": sample.sample_id, "generation": gen}
            for sample, gen in zip(samples, generations)
        ]
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        with temp_output.open("w", encoding="utf-8") as f:
            json.dump(raw_serializable, f, indent=2, ensure_ascii=False)
        print(f"Wrote raw generations to {temp_output}")

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

        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"Delete backup generations at {temp_output}")


if __name__ == "__main__":
    main()
