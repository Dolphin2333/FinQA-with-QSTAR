"""Entry point for FinQA inference with self-consistency decoding."""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
from accelerate.utils import set_seed

from src.eval_finqa import compute_accuracy
from src.evaluate import compute_formatting_accuracy, compute_rationale_stats
from src.infer import run_self_consistency, extract_boxed_answers
from src.load_data import iter_answers, load_finqa_split
from src.load_model import DEFAULT_MODEL_ID, load_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FinR1 self-consistency decoding.")
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
        "--num-sequences",
        type=int,
        default=10,
        help="Number of reasoning paths to sample per question for majority voting.",
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
        help="Sampling temperature used during generation.",
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
        help="Repetition penalty applied during generation.",
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
        help="Optional limit on the number of samples evaluated.",
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
    args = parse_args()
    set_seed(args.seed)

    samples = load_finqa_split(args.dataset_dir, args.split)
    if args.offset:
        samples = samples[args.offset:]
    if args.limit:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} samples from FinQA {args.split} split.")

    model, tokenizer = load_baseline(args.model_name)
    generations, candidate_paths = run_self_consistency(
        model,
        tokenizer,
        samples,
        num_sequences=args.num_sequences,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        return_all_generations=True,
    )

    cost_multiplier = args.num_sequences
    print(
        f"Estimated inference cost multiplier vs. greedy decoding: x{cost_multiplier}"
        f" (approx. +{(cost_multiplier - 1) * 100:.0f}% tokens)."
    )

    if args.output:
        temp_output = args.output.with_suffix(".raw.json")
        print(f"Backing up raw generations to {temp_output}...")

        raw_serializable = [
            {
                "id": sample.sample_id,
                "generations": paths,
                "boxed_values": extract_boxed_answers(paths),
            }
            for sample, paths in zip(samples, candidate_paths)
        ]

        args.output.parent.mkdir(parents=True, exist_ok=True)

        with temp_output.open("w", encoding="utf-8") as f:
            json.dump(raw_serializable, f, indent=2, ensure_ascii=False)
        print(f"Wrote raw generations to {temp_output}")

    accuracy, preds, matches = compute_accuracy(generations, list(iter_answers(samples)))
    fmt_acc = compute_formatting_accuracy(generations)
    avg_len, min_len, max_len = compute_rationale_stats(generations)

    print("==========================================")
    print(f"Answer accuracy.          : {accuracy * 100:.1f}%")
    print(f"Formatting accuracy       : {fmt_acc * 100:.1f}%")
    print(f"Rationale length (avg)    : {avg_len:.1f} tokens")
    print(f"Rationale length (min/max): {min_len} / {max_len}")
    print("==========================================")

    if args.output:
        serializable = [
            {
                "id": sample.sample_id,
                "question": sample.question,
                "ground_truth": sample.answer,
                "prediction": pred,
                "match": match,
                "majority_generation": gen,
                "all_generations": paths,
                "program_text": sample.program_text,
                "pre_text": sample.pre_text,
                "post_text": sample.post_text,
                "table": sample.table,
            }
            for sample, gen, pred, match, paths in zip(
                samples, generations, preds, matches, candidate_paths
            )
        ]
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"Wrote predictions to {args.output}")

        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"Delete backup generations at {temp_output}")


if __name__ == "__main__":
    main()
