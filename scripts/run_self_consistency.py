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
from src.evaluate import BOX_PATTERN


def _reasoning_length(text: str) -> int:
    if "<think>" in text and "</think>" in text:
        inner = text.split("<think>", 1)[1].split("</think>", 1)[0]
        return len(inner.split())
    return 0


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
    generations, candidate_paths, winning_paths = run_self_consistency(
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

    references = list(iter_answers(samples))
    # Accuracy over samples based on voted predictions
    accuracy, _, matches = compute_accuracy(generations, references)

    # Formatting accuracy over samples: 1 if the voted output contains boxed answer
    fmt_scores = [1 if "\\boxed{" in pred and pred.count("\\boxed{") == 1 else 0 for pred in generations]
    fmt_acc = sum(fmt_scores) / len(fmt_scores) if fmt_scores else 0.0

    # Rationale stats: for samples with a voted answer, use concatenated winner reasoning;
    # if no winner, use empty string placeholder.
    sample_lengths = []
    for sample_paths, winner_paths in zip(candidate_paths, winning_paths):
        paths = winner_paths if winner_paths else (sample_paths[:1] if sample_paths else [])
        if paths:
            lengths = [_reasoning_length(p) for p in paths]
            sample_lengths.append(sum(lengths) / len(lengths))
        else:
            sample_lengths.append(0.0)

    if sample_lengths:
        avg_len = sum(sample_lengths) / len(sample_lengths)
        min_len = min(sample_lengths)
        max_len = max(sample_lengths)
    else:
        avg_len = min_len = max_len = 0.0

    print("==========================================")
    print(f"Answer accuracy (per sample) : {accuracy * 100:.1f}%")
    print(f"Formatting accuracy          : {fmt_acc * 100:.1f}%")
    print(f"Rationale length (avg)       : {avg_len:.1f} tokens")
    print(f"Rationale length (min/max)   : {min_len} / {max_len}")
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
                "winning_generations": winning,
                "all_generations": paths,
                "program_text": sample.program_text,
                "pre_text": sample.pre_text,
                "post_text": sample.post_text,
                "table": sample.table,
            }
            for sample, gen, pred, match, paths, winning in zip(
                samples, generations, preds, matches, candidate_paths, winning_paths
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
