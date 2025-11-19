"""Train a model on the FinQA dataset using the QSTAR approach."""

from __future__ import annotations

import os
import argparse
import json
import time
from pathlib import Path
from accelerate.utils import set_seed
from datasets import Dataset
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

from src.eval_finqa import compute_accuracy
from src.load_data import load_finqa_split_text_with_answer
from src.load_model import load_qstar_for_training


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the QSTAR trainer.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including [...].
    """
    parser = argparse.ArgumentParser(description="Run FinR1 baseline on the FinQA dataset.")
    parser.add_argument("--dataset-dir", type=Path, default="data")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--full-batch-size", type=int, default=1)
    parser.add_argument("--eval-and-logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-ahead", type=int, default=4)
    parser.add_argument("--n-ahead-talk", type=int, default=4)
    parser.add_argument("--n-passes", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gumbel-temperature", type=float, default=1.0)
    parser.add_argument("--root-prefix", type=str, default="..")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tokenize_and_format(dataset, tokenizer, batch_size=8):
    """Tokenize a dataset in batches to reduce memory footprint."""
    def preprocess(batch):
        tokens = tokenizer(batch["text"], padding=True, return_tensors="pt")
        tokens["labels"] = tokens["input_ids"]
        return tokens

    return dataset.map(preprocess, batched=True, batch_size=batch_size)


def main() -> None:
    """Run the baseline pipeline end-to-end.

    Loads the requested FinQA split, runs generation with the baseline
    model, computes accuracy against ground-truth answers, and optionally
    writes raw generations and a detailed predictions file to disk.
    """
    # Parse command-line arguments
    args = parse_args()
    set_seed(args.seed)
    cache_dir = f"{args.root_prefix}/.cache/qstar"
    batch_size = args.full_batch_size // args.n_passes
    gradient_accumulation_steps = args.full_batch_size // batch_size
    run_id = int(time.time())

    # Load models
    model, tokenizer = load_qstar_for_training(args.model_name, params=args)
    
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params_trainable}")
    
    # Load datasets
    train_samples = load_finqa_split_text_with_answer(args.dataset_dir, split="train")
    if args.train_limit:
        train_samples = train_samples[: args.train_limit]
    train_dataset = tokenize_and_format(Dataset.from_list(train_samples), tokenizer, batch_size=8)

    eval_samples = load_finqa_split_text_with_answer(args.dataset_dir, split="dev")
    if args.eval_limit:
        eval_samples = eval_samples[: args.eval_limit]
    eval_dataset = tokenize_and_format(Dataset.from_list(eval_samples), tokenizer, batch_size=8)

    print(f"Loaded {len(train_samples)} samples from FinQA train split.")
    print(f"Loaded {len(eval_samples)} samples from FinQA train split.")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"{cache_dir}/{run_id}",
        learning_rate=args.learning_rate,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        label_names=["labels"],
        logging_steps=args.eval_and_logging_steps,
        eval_steps=args.eval_and_logging_steps,
        eval_strategy="steps",
        save_steps=args.save_steps,
        run_name=f"n={args.n_ahead}_nt={args.n_ahead_talk}_np={args.n_passes}",
    )

    # Set up Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        return_tensors="pt",
    )

    class MemTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch):
            out = super().training_step(model, inputs, num_items_in_batch)
            torch.cuda.synchronize()
            print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
            print("Cached:", torch.cuda.memory_reserved() / 1024**2, "MB")
            return out

    # Set up trainer
    trainer = MemTrainer(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics,
        model=model,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
