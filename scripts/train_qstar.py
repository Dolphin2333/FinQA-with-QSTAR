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
from peft import LoraConfig, get_peft_model
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
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--n-ahead", type=int, default=10)
    parser.add_argument("--n-ahead-talk", type=int, default=4)
    parser.add_argument("--n-passes", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gumbel-temperature", type=float, default=1.0)
    parser.add_argument("--root-prefix", type=str, default="..")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-id", type=int, default=1)
    return parser.parse_args()


def tokenize_and_format(dataset, tokenizer, batch_size=8):
    """Tokenize a dataset in batches to reduce memory footprint."""
    def preprocess(batch):
        tokens = tokenizer(batch["text"], padding=True, return_tensors="pt")
        tokens["labels"] = tokens["input_ids"]
        return tokens

    return dataset.map(preprocess, batched=True, batch_size=batch_size)


def count_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if (not trainable_only or p.requires_grad))


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
    gradient_accumulation_steps = args.gradient_accumulation_steps
    run_id = int(time.time())

    # Load models
    model, tokenizer = load_qstar_for_training(args.model_name, params=args)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Create lora
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    # Count number of trainable parameters
    full_params = count_params(model)
    trainable_params = count_params(peft_model, trainable_only=True)
    print(f"Number of trainable parameters: {trainable_params:,}")
    print(f"Total number of parameters    : {full_params:,}")
    
    # Load datasets
    train_samples = load_finqa_split_text_with_answer(args.dataset_dir, split="train", prompt_id=args.prompt_id)
    if args.train_limit:
        train_samples = train_samples[: args.train_limit]
    train_dataset = tokenize_and_format(Dataset.from_list(train_samples), tokenizer, batch_size=8)

    eval_samples = load_finqa_split_text_with_answer(args.dataset_dir, split="dev", prompt_id=args.prompt_id)
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
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_steps=args.save_steps,
        run_name=f"n={args.n_ahead}_nt={args.n_ahead_talk}_np={args.n_passes}",
    )

    # Set up Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=peft_model,
        padding="longest",
        return_tensors="pt",
    )

    # Set up trainer
    trainer = Trainer(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=peft_model,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
