"""Baseline inference utilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .load_data import FinQASample


def build_prompt(sample: FinQASample) -> str:
    """Format a FinQA prompt for autoregressive inference."""
    context_lines = "\n".join(sample.evidence)
    context_section = f"\nContext:\n{context_lines}\n" if context_lines else ""
    return f"Question: {sample.question}{context_section}\nAnswer:"


@torch.inference_mode()
def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: Sequence[FinQASample],
    *,
    max_new_tokens: int = 64,
    temperature: float | None = None,
    top_p: float | None = None,
) -> List[str]:
    """Generate answers for a list of :class:`FinQASample` inputs."""
    device = next(model.parameters()).device
    predictions: List[str] = []

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature is not None or top_p is not None,
    )
    if temperature is not None:
        generation_kwargs["temperature"] = temperature
    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    for sample in samples:
        prompt = build_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][inputs.input_ids.shape[-1] :]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predictions.append(prediction)

    return predictions
