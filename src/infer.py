"""Baseline inference utilities."""

from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .load_data import FinQASample
from .table_utils import table_to_text


def build_prompt(sample: FinQASample) -> str:
    """Format a FinQA prompt for autoregressive inference."""
    context_parts: List[str] = []

    if sample.model_input:
        context_parts.extend([text for text in sample.model_input if text])
    else:
        pre = sample.pre_text.strip()
        post = sample.post_text.strip()
        table_text = table_to_text(sample.table)

        if pre:
            context_parts.append(pre)
        if table_text:
            context_parts.append("Table:\n" + table_text)
        if post:
            context_parts.append(post)

    context_block = "\n\n".join(context_parts).strip()
    if context_block:
        context_block += "\n\n"

    instructions = (
        "You are a financial reasoning assistant. "
        "Read the context carefully and compute the final numeric answer. "
        "If the answer is yes/no, respond with that word. "
        "Otherwise respond with a single numeric value. "
        "Do not include explanations after the final answer."
    )

    prompt_sections = [
        context_block + instructions if context_block else instructions,
        f"Question: {sample.question}",
        "Answer:"
    ]
    return "\n".join(section for section in prompt_sections if section)


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
